import logging
import random
import re

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from transformers import LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from minigpt4.common.utils import POINT_TOKEN
from minigpt4.conversation.conversation import StoppingCriteriaSub
# from minigpt4.models.modeling_llama import LlamaForCausalLM
from minigpt4.models.modeling_llama_addtokens import LlamaForCausalLM


@registry.register_model("minigpt_v2_pose")
class MiniGPTv2Pose(MiniGPTBase):
    """
    MiniGPT-v2 model for pose estimation
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_v2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=448,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            llama_model="",
            prompt_template='[INST] {} [/INST]',
            max_txt_len=300,
            end_sym='\n',
            lora_r=64,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            num_keypoints=17, # number of predicted keypoints
            point_loss_alpha=10, # weight of point regression loss
            half_freeze=False,
            tune_posembed=False,
            tune_layernorm=False,
            use_vit_adapter=False,
            wo_task_identifier=False,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            half_freeze=half_freeze,
            tune_posembed=tune_posembed,
            tune_layernorm=tune_layernorm,
            use_vit_adapter=use_vit_adapter
        )

        for name, param in self.llama_model.named_parameters():
            if any([x in name for x in ["lm_head", "embed_add_tokens"]]):
            # if any([x in name for x in ["lm_head", "embed_tokens"]]):
                # param = param.type(torch.float32)
                param.requires_grad = True

        # initialize ViT-to-LLM projection layer
        img_f_dim = self.visual_encoder.num_features * 4
        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )

        # initialize point embedding projection layer
        in_dim = self.llama_model.config.hidden_size
        out_dim = num_keypoints * 2
        point_proj_modules = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        ]
        self.point_proj = nn.ModuleList([nn.Sequential(*point_proj_modules)])

        self.chat_template = chat_template
        self.point_loss_alpha = point_loss_alpha
        # self.wo_task_identifier = wo_task_identifier
        self.wo_task_identifier = True

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()
    
    def init_llm(self, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                 lora_target_modules=["q_proj","v_proj"], **lora_kargs):
        logging.info('Loading LLAMA')
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        llama_tokenizer.pad_token = "$$"

        if low_resource:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': low_res_device}
            )
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )

        # add '[POINT]' token to large language model
        llama_tokenizer.add_tokens(POINT_TOKEN)
        self.point_token_id = llama_tokenizer(POINT_TOKEN, add_special_tokens=False).input_ids[0]
        llama_model.add_token_embeddings(1)
        # llama_model.resize_token_embeddings(len(llama_tokenizer))

        if lora_r > 0:
            llama_model = prepare_model_for_int8_training(llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            llama_model = get_peft_model(llama_model, loraconfig)

            # llama_model.print_trainable_parameters()

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False
        logging.info('Loading LLAMA Done')
        return llama_model, llama_tokenizer

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

            inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def embed_tokens(self, token_ids):
        old_vocab_size = self.llama_model.config.vocab_size
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            llamamodel = self.llama_model.base_model.model.model
        else:
            llamamodel = self.llama_model.base_model
        if llamamodel.extra_tokens:
            token_id_old_mask = torch.ones_like(token_ids, device=token_ids.device, dtype=token_ids.dtype)
            token_id_old_mask[token_ids >= old_vocab_size] = 0
            token_id_new_mask = 1 - token_id_old_mask
            embeds_old = llamamodel.embed_tokens(token_ids * token_id_old_mask)
            embeds_new = llamamodel.embed_tokens((token_ids - old_vocab_size) * token_id_new_mask)
            embeds = embeds_old * token_id_old_mask.unsqueeze(-1) + embeds_new * token_id_new_mask.unsqueeze(-1)
        else:
            embeds = llamamodel.embed_tokens(token_ids)
        
        return embeds
    
    def forward(self, samples, reduction='mean', debug_mode=False):
        if self.wo_task_identifier: # remove the task identifier
            key = 'instruction_input' if 'instruction_input' in samples.keys() else 'conv_q'
            for i in range(len(samples[key])):
                inst = samples[key][i]
                inst = re.sub(r'\[[a-z\s]*\]', '', inst)
                samples[key][i] = inst

        # prepare the embedding to condition and the embedding to regress
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)

        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        # ensemble the final targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos
        # if 'image_id' in samples.keys():
        #     print('image: ', samples['image_id'])
        # print('text input: ', samples['instruction_input'])
        # print('text output: ', samples['answer'])
        # if 'all_points' in samples.keys():
        #     print('keypoints: ', samples['all_points'])
        #     print('vis: ', samples['all_vis'])
        # print('targets: ', targets.shape, targets)
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                labels=targets,
                reduction=reduction
            )
        llm_loss = outputs.loss

        logit_ids = outputs.logits.argmax(dim=-1)
        # print('logits: ', logit_ids[:, -10:])
        # preds = []
        # for logit in logit_ids:
        #     pred = self.llama_tokenizer.decode(logit, skip_special_tokens=True)
        #     preds.append(pred)
        # print('preds: ', preds)

        # predict keypoints using the hidden states of all [POINT] tokens
        last_hidden_states = outputs.hidden_states[-1]
        point_token_mask = (targets[:, 1:] == self.point_token_id)
        point_token_mask = torch.cat(
            [
                point_token_mask,
                torch.zeros((point_token_mask.shape[0], 1)).bool().to(point_token_mask.device),
            ],
            dim=1,
        )
        # print('point token mask: ', point_token_mask)
        point_embeddings = last_hidden_states[point_token_mask]
        point_token_counts = point_token_mask.int().sum(dim=-1)

        point_token_offset = point_token_counts.cumsum(dim=-1)
        point_token_offset = torch.cat([torch.zeros(1).long().to(point_token_offset.device), point_token_offset], dim=0)

        if len(point_embeddings) > 0:
            point_preds = self.point_proj[0](point_embeddings)
            point_preds_ = []
            for i in range(len(point_token_offset) - 1):
                start_i, end_i = point_token_offset[i], point_token_offset[i + 1]
                point_preds_.append(point_preds[start_i: end_i])
            point_preds = point_preds_
            point_loss = self.point_pred_loss(samples['all_points'], samples['all_vis'], point_preds, samples['bbox'])
            loss = llm_loss + self.point_loss_alpha * point_loss
        else:
            point_loss = torch.tensor(0.0)
            loss = llm_loss

        return {"loss": loss, "llm_loss": llm_loss, "point_loss": point_loss}

    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        
        if self.wo_task_identifier: # remove the task identifier
            for i in range(len(texts)):
                inst = texts[i]
                inst = re.sub(r'\[[a-z\s]*\]', '', inst)
                texts[i] = inst

        img_embeds, atts_img = self.encode_img(images.to(self.device))
        image_lists = [[image_emb[None]] for image_emb in img_embeds]

        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
                output_hidden_states=True,
                # stopping_criteria=stopping_criteria,
            )

        # predict point coordinates using hidden states at [POINT] token
        output_hidden_states = []
        for h in outputs.hidden_states:
            output_hidden_states.append(h[-1][:, -1, :].unsqueeze(1))
        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_ids = outputs.sequences
        point_token_mask = (output_ids[:, 1:] == self.point_token_id)
        # point_token_mask = torch.cat(
        #         [
        #             torch.zeros((point_token_mask.shape[0], max_len)).bool().to(output_ids.device),
        #             point_token_mask,
        #         ],
        #         dim=1,
        #     )
        point_embeddings = output_hidden_states[point_token_mask]
        point_token_counts = point_token_mask.int().sum(dim=-1)
        point_token_offset = point_token_counts.cumsum(dim=-1)
        point_token_offset = torch.cat([torch.zeros(1).long().to(point_token_offset.device), point_token_offset], dim=0)
        if len(point_embeddings) == 0:
            point_preds = [None] * batch_size
        else:
            with self.maybe_autocast(): 
                point_preds = self.point_proj[0](point_embeddings)
            point_preds_ = []
            for i in range(len(point_token_offset) - 1):
                start_i, end_i = point_token_offset[i], point_token_offset[i + 1]
                if end_i > start_i:
                    point_preds_.append(point_preds[start_i: end_i])
                else:
                    point_preds_.append(None)
            point_preds = point_preds_

        # detokenize llm output
        answers = []
        for output_token in outputs.sequences:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)
        
        # print('input: ', texts)
        # print('output: ', answers)

        return answers, point_preds

    def point_pred_loss(self, target_points, target_vis, pred_points, bboxes):
        n_samples = len(target_points)
        losses = []
        for i in range(n_samples):
            pred = pred_points[i]
            target = target_points[i].to(pred.dtype).unsqueeze(0)
            vis = target_vis[i].to(pred.dtype).repeat_interleave(2)
            # bbox = bboxes[i]
            # scale = bbox[2:4] - bbox[0:2]
            # scale = (scale + 1e-5) ** -1
            # scale = scale.repeat(len(vis) // 2)
            loss_fct = nn.SmoothL1Loss(reduction='none', beta=0.002)
            # loss_fct = nn.MSELoss(reduction='none')
            l1_loss = loss_fct(pred, target) * vis
            if vis.sum() > 0:
                l1_loss = l1_loss.sum() / vis.sum()
            else:
                l1_loss = l1_loss.sum()
            losses.append(l1_loss)
        losses = torch.stack(losses).mean()

        return losses.mean()

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)

        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        num_keypoints = cfg.get("num_keypoints", 17)
        point_loss_alpha = cfg.get("point_loss_alpha", 10.0)

        half_freeze = cfg.get("half_freeze", False)
        tune_posembed = cfg.get("tune_posembed", False)
        tune_layernorm = cfg.get("tune_layernorm", False)
        use_vit_adapter= cfg.get("use_vit_adapter", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
            num_keypoints=num_keypoints,
            point_loss_alpha=point_loss_alpha,
            half_freeze=half_freeze,
            tune_posembed=tune_posembed,
            tune_layernorm=tune_layernorm,
            use_vit_adapter=use_vit_adapter
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Minigpt-4-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            # print(msg)

        return model
