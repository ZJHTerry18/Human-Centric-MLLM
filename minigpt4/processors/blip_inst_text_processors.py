import re
import torch
import numpy as np
import random

import omegaconf
from omegaconf import OmegaConf
from minigpt4.common.registry import registry
from minigpt4.processors.blip_processors import BlipImageBaseProcessor, BlipCaptionProcessor
from minigpt4.common.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, BBOX_TEMPLATE, PERSON_PLACEHOLDER, CLASSES_PLACEHOLDER
from minigpt4.conversation.conversation import CONV_VISION_minigptv2


@registry.register_processor('blip2_densecap_train')
class Blip2TextDensecapTrainProcessor(BlipCaptionProcessor):
    def __init__(self, quantize_bins=100):
        self.quantize_bins = quantize_bins
    
    def conv_process(self, raw_conv, target, task_identifier=''):
        ## construct input sentence
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()

        return raw_conv

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)

        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        
        return cls(quantize_bins=quantize_bins)

@registry.register_processor('blip2_densecap_eval')
class Blip2TextDensecapEvalProcessor(Blip2TextDensecapTrainProcessor):
    def __init__(self, quantize_bins=100):
        super(Blip2TextDensecapEvalProcessor, self).__init__(quantize_bins)
    
    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.append_message(conv_temp.roles[0], raw_conv[0]['value'])
        conv_temp.append_message(conv_temp.roles[1], None)
        raw_conv[0]['value'] = conv_temp.get_prompt()

        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        
        return cls(quantize_bins=quantize_bins)



@registry.register_processor('blip2_text_instcap_train')
class Blip2TextInstcapTrainProcessor(BlipCaptionProcessor):
    def __init__(self, quantize_bins=100, bbox_ref_prob=0.2):
        self.quantize_bins = quantize_bins
        self.bbox_ref_prob = bbox_ref_prob
    
    def conv_process(self, raw_conv, target, task_identifier=''):
        # choose reference string
        if random.random() < self.bbox_ref_prob:
            bbox = target['bbox']
            bbox[2:4] += bbox[0:2]
            bbox = bbox.clip(min=0.0, max=1.0)
            bbox = [int(x * self.quantize_bins) for x in bbox]
            person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
        else:
            refer_strs = target['refer']
            person_refer_str = random.choice(refer_strs) if isinstance(refer_strs, list) else refer_strs

        ## construct input sentence
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()

        return raw_conv

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)

        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        bbox_ref_prob = cfg.get("bbox_ref_prob", 0.2)
        
        return cls(quantize_bins=quantize_bins,
                   bbox_ref_prob=bbox_ref_prob)

@registry.register_processor('blip2_text_instcap_eval')
class Blip2TextInstcapEvalProcessor(Blip2TextInstcapTrainProcessor):
    def __init__(self, quantize_bins=100, bbox_ref_prob=0.2):
        super(Blip2TextInstcapEvalProcessor, self).__init__(quantize_bins, bbox_ref_prob)
    
    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.append_message(conv_temp.roles[0], raw_conv[0]['value'])
        conv_temp.append_message(conv_temp.roles[1], None)
        raw_conv[0]['value'] = conv_temp.get_prompt()

        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        bbox_ref_prob = cfg.get("bbox_ref_prob", 0.2)
        
        return cls(quantize_bins=quantize_bins,
                   bbox_ref_prob=bbox_ref_prob)

@registry.register_processor('blip2_text_instground_train')
class Blip2TextInstgroundingTrainProcessor(BlipCaptionProcessor):
    def __init__(self, quantize_bins=100, bbox_ref_prob=0.2):
        self.quantize_bins = quantize_bins
        self.bbox_ref_prob = bbox_ref_prob
    
    def conv_process(self, raw_conv, target, task_identifier=''):
        if 'bbox' in target.keys():
            # choose reference string
            if random.random() < self.bbox_ref_prob:
                bbox = target['bbox']
                bbox[2:4] += bbox[0:2]
                bbox = bbox.clip(min=0.0, max=1.0)
                bbox = [int(x * self.quantize_bins) for x in bbox]
                person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
            else:
                refer_strs = target['refer']
                person_refer_str = random.choice(refer_strs) if isinstance(refer_strs, list) else refer_strs

        ## construct input sentence
        if 'bbox' in target.keys():
            raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()

        ## construct output sentence
        answer = raw_conv[1]['value']
        object_locs = target['locs']
        objects = re.findall(r'<p>(.*?)</p>', answer)
        added_objects = []
        for i, obj in enumerate(objects):
            if len(object_locs[i]) == 0 and obj not in added_objects: # no location boxes
                answer = answer.replace(f'<p>{obj}</p>', obj)
                continue
            
            box_list = []
            for box in object_locs[i]:
                box[2:4] += box[0:2]
                box = box.clip(min=0.0, max=1.0).flatten().tolist() 
                box = [int(x * self.quantize_bins) for x in box]
                box_str = BBOX_TEMPLATE.format(*box)
                box_list.append(box_str)
            box_str = "<delim>".join(box_list)
            if obj not in added_objects:
                answer = answer.replace(f'<p>{obj}</p>', f'<p>{obj}</p>{box_str}')
                added_objects.append(obj)
        raw_conv[1]['value'] = answer.lower()

        return raw_conv

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)

        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        bbox_ref_prob = cfg.get("bbox_ref_prob", 0.2)
        
        return cls(quantize_bins=quantize_bins,
                   bbox_ref_prob=bbox_ref_prob)

@registry.register_processor('blip2_text_instground_eval')
class Blip2TextInstgroundingEvalProcessor(Blip2TextInstgroundingTrainProcessor):
    def __init__(self, quantize_bins=100, bbox_ref_prob=0.2):
        super(Blip2TextInstgroundingEvalProcessor, self).__init__(quantize_bins, bbox_ref_prob)

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.append_message(conv_temp.roles[0], raw_conv[0]['value'])
        conv_temp.append_message(conv_temp.roles[1], None)
        raw_conv[0]['value'] = conv_temp.get_prompt()

        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        bbox_ref_prob = cfg.get("bbox_ref_prob", 0.2)
        
        return cls(quantize_bins=quantize_bins,
                   bbox_ref_prob=bbox_ref_prob)

@registry.register_processor('blip2_text_parsingrec_train')
class Blip2TextParsingRECTrainProcessor(BlipCaptionProcessor):
    def __init__(self, quantize_bins=100, bbox_ref_prob=0.2):
        self.quantize_bins = quantize_bins
        self.bbox_ref_prob = bbox_ref_prob
    
    def conv_process(self, raw_conv, target, task_identifier=''):
        # choose reference string
        if random.random() < self.bbox_ref_prob:
            bbox = target['bbox']
            bbox[2:4] += bbox[0:2]
            bbox = bbox.clip(min=0.0, max=1.0)
            bbox = [int(x * self.quantize_bins) for x in bbox]
            person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
        else:
            refer_strs = target['refer']
            person_refer_str = random.choice(refer_strs) if isinstance(refer_strs, list) else refer_strs
        
        # choose body part
        valid_index = [i for i in range(len(target['segmentation'])) if len(target['segmentation'][i]) > 0]
        segs_list = [target['segmentation'][i] for i in valid_index]
        class_list = [target['class'][i] for i in valid_index]
        select_index = random.choice(list(range(len(segs_list))))

        poly_list = []
        for poly in segs_list[select_index]:
            poly = poly.clip(min=0.0, max=1.0).flatten().tolist() 
            poly = [int(x * self.quantize_bins) for x in poly]
            poly_template = "{{" + "<{}>" * len(poly) + "}}"
            poly_str = poly_template.format(*poly)
            poly_list.append(poly_str)
        part_str = "<delim>".join(poly_list)
        cls_str = class_list[select_index]

        ## construct input sentence
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str).replace(CLASSES_PLACEHOLDER, cls_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()
        
        ## construct output sentence
        raw_conv[1]['value'] = part_str.lower()

        return raw_conv

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        bbox_ref_prob = cfg.get("bbox_ref_prob", 0.2)
        
        return cls(quantize_bins=quantize_bins,
                   bbox_ref_prob=bbox_ref_prob)

@registry.register_processor('blip2_text_parsingrec_eval')
class Blip2TextParsingRECEvalProcessor(Blip2TextParsingRECTrainProcessor):
    def __init__(self, quantize_bins=100, bbox_ref_prob=0.2):
        super(Blip2TextParsingRECEvalProcessor, self).__init__(quantize_bins, bbox_ref_prob)

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.append_message(conv_temp.roles[0], raw_conv[0]['value'])
        conv_temp.append_message(conv_temp.roles[1], None)
        raw_conv[0]['value'] = conv_temp.get_prompt()

        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        bbox_ref_prob = cfg.get("bbox_ref_prob", 0.2)
        
        return cls(quantize_bins=quantize_bins,
                   bbox_ref_prob=bbox_ref_prob)

@registry.register_processor('blip2_text_instpartrec_train')
class Blip2TextInstPartRECTrainProcessor(BlipCaptionProcessor):
    def __init__(self, quantize_bins=100, bbox_ref_prob=0.2):
        self.quantize_bins = quantize_bins
        self.bbox_ref_prob = bbox_ref_prob
    
    def conv_process(self, raw_conv, target, task_identifier=''):
        # choose reference string
        if random.random() < self.bbox_ref_prob:
            bbox = target['bbox']
            bbox[2:4] += bbox[0:2]
            bbox = bbox.clip(min=0.0, max=1.0)
            bbox = [int(x * self.quantize_bins) for x in bbox]
            person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
        else:
            refer_strs = target['refer']
            person_refer_str = random.choice(refer_strs) if isinstance(refer_strs, list) else refer_strs
        
        # choose body part
        classes = re.findall(r'<p>(.*?)</p>', raw_conv[1]['value'])
        valid_index = [i for i in range(len(target['locs'])) if len(target['locs'][i]) > 0]
        objects_list = [target['locs'][i] for i in valid_index]
        class_list = [classes[i] for i in valid_index]
        select_index = random.choice(list(range(len(objects_list))))

        box_list = []
        for box in objects_list[select_index]:
            box[2:4] += box[0:2]
            box = box.clip(min=0.0, max=1.0).flatten().tolist() 
            box = [int(x * self.quantize_bins) for x in box]
            box_str = BBOX_TEMPLATE.format(*box)
            box_list.append(box_str)
        part_str = "<delim>".join(box_list)
        cls_str = class_list[select_index]

        ## construct input sentence
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str).replace(CLASSES_PLACEHOLDER, cls_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()
        
        ## construct output sentence
        raw_conv[1]['value'] = part_str.lower()

        return raw_conv

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        bbox_ref_prob = cfg.get("bbox_ref_prob", 0.2)
        
        return cls(quantize_bins=quantize_bins,
                   bbox_ref_prob=bbox_ref_prob)

@registry.register_processor('blip2_text_instrec_train')
class Blip2TextInstRECTrainProcessor(BlipCaptionProcessor):
    def __init__(self, quantize_bins=100):
        self.quantize_bins = quantize_bins
    
    def conv_process(self, raw_conv, target, task_identifier=''):
        ## construct input sentence
        if random.random() < 0.5:
            refer_strs = target['refer']
            person_refer_str = random.choice(refer_strs) if isinstance(refer_strs, list) else refer_strs
        else:
            caption = raw_conv[1]['value'].replace('<p>', '').replace('</p>', '')
            templates = ['the person according to the following description {}',
                         'the person described as follows {}',
                         'the described person {}']
            person_refer_str = random.choice(templates).format(caption)
        # print(person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()

        # construct output sentence
        bbox = target['bbox']
        bbox[2:4] += bbox[0:2]
        bbox = bbox.clip(min=0.0, max=1.0)
        bbox = [int(x * self.quantize_bins) for x in bbox]
        bbox_str = BBOX_TEMPLATE.format(*bbox)
        raw_conv[1]['value'] = bbox_str.lower()

        return raw_conv

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        
        return cls(quantize_bins=quantize_bins,)
    
@registry.register_processor('blip2_text_instrec_eval')
class Blip2TextInstRECEvalProcessor(BlipCaptionProcessor):
    def __init__(self, quantize_bins=100):
        self.quantize_bins = quantize_bins
    
    def conv_process(self, raw_conv, target, task_identifier=''):
        # choose reference string
        caption = raw_conv[1]['value'].replace('<p>', '').replace('</p>', '')
        template = f'give me the location of the person according to the following description {caption}'

        # task_identifier = '[refer]'
        raw_conv[0]['value'] = template
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()

        # construct output sentence
        bbox = target['bbox']
        bbox[2:4] += bbox[0:2]
        bbox = bbox.clip(min=0.0, max=1.0)
        bbox = [int(x * self.quantize_bins) for x in bbox]
        bbox_str = BBOX_TEMPLATE.format(*bbox)
        raw_conv[1]['value'] = bbox_str.lower()

        return raw_conv

    def __call__(self, raw_conv, target, task_identifier=''):
        raw_conv = self.conv_process(raw_conv, target, task_identifier)
        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.append_message(conv_temp.roles[0], raw_conv[0]['value'])
        conv_temp.append_message(conv_temp.roles[1], None)
        raw_conv[0]['value'] = conv_temp.get_prompt()
        for sentence in raw_conv:
            sentence['value'] = re.sub(
                r"([.!\"()*#:;~])",
                " ",
                sentence['value'],
            )
            sentence['value'] = re.sub(
                r"\s{2,}",
                " ",
                sentence['value'],
            )

        return raw_conv
    
    @classmethod
    def from_config(cls, cfg=None):
        quantize_bins = cfg.get("quantize_bins", 100)
        
        return cls(quantize_bins=quantize_bins,)