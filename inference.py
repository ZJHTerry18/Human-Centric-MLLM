import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.datasets.datasets.multitask_qa_datasets import EvalDataset
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def pred_minigptv2_default(eval_dataloader, minigpt4_predict_list):
    for images, questions, img_ids, img_paths in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        print('img_path:', img_paths)
        print('input: ', texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, img_path, question in zip(answers, img_ids, img_paths, questions):
            answer = answer.replace("<unk>","").strip()
            img_id = img_id.item()
            minigpt4_predict_list.append({
                    'id': img_id,
                    'image': img_path,
                    'input': question,
                    'answer': answer
            })

def pred_minigptv2_posereg(eval_dataloader, minigpt4_predict_list):
    for images, questions, img_ids, img_paths in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers, keypoints = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, img_path, question, keypoint in zip(answers, img_ids, img_paths, questions, keypoints):
            keypoint = keypoint.cpu().numpy().tolist() if keypoint is not None else []
            answer = answer.replace("<unk>","").strip()
            img_id = img_id.item()
            minigpt4_predict_list.append({
                    'id': img_id,
                    'image': img_path,
                    'input': question,
                    'answer': answer,
                    'keypoint': keypoint
            })

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

cfg = Config(args)

model, vis_processor = init_model(args)
model_arch = cfg.model_cfg.arch
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

# 
model.eval()
save_path = cfg.run_cfg.save_path
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = os.path.join(save_path, timestamp)

for dataset in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    input_samples = []
    with open(eval_file_path, 'r') as f:
        for line in f:
            input_samples.append(json.loads(line))
    data = EvalDataset(input_samples, vis_processor)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict_list = []

    resamples = []
    # for images, questions, img_ids, img_paths in tqdm(eval_dataloader):
    #         texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
    #         answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
    #         for answer, img_id, img_path, question in zip(answers, img_ids, img_paths, questions):
    #             answer = answer.replace("<unk>","").replace(" ","").strip()
    #             img_id = img_id.item()
    #             minigpt4_predict_list.append({
    #                  'id': img_id,
    #                  'image': img_path,
    #                  'input': question,
    #                  'answer': answer
    #             })
    if model_arch == "minigpt_v2_pose":
        pred_minigptv2_posereg(eval_dataloader, minigpt4_predict_list)
    elif model_arch == "minigpt_v2":
        pred_minigptv2_default(eval_dataloader, minigpt4_predict_list)
    else:
        raise TypeError(f"{model_arch} model type is not supported")

    
    file_save_path = os.path.join(save_path,f"{dataset}.jsonl")
    print('file_save_path:', file_save_path)
    directory = os.path.dirname(file_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(file_save_path,'w') as f:
        for p in minigpt4_predict_list:
            f.write(json.dumps(p) + '\n')



