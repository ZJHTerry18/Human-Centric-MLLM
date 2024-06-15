import os
import re
import json
import argparse
import pandas as pd
import math
from collections import defaultdict
import random
import base64
from io import BytesIO
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.datasets.datasets.multitask_qa_datasets import EvalDataset, MultiChoiceEvalDataset, BBoxEvalDataset, MultiBBoxEvalDataset
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
# parser.add_argument("--cfg-path", type=str, default="eval_configs/minigptv2_free_evaluation.yaml")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--all-rounds", action="store_true")
parser.add_argument("--single-pred-prompt", action="store_true")
parser.add_argument("--lang", type=str, default="en")
parser.add_argument("--question-file", type=str, default="inference_cases/mmbench_dev_20230712.tsv")
# parser.add_argument("--max_new_tokens", type=int, default=500)
args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(args.seed)

all_options = ['A', 'B', 'C', 'D']

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

model, vis_processor = init_model(args)
model_arch = 'minigpt_v2'
model.eval()

CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""


questions = pd.read_table(os.path.expanduser(args.question_file))
questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

minigpt4_predict_list = []
i = 0
for index, row in tqdm(questions.iterrows(), total=len(questions)):
    options = get_options(row, all_options)
    cur_option_char = all_options[:len(options)]

    if args.all_rounds:
        num_rounds = len(options)
    else:
        num_rounds = 1

    for round_idx in range(num_rounds):
        idx = row['index']
        question = row['question']
        hint = row['hint']
        gt_answer = row['answer']
        image = load_image_from_base64(row['image'])
        if not is_none(hint):
            question = hint + '\n' + question
        for option_char, option in zip(all_options[:len(options)], options):
            question = question + '\n' + option_char + '. ' + option
        qs = cur_prompt = question

        if args.single_pred_prompt:
            if args.lang == 'cn':
                qs = qs + '\n' + "请直接回答选项字母。"
            else:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        qs = [qs.strip()]

        # print(image.size, qs, gt_answer, idx)
        texts = prepare_texts(qs, conv_temp)
        image = vis_processor(image).unsqueeze(0).cuda()

        with torch.inference_mode():
            answer = model.generate(image, texts, max_new_tokens=args.max_new_tokens, do_sample=False)
        
        answer = answer[0].replace("<unk>","").strip()
        minigpt4_predict_list.append({
                'id': idx,
                'input': question,
                'output': answer,
                'gt': gt_answer
        })
    
    i += 1
    # if i == 50:
    #     break
        

file_save_path = 'evaluation/mmbench.jsonl'
print('file_save_path:', file_save_path)
directory = os.path.dirname(file_save_path)
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

with open(file_save_path, 'w') as f:
    for dat in minigpt4_predict_list:
        f.write(json.dumps(dat) + '\n')