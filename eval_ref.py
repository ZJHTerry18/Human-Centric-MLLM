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
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

from minigpt4.datasets.datasets.coco_caption import RefCOCOEvalData

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

cfg = Config(args)

eval_dict = {'refcoco': ['val','testA','testB'], 
            'refcoco+': ['val','testA','testB'],
            'refcocog': ['val','test']}


model, vis_processor = init_model(args)
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

# 
model.eval()
save_path = cfg.run_cfg.save_path
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = os.path.join(save_path, 'timestamp')


for dataset in args.dataset:
    for split in eval_dict[dataset]:

        eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

        with open(os.path.join(eval_file_path,f"{dataset}/{dataset}_{split}.json"), 'r') as f:
            refcoco = json.load(f)

        data = RefCOCOEvalData(refcoco, vis_processor, img_path)
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict_dict = defaultdict(dict) #defaultdict(list)
        resamples = []

        for images, questions, img_ids in tqdm(eval_dataloader):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, img_id, question in zip(answers, img_ids, questions):
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    exp = question.replace("[refer] give me the location of ","")
                    minigpt4_predict_dict[img_id][exp] = answer
                    #minigpt4_predict[img_id].append(answer)
                else:
                    print(question.replace('[refer] give me the location of','').strip())
                    resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] give me the location of','').strip()]})
        if args.resample:
            for i in range(20):
                data = RefCOCOEvalData(resamples, vis_processor, img_path)
                resamples = []
                eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
                for images, questions, img_ids in tqdm(eval_dataloader):
                    texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                    answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                    for answer, img_id, question in zip(answers, img_ids, questions):
                        answer = answer.replace("<unk>","").replace(" ","").strip()
                        pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                        if re.match(pattern, answer) or i == 4:
                            exp = question.replace("[refer] give me the location of ","")
                            minigpt4_predict_dict[img_id][exp] = answer
                        else:
                            print(question.replace('[refer] give me the location of','').strip())
                            resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] give me the location of','').strip()]})
                            
                if len(resamples) == 0:
                    break
        
        file_save_path = os.path.join(save_path,f"{dataset}_{split}.json")
        print('file_save_path:', file_save_path)
        directory = os.path.dirname(file_save_path)
        print(directory)
        # 如果目录不存在，则创建它
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        std_dict = dict(minigpt4_predict_dict)
        with open(file_save_path,'w') as f:
            json.dump(std_dict, f)

        count=0
        total=len(refcoco)
        res=args.res
        #refcoco_dict = defaultdict()
        for item in refcoco:
            #refcoco_dict[item['img_id']] = item
            img_id = item['img_id']
            bbox = item['bbox']
            #item = refcoco_dict[img_id]
            
            output = minigpt4_predict_dict[img_id].get(item['sents'], None)
            if output is None:
                print(item['sents'])
                output = '{<0><0><0><0>}'
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [0,0,0,0]
                gt_bbox[0] = bbox[0]
                gt_bbox[1] = bbox[1]
                gt_bbox[2] = bbox[2] #bbox[0] + bbox[2]
                gt_bbox[3] = bbox[3] #bbox[1] + bbox[3]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count+=1
            except:
                continue
        
        print(f'{dataset} {split}:', count / total * 100, flush=True)