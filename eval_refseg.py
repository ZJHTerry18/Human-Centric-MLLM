import os
import os.path as osp
import re
import cv2
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

from minigpt4.datasets.datasets.coco_seg_dataset import RefCOCOSegEvalData
from minigpt4.metrics.parsing_metric import HumParEvaluator

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
save_path = os.path.join(save_path, timestamp)


for dataset in args.dataset:
    for split in eval_dict[dataset]:

        eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

        with open(os.path.join(eval_file_path,f"{dataset}/{dataset}_seg_{split}.json"), 'r') as f:
            refcoco = json.load(f)

        # refcoco = refcoco[:10] # sample mini-set for quick evaluation
        data = RefCOCOSegEvalData(refcoco, vis_processor, img_path)
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict_dict = defaultdict(dict) #defaultdict(list)
        resamples = []

        for images, questions, img_ids in tqdm(eval_dataloader):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, img_id, question in zip(answers, img_ids, questions):
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{(<\d{1,3}>)+\}'
                if re.match(pattern, answer):
                    exp = question.replace("[refer segmentation] give me the boundary points of ","")
                    minigpt4_predict_dict[img_id][exp] = answer
                    #minigpt4_predict[img_id].append(answer)
                else:
                    # print(question.replace('[refer segmentation] give me the boundary points of','').strip())
                    resamples.append({'img_id': img_id, 'sents': [question.replace('[refer segmentation] give me the boundary points of','').strip()]})
        if args.resample:
            for i in range(20):
                data = RefCOCOSegEvalData(resamples, vis_processor, img_path)
                resamples = []
                eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
                for images, questions, img_ids in tqdm(eval_dataloader):
                    texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                    answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                    for answer, img_id, question in zip(answers, img_ids, questions):
                        answer = answer.replace("<unk>","").replace(" ","").strip()
                        pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                        if re.match(pattern, answer) or i == 4:
                            exp = question.replace("[refer segmentation] give me the boundary points of ","")
                            minigpt4_predict_dict[img_id][exp] = answer
                        else:
                            # print(question.replace('[refer segmentation] give me the boundary points of','').strip())
                            resamples.append({'img_id': img_id, 'sents': [question.replace('[refer segmentation] give me the boundary points of','').strip()]})
                            
                if len(resamples) == 0:
                    break
        
        file_save_path = os.path.join(save_path,f"{dataset}_{split}.json")
        directory = os.path.dirname(file_save_path)
        # 如果目录不存在，则创建它
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        vis_root = os.path.join(save_path, split, 'vis')
        os.makedirs(vis_root, exist_ok=True)
        std_dict = dict(minigpt4_predict_dict)
        with open(file_save_path,'w') as f:
            json.dump(std_dict, f)

        gt_masks = []
        pred_masks = []
        res=args.res
        evaluator = HumParEvaluator(class_names=[0, 1])
        evaluator.reset()
        for item in refcoco:
            #refcoco_dict[item['img_id']] = item
            img_id = item['img_id']
            sent_id = item['sent_id']
            show_image = cv2.imread(osp.join(img_path, img_id))
            gt_segmentation = item['segmentation']
            #item = refcoco_dict[img_id]
            
            output = minigpt4_predict_dict[img_id].get(item['sents'], None)
            if output is None:
                # print(item['sents'])
                output = '{<0><0><0><0>}'
            height = item['height']
            width = item['width']
            seg_strs = output.split('<delim>')
            seg_pattern = r"<(\d{1,3})>"
            pred_segmentation = []
            for seg in seg_strs:
                poly_list = re.findall(seg_pattern, seg)
                poly_list = poly_list[:-1] if len(poly_list) % 2 == 1 else poly_list[:] # TODO: how to deal with odd number of outputs
                polygon = np.array(poly_list).reshape(-1, 2).astype(float)
                polygon = polygon / res * np.array([width, height])
                polygon_list = np.round(polygon, 2).flatten().tolist()
                pred_segmentation.append(polygon_list)
            
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in gt_segmentation:
                polygon = np.array(seg).reshape(-1, 2)
                cv2.fillConvexPoly(gt_mask, np.int32([polygon]), 1)
                cv2.polylines(show_image, np.int32([polygon]), True, (0, 0, 255), 2)
            gt_masks.append(gt_mask)
            
            pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
            for seg in pred_segmentation:
                polygon = np.array(seg).reshape(-1, 2)
                cv2.fillConvexPoly(pred_mask, np.int32([polygon]), 1)
                cv2.polylines(show_image, np.int32([polygon]), True, (0, 255, 0), 1)
            pred_masks.append(pred_mask)

            save_file = osp.join(vis_root, img_id[:-4] + '_' + str(sent_id) + '.jpg')
            cv2.imwrite(save_file, show_image)
        
        evaluator.process(gt_masks, pred_masks)
        print(f'Evalute results on {dataset}-{split}:')
        results = evaluator.evaluate()
        

# CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run --master-port 29505 --nproc_per_node 1 eval_scripts/eval_ref.py \
#  --cfg-path /home/human/codes/human/MiniGPT-4/eval_configs/minigptv2_benchmark_evaluation.yaml --dataset refcoco,refcoco+,refcocog #--resample