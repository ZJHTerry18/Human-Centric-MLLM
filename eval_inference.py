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
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.datasets.datasets.multitask_qa_datasets import EvalDataset, VizWizEvalDataset, MultiChoiceEvalDataset, BBoxEvalDataset, MultiBBoxEvalDataset
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

vizwiz_image_root = '/home/human/data/VizWiz/val'

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def pred_minigptv2_multichoice_default(eval_dataloader, minigpt4_predict_list):
    for images, questions, choices, gts, img_ids, img_paths in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for i, (answer, gt, img_id, img_path, question) in enumerate(zip(answers, gts, img_ids, img_paths, questions)):
            answer = answer.replace("<unk>","").strip()
            img_id = img_id.item() if not isinstance(img_id, str) else img_id
            choices_dict = {key: choices[key][i] for key in choices}
            minigpt4_predict_list.append({
                    'id': img_id,
                    'image': img_path,
                    'input': question,
                    'output': answer,
                    'choices': choices_dict,
                    'gt': gt
            })

def pred_minigptv2_vizwiz_default(eval_dataloader, minigpt4_predict_list):
    for images, questions, answers_s, answer_types, answerables, img_paths in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for i, (answer, gtanswers, answer_type, answerable, img_path, question) in enumerate(zip(answers, answers_s, answer_types, answerables, img_paths, questions)):
            answer = answer.replace("<unk>","").strip()
            minigpt4_predict_list.append({
                    'image': img_path,
                    'question': question,
                    'answers': gtanswers,
                    'answer': answer,
                    'answer_type': answer_type,
                    'answerable': int(answerable),
            })

def pred_minigptv2_default(eval_dataloader, minigpt4_predict_list):
    for images, questions, gts, img_ids, img_paths in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for i, (answer, gt, img_id, img_path, question) in enumerate(zip(answers, gts, img_ids, img_paths, questions)):
            answer = answer.replace("<unk>","").strip()
            img_id = img_id.item() if not isinstance(img_id, str) else img_id
            minigpt4_predict_list.append({
                    'id': img_id,
                    'image': img_path,
                    'input': question,
                    'output': answer,
                    'gt': [float(num) for num in gt.split(' ')]
            })

# def pred_minigptv2_default(eval_dataloader, minigpt4_predict_list):
#     for images, questions, gts, img_ids, img_paths in tqdm(eval_dataloader):
#         texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
#         answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
#         for i, (answer, gt, img_id, img_path, question) in enumerate(zip(answers, gts, img_ids, img_paths, questions)):
#             answer = answer.replace("<unk>","").strip()
#             img_id = img_id.item() if not isinstance(img_id, str) else img_id
#             minigpt4_predict_list.append({
#                     'id': img_id,
#                     'image': img_path,
#                     'input': question,
#                     'output': answer,
#                     'gt': gt
#             })

def pred_minigptv2_multibbox(eval_dataloader, minigpt4_predict_list):
    for images, questions, gts, img_ids, img_paths in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for i, (answer, gt, img_id, img_path, question) in enumerate(zip(answers, gts, img_ids, img_paths, questions)):
            # print('gt:', gt)
            answer = answer.replace("<unk>","").strip()
            img_id = img_id.item() if not isinstance(img_id, str) else img_id
            minigpt4_predict_list.append({
                    'id': img_id,
                    'image': img_path,
                    'input': question,
                    'output': answer,
                    'gt': [[float(num) for num in bbox.split(',')] for bbox in gt.split('<delim>')]
            })

def transfer(answer):
    bbox = [num/100 for num in answer]
    bbox = [bbox[0], bbox[1], round(bbox[2] - bbox[0], 3), round(bbox[3] - bbox[1], 3)]
    return bbox

def compute_IoU(output_bbox,gt):
    x1, y1, w1, h1 = output_bbox
    x2, y2, w2, h2 = gt
    
    # 计算交集部分的坐标
    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + w1, x2 + w2)
    ymax = min(y1 + h1, y2 + h2)
    
    # 计算交集的面积
    intersection_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    
    # 计算并集的面积
    box1_area = (w1) * (h1)
    box2_area = (w2) * (h2)
    union_area = box1_area + box2_area - intersection_area
    
    # 计算 IoU
    iou = intersection_area / union_area
    
    return iou

def min_iou_distance_matching(boxes1, boxes2):
    matched_pairs = []

    for i, box1 in enumerate(boxes1):
        best_match_index = -1
        best_iou = 0.0
        
        for j, box2 in enumerate(boxes2):
            iou = compute_IoU(box1, box2)
            if iou > best_iou:
                best_iou = iou
                best_match_index = j

        matched_pairs.append((i, best_match_index, best_iou))
    
    # 根据匹配结果对位排列两个列表
    sorted_boxes1 = [boxes1[pair[0]] for pair in matched_pairs]
    sorted_boxes2 = [boxes2[pair[1]] for pair in matched_pairs]

    return sorted_boxes1, sorted_boxes2

def eval_bbox(minigpt4_predict_list):
    correct = 0
    pattern = r'{<(\d+)><(\d+)><(\d+)><(\d+)>}'
    for qap in minigpt4_predict_list:
        output = qap['output']
        gt = qap['gt']
        output_match = re.search(pattern, output)
        if output_match:
            output_bbox = [int(output_match.group(i)) for i in range(1, 5)]
        else:
            continue
        output_bbox = transfer(output_bbox)
        iou = compute_IoU(output_bbox,gt)
        if iou > 0.5:
            correct+=1
    acc = correct/len(minigpt4_predict_list)
    print('acc:',acc)
    return acc

# def eval_multi_bbox(minigpt4_predict_list):
#     correct = 0
#     pattern = r'{<(\d+)><(\d+)><(\d+)><(\d+)>}'
#     for qap in minigpt4_predict_list:
#         output = qap['output']
#         gt_boxes = qap['gt']
#         output_bboxes = re.findall(pattern, output)
#         output_transfer_bboxes = []
#         for bbox in output_bboxes:
#             bbox = [int(x) for x in bbox]
#             bbox = transfer(bbox)
#             output_transfer_bboxes.append(bbox)
#         if len(output_transfer_bboxes) != len(gt_boxes):
#             continue
#         new_output_bboxes, new_gt_bboxes = min_iou_distance_matching(output_transfer_bboxes, gt_boxes)
#         c = 1
#         for o, gt in zip(new_output_bboxes, new_gt_bboxes):
#             iou = compute_IoU(o, gt)
#             if iou < 0.5:
#                 c = 0
#         correct += c
#     acc = correct/len(minigpt4_predict_list)
#     print('acc:',acc)
#     return acc

def eval_multi_bbox(minigpt4_predict_list):
    correct = 0
    pattern = r'{<(\d+)><(\d+)><(\d+)><(\d+)>}'
    for qap in minigpt4_predict_list:
        output = qap['output']
        gt_boxes = qap['gt']
        output_bboxes = re.findall(pattern, output)
        output_transfer_bboxes = []
        for bbox in output_bboxes:
            bbox = [int(x) for x in bbox]
            bbox = transfer(bbox)
            output_transfer_bboxes.append(bbox)
        if len(output_transfer_bboxes) == 0:
            continue
        c = 0
        for gt in gt_boxes:
            iou = compute_IoU(output_transfer_bboxes[0], gt)
            if iou > 0.5:
                c = 1
        correct += c
    acc = correct/len(minigpt4_predict_list)
    print('acc:',acc)
    return acc


def eval_multichoice_minigptv2_default(eval_dataloader, minigpt4_predict_list):
    for images, questions, choices, gts, img_ids, img_paths in tqdm(eval_dataloader):
        questions = list(questions)
        for i in range(len(questions)):
            questions[i] = '<Img><ImageHere></Img> {}'.format(questions[i])
        losses = [dict() for _ in range(len(questions))]
        for i in range(len(questions)):
            for key in ["A", "B", "C", "D"]:
                samples = {
                    'image': images[i:i+1],
                    'instruction_input': questions[i:i+1],
                    'answer': choices[key][i:i+1]
                }
                samples = prepare_sample(samples)
                loss = model.forward(samples)['loss']
                losses[i][key] = loss.item()

        for i, (loss, gt, img_id, img_path, question) in enumerate(zip(losses, gts, img_ids, img_paths, questions)):
            min_loss = min(loss.values())
            answer_index = [key for key, value in loss.items() if value == min_loss][0]
            answer_content = choices[answer_index][i]
            loss_list = list(loss.values())
            img_id = img_id.item() if not isinstance(img_id, str) else img_id
            minigpt4_predict_list.append({
                    'id': img_id,
                    'image': img_path,
                    'input': question,
                    'output': f"({answer_index}) {answer_content}",
                    "loss_values": loss_list,
                    'gt': gt
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
dataset_type = cfg.run_cfg.dataset_type
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = os.path.join(save_path, timestamp)

for dataset in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
    batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

    input_samples = []
    if eval_file_path.endswith('.jsonl'):
        with open(eval_file_path, 'r') as f:
            for line in f:
                input_samples.append(json.loads(line))
    elif eval_file_path.endswith('.json'):
        with open(eval_file_path, 'r') as f:
            input_samples = json.load(f)
    # data = EvalDataset(input_samples, vis_processor)
    if dataset_type =='bbox':
        data = BBoxEvalDataset(input_samples, vis_processor)
    elif dataset_type == 'multi_bbox':
        data = MultiBBoxEvalDataset(input_samples, vis_processor)
    elif dataset_type == 'multichoice':
        data = MultiChoiceEvalDataset(input_samples, vis_processor)
    elif dataset_type == 'vizwiz':
        data = VizWizEvalDataset(vizwiz_image_root, input_samples, vis_processor)
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
        # pred_minigptv2_default(eval_dataloader, minigpt4_predict_list)
        if dataset_type =='bbox':
            pred_minigptv2_default(eval_dataloader, minigpt4_predict_list)
        elif dataset_type == 'multi_bbox':
            pred_minigptv2_multibbox(eval_dataloader, minigpt4_predict_list)
        elif dataset_type == 'multichoice':
            pred_minigptv2_multichoice_default(eval_dataloader, minigpt4_predict_list)
        elif dataset_type == 'vizwiz':
            pred_minigptv2_vizwiz_default(eval_dataloader, minigpt4_predict_list)
        # with torch.no_grad():
        #     eval_multichoice_minigptv2_default(eval_dataloader, minigpt4_predict_list)
    else:
        raise TypeError(f"{model_arch} model type is not supported")

    
    file_save_path = os.path.join(save_path,f"{dataset}.jsonl")
    print('file_save_path:', file_save_path)
    directory = os.path.dirname(file_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if dataset_type == 'vizwiz':
        with open(file_save_path, 'w') as f:
            json.dump(minigpt4_predict_list, f)
    else:
        with open(file_save_path,'w') as f:
            for p in minigpt4_predict_list:
                
                f.write(json.dumps(p) + '\n')

            if dataset_type =='bbox':
                acc = eval_bbox(minigpt4_predict_list)
                f.write(str(acc) + '\n')
            elif dataset_type == 'multi_bbox':
                acc = eval_multi_bbox(minigpt4_predict_list)
                f.write(str(acc) + '\n')




