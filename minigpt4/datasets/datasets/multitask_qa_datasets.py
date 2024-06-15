import warnings
import os
import os.path as osp
from functools import partial
from typing import Dict, Any, Callable, List, Optional, Tuple, Type

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainingArguments
from torch.utils.data import Dataset
import json, random
import numpy as np
import cv2
from minigpt4.common.utils import DETECTION_IDENTIFIER, POSE_IDENTIFIER, PARSING_IDENTIFIER, BBOX_TEMPLATE


class VisionQADataset(Dataset):
    def __init__(
            self, vis_processor, text_processor, vis_root, ann_paths, template_path: str = None, task: str = None,
    ):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.task = task
        
        self.question_template = []
        self.answer_template = []
        if template_path is not None:
            with open(template_path, 'r') as file:
                templates = json.load(file)
            self.question_template.extend(templates['question'])
            self.answer_template.extend(templates['answer'])
        
        self.annotation = []
        instance_idx = 0
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]
        for ann_path in ann_paths:
            with open(ann_path, 'r', encoding='utf8') as f:
                # for line in tqdm(f, desc=f'{self.__class__.__name__} loading ann {self.filename}'):
                for line in f:
                    ann = json.loads(line)
                    ann['instance_id'] = instance_idx
                    self.annotation.append(ann)
                    instance_idx += 1
    
    def __len__(self,):
        return len(self.annotation)

    def __getitem__(self, index, debug_mode=False, return_conv=False) -> Dict[str, Any]:
        # getitem
        item = self.get_raw_item(index)
        raw_image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']
        # TODO: complete validity check for tasks
        # self.validate_raw_item(item)

        # transform
        image, target = self.vis_processor(raw_image, target) # target: {'boxes'/'points'/'classes', 'size', }
        _, h, w = image.shape

        # preprocess
        raw_conv= self.process_conv_and_target(raw_conv, target) #handle index of box and insert box into text
        text_seg = self.segment_conv(raw_conv) #transform list to conversation

        # debug
        # import cv2
        # import numpy as np
        # import re
        # if self.task == 'detection':
        #     show_img = image.permute(1,2,0).contiguous().numpy()
        #     show_img = (show_img * 255).astype('uint8')
        #     show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
        #     H, W, _, = show_img.shape
        #     pattern = re.compile(r"{<(\d+)><(\d+)><(\d+)><(\d+)>}")
        #     bbox_list = re.findall(pattern, text_seg['answer'])
        #     bbox_list = [np.array(list(map(float, x))) for x in bbox_list]
        #     if len(bbox_list) > 0:
        #         bboxes = np.stack(bbox_list, axis=0) / 100
        #         for box in bboxes:
        #             x1 = int(box[0] * W)
        #             y1 = int(box[1] * H)
        #             x2 = int(box[2] * W)
        #             y2 = int(box[3] * H)

        #             show_img = cv2.rectangle(show_img, (x1, y1), (x2, y2), (0,255,0), 5)
        #     cv2.imwrite(os.path.join('debug', 'detection', os.path.basename(item['image_id'])), show_img)
        #     # print(item['image_id'])
        #     # print(text_seg['instruction_input'])
        #     # print(text_seg['answer'])
        # if self.task == 'pose_estimation':
        #     show_img = image.permute(1,2,0).contiguous().numpy()
        #     show_img = (show_img * 255).astype('uint8')
        #     show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
        #     H, W, _, = show_img.shape
        #     # skeleton = [[0,1], [0,2], [1,2], [0,3], [0,4], [3,5], [4,6], [3,4], [1,2], [2,7], [2,8], [8,10], [7,9], [10,12], [9,11], [8,14], [7,13], [13,14], [14,16], [13,15], [16,18], [15,17]]
        #     skeleton = []
        #     keypoint_captions = text_seg['answer'].split('\n')
        #     keypoint_pattern = r"<p>(.*?)<\/p>{<(\d+)><(\d+)>}"
        #     keypoint_list = target['all_points'].reshape(-1, 2) * np.array([W, H])
        #     vis_list = target['all_vis']
        #     matches_keypoint = []
        #     for kc in keypoint_captions:
        #         matches_keypoint.extend(re.findall(keypoint_pattern, kc))

        #     if len(target['bbox'])!=0:
        #         bbox = target['bbox'].clip(min=0.0, max=1.0)
        #         bbox = bbox*[target['size'][0],target['size'][1],target['size'][0],target['size'][1]]
        #         bbox = list(map(int,bbox))
        #         # print(bbox)
        #         x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        #         show_img = cv2.rectangle(show_img, (x1, y1), (x2, y2), (0,255,0), 3)

        #     for i in range(len(keypoint_list)):
        #         x, y = keypoint_list[i][0], keypoint_list[i][1]
        #         if vis_list[i] > 0:
        #             show_img = cv2.circle(show_img, (int(x), int(y)), 5,(0,0,255), -1)
        #         # show_img = cv2.circle(show_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        #         # cv2.putText(show_img, name, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
        #     # for name, x, y in matches_keypoint:
        #     #     x = float(x) / 448 * W
        #     #     y = float(y) / 448 * H
        #     #         # show_img = cv2.circle(show_img, (int(x), int(y)), 5,(0,0,255), -1)
        #     #     show_img = cv2.circle(show_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        #     #     cv2.putText(show_img, name, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 255), 1)
        #     for line_pair in skeleton:
        #         start_i = line_pair[0]
        #         end_i = line_pair[1]
        #         if vis_list[start_i] > 0 and vis_list[end_i] > 0:
        #             x1, y1, x2, y2 = keypoint_list[start_i][0], keypoint_list[start_i][1], keypoint_list[end_i][0], keypoint_list[end_i][1]
        #             show_img = cv2.line(show_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0),thickness=2)

        #     cv2.imwrite(os.path.join('debug', 'pose_estimation', os.path.basename(item['image_id'])), show_img)
        #     # print(item['image_id'])
        #     # print(text_seg['instruction_input'])
        #     # print(text_seg['answer'])
        # if self.task == 'parsing':
        #     show_img = image.permute(1,2,0).contiguous().numpy()
        #     show_img = (show_img * 255).astype('uint8')
        #     show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
        #     import scipy.io as scio
        #     colormap = scio.loadmat('/data/datasets/dataset_parsing/CIHP/human_colormap.mat')['colormap']
        #     seg_caption = text_seg['answer'].split('\n')
        #     class_seg_pattern = r"<p>(.*?)<\/p>{(.*?)}"
        #     seg_pattern = r"<(\d+)>"
        #     matches_class_seg = []
        #     for c in seg_caption:
        #         matches_class_seg.extend(re.findall(class_seg_pattern, c))
        #     H, W, _, = show_img.shape
        #     for i, (cls, seg) in enumerate(matches_class_seg):
        #         part = seg.split('<delim>')
        #         for p in part:
        #             poly_list = re.findall(seg_pattern, p)
        #             poly = np.array(poly_list).reshape(-1, 2).astype(float) / 100 * np.array([W, H])
        #             color = [(x * 255) for x in colormap[i]]
        #             center = np.mean(poly, axis=0)
        #             cv2.polylines(show_img, np.int32([poly]), True, color, 1)
        #             cv2.putText(show_img, cls, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
        #     cv2.imwrite(os.path.join('debug', 'parsing', os.path.basename(item['image_id'])), show_img)
        #     # print(item['image_id'])
        #     # print(text_seg['instruction_input'])
        #     # print(text_seg['answer'])

        # return
        ret_dict = {}
        ret_dict.update(text_seg)
        ret_dict.update({'task': self.task, 'image': image, 'image_id': item['image_id'], 'orig_image_size': item['size'], 
                         'image_size': (w, h), 'instance_id': item['instance_id']})
        if self.task == 'pose_estimation':
            ret_dict.update({'bbox': target['bbox'], 'center': target['center'], 'scale': target['scale'], 
                             'all_points': target['all_points'], 'all_vis': target['all_vis']})
        if self.task in ['appearance', 'modality', 'pose', 'relation']:
            ret_dict.update({'bbox': target['bbox']})
        # if self.task == 'parsing': # for debugging
        #     ret_dict.update({'class': target['class']})
        if debug_mode:
            return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': text_seg, 'image': image}
        return ret_dict

    def process_conv_and_target(self, raw_conv, target, multimage_mode=False):
        if multimage_mode:
            raise NotImplementedError
        return self.text_processor(raw_conv, target)

    def get_raw_item(self, index) -> Dict[str, Any]:
        raise NotImplementedError

    def segment_conv(self, source: List[Dict[str, Any]]): 
        role_map = {"human": 'instruction_input', "gpt": 'answer'}
        assert len(source) > 0
        assert source[0]['from'] == 'human'
        text_seg = {"instruction_input": [], "answer": []}
        for sentence in source:
            text_seg[role_map[sentence['from']]].append(sentence)
        #TODO: MULTI-ROUND CONVERSATION
        if len(text_seg['instruction_input']) == 1:
            text_seg['instruction_input']=text_seg['instruction_input'][0]['value']
        else:
            return NotImplementedError
        if len(text_seg['answer']) == 1:
            text_seg['answer']=text_seg['answer'][0]['value']
        else:
            return NotImplementedError
        return text_seg
    
class InstructTuningDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))

        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = item['question']
        answer = item['answer']

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret
    
class DenseCaptionDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))

        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size
        question = random.choice(self.question_template)
        answer = item['dense_caption'].replace('<p>', '').replace('</p>', '')

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class InstanceCaptionDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))

        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        answer = item[self.task].replace('<p>', '').replace('</p>', '')

        bbox = np.array(item['bbox']) / [rw, rh, rw, rh] # normalize to [0, 1]

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'bbox': bbox, 'refer': item['refer']},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class ParsingRECDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation = [ann for ann in self.annotation if not all(not sublist for sublist in ann['segmentation'])]
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))
        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        answer = ''

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'segmentation': item['segmentation'], 'bbox': item['bbox'], 'refer': item['refer']},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class InstanceGroundingDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation = [ann for ann in self.annotation if 'appearance_boxes' in ann.keys()]
        self.annotation = [ann for ann in self.annotation if not all(not sublist for sublist in ann['appearance_boxes'])]
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))

        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        answer = item[self.task]

        bbox = np.array(item['bbox']) / [rw, rh, rw, rh] # normalize to [0, 1]

        locs = [[np.array(box) / [rw, rh, rw, rh] for box in part_boxes] for part_boxes in item['appearance_boxes']]

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'bbox': bbox, 'refer': item['refer'], 'locs': locs},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class GRITGroundingDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation = [ann for ann in self.annotation if 'caption_bboxes' in ann.keys()]
        self.annotation = [ann for ann in self.annotation if not all(not sublist for sublist in ann['caption_bboxes'])]
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))

        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        answer = item['caption']

        locs = [[np.array(box) / [rw, rh, rw, rh] for box in part_boxes] for part_boxes in item['caption_bboxes']]

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'locs': locs},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class InstancePartRECDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation = [ann for ann in self.annotation if 'appearance_boxes' in ann.keys()]
        self.annotation = [ann for ann in self.annotation if not all(not sublist for sublist in ann['appearance_boxes'])]
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))

        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        answer = item['appearance']

        bbox = np.array(item['bbox']) / [rw, rh, rw, rh] # normalize to [0, 1]

        locs = [[np.array(box) / [rw, rh, rw, rh] for box in part_boxes] for part_boxes in item['appearance_boxes']]

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'bbox': bbox, 'refer': item['refer'], 'locs': locs},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class InstanceRECDataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))

        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        if self.task == 'inst_rec':
            task = random.choice(['appearance', 'pose', 'modality', 'relation'])
        else:
            task = self.task
        answer = item[task]
        # answer = ', '.join([item['appearance'], item['pose'], item['modality'], item['relation']])

        bbox = np.array(item['bbox']) / [rw, rh, rw, rh] # normalize to [0, 1]

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'bbox': bbox, 'refer': item['refer']},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class DetQADataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_identifier = DETECTION_IDENTIFIER
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))
        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        # question = "people"
        answer = random.choice(self.answer_template)

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'boxes': item['boxes']},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class PoseQADataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_identifier = POSE_IDENTIFIER
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))
        # # cv2 read
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = image[:, :, ::-1]  # BGR -> RGB
        # image = image / 255.0

        # PIL read
        # image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        # rh, rw, _ = image.shape
        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        # question = 'Predict <class> keypoints of <person> in the image <ImageHere>.'
        answer = random.choice(self.answer_template)

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'bbox': item['bbox'], 'points': item['position'],  'vis': item['vis'], 'image_id': image_path},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret

class ParsingQADataset(VisionQADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_identifier = PARSING_IDENTIFIER
        # self.task_identifier = DETECTION_IDENTIFIER
    
    def get_raw_item(self, index):
        item = self.annotation[index]
        image_path = osp.join(self.vis_root, item.get("image"))
        # # cv2 read
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # PIL read
        # image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        # rh, rw, _ = image.shape
        image = Image.open(image_path).convert('RGB')
        rw, rh = image.size

        question = random.choice(self.question_template)
        answer = random.choice(self.answer_template)

        ret = {
            'image_id': image_path,
            'image': image,
            'size': (rw, rh),
            'instance_id': item['instance_id'],
            'target': {'segmentation': item['segmentation'], 'bbox': item['bbox']},
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': answer,
                }
            ]
        }

        return ret
    
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor):
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.quantize_bins = 100

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_path = data['image']
        img_id = data['id']
        # question = '[refer] ' + data['question'].lower().strip()
        question = data['question'].lower().strip()
        answer = data['answer']
        # image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        return image, question, answer, img_id, image_path

class MultiChoiceEvalDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor):
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.quantize_bins = 100

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_path = data['image']
        img_id = data['id']
        question = data['question'] + ' Focus on the relations between multiple human.'
        choices = data['choices']
        answer = data['answer']
        answer = f'({answer})' + choices[answer]

        # choice_str = ' choose from one of the following options: '
        # for index, content in choices.items():
        #     choice_str += f'({index}) '
        #     choice_str += content
        # question = question + choice_str
        question = question.lower().strip()
        
        # image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        image = Image.open(image_path).convert('RGB')
        # w, h = image.size

        image = self.vis_processor(image)
        return image, question, choices, answer, img_id, image_path

class BBoxEvalDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor):
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.quantize_bins = 100

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_path = data['image']
        img_id = data['id']
        question = data['question']
        answer = [str(num) for num in data['answer']]
        answer = ' '.join(answer)
        
        question = question.lower().strip()
        question = '[refer] ' + question
        
        # image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        image = Image.open(image_path).convert('RGB')
        # w, h = image.size

        image = self.vis_processor(image)
        return image, question, answer, img_id, image_path

class MultiBBoxEvalDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor):
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.quantize_bins = 100

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_path = data['image']
        img_id = data['id']
        question = data['question']
        answer = [','.join([str(x) for x in bbox]) for bbox in data['answer']]
        answer = '<delim>'.join(answer)
        
        question = question.lower().strip()
        question = '[refer] ' + question
        
        # image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        image = Image.open(image_path).convert('RGB')
        # w, h = image.size

        image = self.vis_processor(image)
        return image, question, answer, img_id, image_path

class VizWizEvalDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, loaded_data, vis_processor):
        self.image_root = image_root
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor
        self.quantize_bins = 100

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_path = data['image']
        # question = '[refer] ' + data['question'].lower().strip()
        question = f"[vqa] The question is '{data['question']}' Based on the image, answer the question with a single word or phrase."
        question = question.lower().strip()
        answers = data['answers']
        answer_type = data['answer_type']
        answerable = data['answerable']
        # image = np.array(Image.open(image_path).convert('RGB')) / 255.0
        image = Image.open(os.path.join(self.image_root, image_path)).convert('RGB')
        image = self.vis_processor(image)
        return image, question, answers, answer_type, answerable, image_path