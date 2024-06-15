import re
import torch
import numpy as np
import random

import omegaconf
from omegaconf import OmegaConf
from minigpt4.common.registry import registry
from minigpt4.processors.blip_processors import BlipImageBaseProcessor, BlipCaptionProcessor
from minigpt4.common.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, BBOX_TEMPLATE, POINTS_TEMPLATE, PARSING_TEMPLATE, PERSON_PLACEHOLDER, CLASSES_PLACEHOLDER
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

@registry.register_processor("blip2_text_det_qa_train")
class Blip2TextDetQATrainProcessor(BlipCaptionProcessor):
    def __init__(self, image_token_num: int, sep_image_conv_front=False, use_im_start_end = False, precision=3, quantize_bins=100,
                 max_bbox_num=50):
        self.image_token_num = image_token_num
        self.sep_image_conv_front = sep_image_conv_front
        self.use_im_start_end = use_im_start_end
        self.precision = precision
        self.quantize_bins = quantize_bins
        self.max_bbox_num = max_bbox_num
    
    def __call__(self, raw_conv, target, task_identifier):
        ## construct output sentence
        output_subsentences = []
        bbox_list = target["boxes"]
        if bbox_list.shape[0] > 0:
            # for images with too many bounding boxes, drop the smallest boxes
            areas = bbox_list[:, 2] * bbox_list[:, 3]
            _, sorted_indices = torch.sort(areas, descending=True)
            bbox_list = bbox_list[sorted_indices][:self.max_bbox_num, :]
            rand_perm = torch.randperm(bbox_list.shape[0])
            bbox_list = bbox_list[rand_perm]
            # convert [x,y,w,h] to [x1,y1,x2,y2]
            bbox_list[:, 2:4] += bbox_list[:, 0:2]
            bbox_list = bbox_list.clamp(min=0.0, max=1.0)
            for i in range(bbox_list.shape[0]):
                bbox = bbox_list[i].numpy().tolist()
                bbox = [int(x * self.quantize_bins) for x in bbox]
                bbox_str = BBOX_TEMPLATE.format(*bbox)
                output_subsentences.append(bbox_str)
            
            output_sentence = "<delim>".join(output_subsentences)
            raw_conv[1]['value'] = raw_conv[1]['value'] + " " + output_sentence
        else:
            raw_conv[1]['value'] = 'There are no people.'
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()

        ## construct input sentence
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()

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
        image_token_num = cfg.get('num_query_token', 256)
        sep_image_conv_front = cfg.get('sep_image_conv_front', False) 
        use_im_start_end = cfg.get('use_im_start_end', False) 
        precision = cfg.get('precision', 2)
        quantize_bins = cfg.get("quantize_bins", 100)
        max_bbox_num = cfg.get("max_bbox_num", 30)
        
        return cls(image_token_num=image_token_num,
                   sep_image_conv_front=sep_image_conv_front,
                   use_im_start_end=use_im_start_end,
                   precision=precision,
                   quantize_bins=quantize_bins,
                   max_bbox_num=max_bbox_num)

@registry.register_processor("blip2_text_det_qa_eval")
class Blip2TextDetQAEvalProcessor(BlipCaptionProcessor):
    def __init__(self, image_token_num: int, sep_image_conv_front=False, use_im_start_end = False, precision=3, quantize_bins=100):
        self.image_token_num = image_token_num
        self.sep_image_conv_front = sep_image_conv_front
        self.use_im_start_end = use_im_start_end
        self.precision = precision
        self.quantize_bins = quantize_bins
    
    def __call__(self, raw_conv, target, task_identifier):
        ## construct output sentence
        output_subsentences = []
        bbox_list = target["boxes"]
        if bbox_list.shape[0] > 0:
            # convert [x,y,w,h] to [x1,y1,x2,y2]
            bbox_list[:, 2:4] += bbox_list[:, 0:2]
            bbox_list = bbox_list.clamp(min=0.0, max=1.0)
            for i in range(bbox_list.shape[0]):
                bbox = bbox_list[i].numpy().tolist()
                bbox = [int(x * self.quantize_bins) for x in bbox]
                bbox_str = BBOX_TEMPLATE.format(*bbox)
                output_subsentences.append(bbox_str)
            
            output_sentence = "<delim>".join(output_subsentences)
            raw_conv[1]['value'] = raw_conv[1]['value'] + " " + output_sentence
        else:
            raw_conv[1]['value'] = 'There are no people.'
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()

        ## construct input sentence
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()
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
        image_token_num = cfg.get('num_query_token', 256)
        sep_image_conv_front = cfg.get('sep_image_conv_front', False) 
        use_im_start_end = cfg.get('use_im_start_end', False) 
        precision = cfg.get('precision', 2)
        quantize_bins = cfg.get("quantize_bins", 100)
        
        return cls(image_token_num=image_token_num,
                   sep_image_conv_front=sep_image_conv_front,
                   use_im_start_end=use_im_start_end,
                   precision=precision,
                   quantize_bins=quantize_bins)


@registry.register_processor("blip2_text_pose_qa_train")
class Blip2TextPoseQATrainProcessor(BlipCaptionProcessor):
    def __init__(self, image_token_num: int, sep_image_conv_front=False, use_im_start_end = False, precision=3, quantize_bins=100,
                 sample_p=0.5, sample_range=[1, 5]):
        self.image_token_num = image_token_num
        self.sep_image_conv_front = sep_image_conv_front
        self.use_im_start_end = use_im_start_end
        self.precision = precision
        self.quantize_bins = quantize_bins
        self.sample_p = sample_p
        self.sample_range = sample_range
    
    def __call__(self, raw_conv, target, task_identifier):
        ## construct output sentence
        output_subsentences = []
        points_list = target["points"]
        class_list = target["class"]
        old_length = len(points_list)
        if len(points_list) > 0:
            # randomly perform keypoint sampling
            if len(points_list) > 1 and random.random() < self.sample_p:
                k = random.randint(self.sample_range[0], min(self.sample_range[1], len(points_list) - 1))
                sample_indices = random.sample(range(len(points_list)), k)
                points_list = [points_list[i] for i in sample_indices]
                class_list = [class_list[i] for i in sample_indices]

            for i in range(len(points_list)):
                point = points_list[i].clip(min=0.0, max=1.0).tolist()
                point = [int(x * self.quantize_bins) for x in point]
                cls = class_list[i]
                keypoint_str = POINTS_TEMPLATE.format(cls, *point)
                output_subsentences.append(keypoint_str)
            
            output_sentence = "\n".join(output_subsentences)
            raw_conv[1]['value'] = raw_conv[1]['value'] + " " + output_sentence
        else:
            raw_conv[1]['value'] = 'There are no keypoints.'
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()
        new_length = len(points_list)

        ## construct input sentence
        if len(target['bbox']) > 0: # with bounding box
            bbox = target['bbox']
            bbox[2:4] += bbox[0:2]
            bbox = bbox.clip(min=0.0, max=1.0)
            bbox = [int(x * self.quantize_bins) for x in bbox]
            person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
        else: # without bounding box
            person_refer_str = "the person"

        if old_length == new_length:
            class_str = "all the"
        else:
            class_str = "the " + ", ".join(class_list)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(CLASSES_PLACEHOLDER, class_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()

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
        image_token_num = cfg.get('num_query_token', 256)
        sep_image_conv_front = cfg.get('sep_image_conv_front', False) 
        use_im_start_end = cfg.get('use_im_start_end', False) 
        precision = cfg.get('precision', 2)
        quantize_bins = cfg.get("quantize_bins", 100)
        sample_p = cfg.get("sample_p", 0.1)
        sample_range = cfg.get("sample_range", [1, 5])
        
        return cls(image_token_num=image_token_num,
                   sep_image_conv_front=sep_image_conv_front,
                   use_im_start_end=use_im_start_end,
                   precision=precision,
                   quantize_bins=quantize_bins,
                   sample_p=sample_p,
                   sample_range=sample_range)

@registry.register_processor("blip2_text_pose_qa_eval")
class Blip2TextPoseQAEvalProcessor(BlipCaptionProcessor):
    def __init__(self, image_token_num: int, sep_image_conv_front=False, use_im_start_end = False, precision=3, quantize_bins=100):
        self.image_token_num = image_token_num
        self.sep_image_conv_front = sep_image_conv_front
        self.use_im_start_end = use_im_start_end
        self.precision = precision
        self.quantize_bins = quantize_bins
    
    def __call__(self, raw_conv, target, task_identifier):
        ## construct output sentence
        output_subsentences = []
        points_list = target["points"]
        class_list = target["class"]
        if len(points_list) > 0:
            for i in range(len(points_list)):
                point = points_list[i].clip(min=0.0, max=1.0).tolist()
                point = [int(x * self.quantize_bins) for x in point]
                cls = class_list[i]
                keypoint_str = POINTS_TEMPLATE.format(cls, *point)
                output_subsentences.append(keypoint_str)
            
            output_sentence = "\n".join(output_subsentences)
            raw_conv[1]['value'] = raw_conv[1]['value'] + " " + output_sentence
        else:
            raw_conv[1]['value'] = 'There are no keypoints.'
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()

        ## construct input sentence
        if len(target['bbox']) > 0: # with bounding box
            bbox = target['bbox']
            bbox[2:4] += bbox[0:2]
            bbox = bbox.clip(min=0.0, max=1.0)
            bbox = [int(x * self.quantize_bins) for x in bbox]
            person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
        else: # without bounding box
            person_refer_str = "the person"
        class_list = target["class"]
        # class_str = "the " + ", ".join(class_list)
        class_str = "all the" # always detect all the keypoints when evaluating
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(CLASSES_PLACEHOLDER, class_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()
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
        image_token_num = cfg.get('num_query_token', 256)
        sep_image_conv_front = cfg.get('sep_image_conv_front', False) 
        use_im_start_end = cfg.get('use_im_start_end', False) 
        precision = cfg.get('precision', 2)
        quantize_bins = cfg.get("quantize_bins", 100)
        
        return cls(image_token_num=image_token_num,
                   sep_image_conv_front=sep_image_conv_front,
                   use_im_start_end=use_im_start_end,
                   precision=precision,
                   quantize_bins=quantize_bins)


@registry.register_processor("blip2_text_parsing_qa_train")
class Blip2TextParsingQATrainProcessor(BlipCaptionProcessor):
    def __init__(self, image_token_num: int, sep_image_conv_front=False, use_im_start_end = False, precision=3, quantize_bins=100,
                 sample_p=0.5, sample_range=[1, 5], max_polygon_len=250):
        self.image_token_num = image_token_num
        self.sep_image_conv_front = sep_image_conv_front
        self.use_im_start_end = use_im_start_end
        self.precision = precision
        self.quantize_bins = quantize_bins
        self.sample_p = sample_p
        self.sample_range = sample_range
        self.max_polygon_len = max_polygon_len
    
    def __call__(self, raw_conv, target, task_identifier):
        ## construct output sentence
        output_subsentences = []
        segs_list = target["segmentation"]
        class_list = target["class"]
        segs_polylen = []
        old_length = len(segs_list)
        if len(segs_list) > 0:
            for i in range(len(segs_list)):
                l = 0
                for poly in segs_list[i]:
                    l += poly.shape[0] * 2
                segs_polylen.append(l)
            # random drop body parts until total polygon length is under maximum length
            while sum(segs_polylen) > self.max_polygon_len:
                remove_index = random.choice(range(len(segs_list)))
                del segs_list[remove_index]
                del class_list[remove_index]
                del segs_polylen[remove_index]
            
            # randomly perform part sampling
            if len(segs_list) > 1 and random.random() < self.sample_p:
                k = random.randint(self.sample_range[0], min(self.sample_range[1], len(segs_list) - 1))
                sample_indices = random.sample(range(len(segs_list)), k)
                segs_list = [segs_list[i] for i in sample_indices]
                class_list = [class_list[i] for i in sample_indices]
                segs_polylen = [segs_polylen[i] for i in sample_indices]
            for i in range(len(segs_list)):
                poly_list = []
                for poly in segs_list[i]:
                    poly = poly.clip(min=0.0, max=1.0).flatten().tolist()  
                    poly = [int(x * self.quantize_bins) for x in poly]
                    poly_template = "{{" + "<{}>" * len(poly) + "}}"
                    poly_str = poly_template.format(*poly)
                    poly_list.append(poly_str)
                part_str = "<delim>".join(poly_list)
                cls = class_list[i]
                parsing_str = PARSING_TEMPLATE.format(cls, part_str)
                output_subsentences.append(parsing_str)
            
            output_sentence = "\n".join(output_subsentences)
            raw_conv[1]['value'] = raw_conv[1]['value'] + " " + output_sentence
        else:
            raw_conv[1]['value'] = 'There are no body parts.'
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()
        new_length = len(segs_list)

        ## construct input sentence
        if len(target['bbox']) > 0: # with bounding box
            bbox = target['bbox']
            bbox[2:4] += bbox[0:2]
            bbox = bbox.clip(min=0.0, max=1.0)
            bbox = [int(x * self.quantize_bins) for x in bbox]
            person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
        else: # without bounding box
            person_refer_str = "the person"

        if old_length == new_length: # all the body parts are preserved
            class_str = "all the"
        else:
            class_str = "the " + ", ".join(class_list)

        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(CLASSES_PLACEHOLDER, class_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()

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
        image_token_num = cfg.get('num_query_token', 256)
        sep_image_conv_front = cfg.get('sep_image_conv_front', False) 
        use_im_start_end = cfg.get('use_im_start_end', False) 
        precision = cfg.get('precision', 2)
        quantize_bins = cfg.get("quantize_bins", 100)
        sample_p = cfg.get("sample_p", 0.5)
        sample_range = cfg.get("sample_range", [1, 5])
        max_polygon_len = cfg.get("max_polygon_len", 200)
        
        return cls(image_token_num=image_token_num,
                   sep_image_conv_front=sep_image_conv_front,
                   use_im_start_end=use_im_start_end,
                   precision=precision,
                   quantize_bins=quantize_bins,
                   sample_p=sample_p,
                   sample_range=sample_range,
                   max_polygon_len=max_polygon_len)


@registry.register_processor("blip2_text_parsing_qa_eval")
class Blip2TextParsingQAEvalProcessor(BlipCaptionProcessor):
    def __init__(self, image_token_num: int, sep_image_conv_front=False, use_im_start_end = False, precision=3, quantize_bins=100):
        self.image_token_num = image_token_num
        self.sep_image_conv_front = sep_image_conv_front
        self.use_im_start_end = use_im_start_end
        self.precision = precision
        self.quantize_bins = quantize_bins
    
    def __call__(self, raw_conv, target, task_identifier):
        ## construct output sentence
        output_subsentences = []
        segs_list = target["segmentation"]
        class_list = target["class"]
        if len(segs_list) > 0:
            for i in range(len(segs_list)):
                poly_list = []
                for poly in segs_list[i]:
                    poly = poly.clip(min=0.0, max=1.0).flatten().tolist()
                    poly = [int(x * self.quantize_bins) for x in poly]
                    poly_template = "{{" + "<{}>" * len(poly) + "}}"
                    poly_str = poly_template.format(*poly)
                    poly_list.append(poly_str)
                part_str = "<delim>".join(poly_list)
                cls = class_list[i]
                parsing_str = PARSING_TEMPLATE.format(cls, part_str)
                output_subsentences.append(parsing_str)
            
            output_sentence = "\n".join(output_subsentences)
            raw_conv[1]['value'] = raw_conv[1]['value'] + " " + output_sentence
        else:
            raw_conv[1]['value'] = 'There are no body parts.'
        raw_conv[1]['value'] = raw_conv[1]['value'].lower()

        ## construct input sentence
        if len(target['bbox']) > 0: # with bounding box
            bbox = target['bbox']
            bbox[2:4] += bbox[0:2]
            bbox = bbox.clip(min=0.0, max=1.0)
            bbox = [int(x * self.quantize_bins) for x in bbox]
            person_refer_str = "the person at " + BBOX_TEMPLATE.format(*bbox)
        else: # without bounding box
            person_refer_str = "the person"
        # class_str = "the " + ", ".join(class_list)
        class_str = "all the"  # always output all parts when evaluating
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(PERSON_PLACEHOLDER, person_refer_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(CLASSES_PLACEHOLDER, class_str)
        raw_conv[0]['value'] = raw_conv[0]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        raw_conv[0]['value'] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + " " +  task_identifier + " " + raw_conv[0]['value'].lower()
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
        image_token_num = cfg.get('num_query_token', 256)
        sep_image_conv_front = cfg.get('sep_image_conv_front', False) 
        use_im_start_end = cfg.get('use_im_start_end', False) 
        precision = cfg.get('precision', 2)
        quantize_bins = cfg.get("quantize_bins", 100)
        
        return cls(image_token_num=image_token_num,
                   sep_image_conv_front=sep_image_conv_front,
                   use_im_start_end=use_im_start_end,
                   precision=precision,
                   quantize_bins=quantize_bins)