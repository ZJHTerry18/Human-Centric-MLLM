import re
import torch
import numpy as np
import random

import omegaconf
from omegaconf import OmegaConf
from minigpt4.common.registry import registry
from minigpt4.processors.blip_processors import BlipImageBaseProcessor, BlipCaptionProcessor
from minigpt4.common.utils import (
    DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, PERSON_PLACEHOLDER, CLASSES_PLACEHOLDER, BBOX_TEMPLATE, POINT_TOKEN
)
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

@registry.register_processor("blip2_text_pose_reg_train")
class Blip2TextPoseRegTrainProcessor(BlipCaptionProcessor):
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
        pose_prefix = ["The pose is", "Here is the pose", "The body pose is", "It is", "Sure, "]
        raw_conv[1]['value'] = random.choice(pose_prefix) + " " + POINT_TOKEN
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
        class_str = "all the"
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

@registry.register_processor("blip2_text_pose_reg_eval")
class Blip2TextPoseRegEvalProcessor(BlipCaptionProcessor):
    def __init__(self, image_token_num: int, sep_image_conv_front=False, use_im_start_end = False, precision=3, quantize_bins=100):
        self.image_token_num = image_token_num
        self.sep_image_conv_front = sep_image_conv_front
        self.use_im_start_end = use_im_start_end
        self.precision = precision
        self.quantize_bins = quantize_bins
    
    def __call__(self, raw_conv, target, task_identifier):
        ## construct output sentence
        ## construct output sentence
        raw_conv[1]['value'] = 'The pose is ' + POINT_TOKEN
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