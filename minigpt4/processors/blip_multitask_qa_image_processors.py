import re
import torch
import numpy as np

from minigpt4.common.registry import registry
from minigpt4.processors.blip_processors import BlipImageBaseProcessor, BlipCaptionProcessor
import omegaconf
from omegaconf import OmegaConf
import minigpt4.processors.transforms_det as Td
import minigpt4.processors.transforms_pose as Tp
import minigpt4.processors.transforms_parsing as Ts

@registry.register_processor("blip2_image_det_qa_train")
class Blip2ImageDetQATrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0, scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
    ):
        self.scales = list(scales)

        normalize = Td.Compose([
                Td.ToTensor(),
                Td.Normalize(mean, std)
            ])
        
        scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        # for unihcp
        self.transform = Td.Compose([
            Td.RandomHorizontalFlip(),
            Td.RandomSelect(
                Td.Compose([
                    # Td.PadToSquare(),
                    Td.RandomResize([image_size]),
                ]),
                Td.Compose([
                    Td.RandomResize([512, 640, 800]),
                    Td.RandomSizeCrop(384, 600),
                    # Td.PadToSquare(),
                    Td.RandomResize([image_size]),
                ]),
            ),
            normalize,
        ])

    def __call__(self, image, target):
        target["boxes"] = torch.as_tensor(target["boxes"])
        target["keep_index"] = list(range(target["boxes"].shape[0]))

        t_image, t_target = self.transform(image, target)
        th, tw = t_target["size"]
        t_target["boxes"] = t_target["boxes"]
 
        return t_image, t_target

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        scales = cfg.get("scales", [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            scales=scales,
        )

@registry.register_processor("blip2_image_det_qa_eval")
class Blip2ImageDetQAEvalProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0, scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
    ):
        super().__init__(mean=mean, std=std)
        self.scales = list(scales)

        normalize = Td.Compose([
                Td.ToTensor(),
                Td.Normalize(mean, std)
            ])

        self.transform = Td.Compose([
            # Td.PadToSquare(),
            Td.RandomResize([image_size]),
            normalize,
        ])

    def __call__(self, image, target):
        target["boxes"] = torch.as_tensor(target["boxes"])
        target["keep_index"] = list(range(target["boxes"].shape[0]))

        t_image, t_target = self.transform(image, target)
        th, tw = t_target["size"]
        t_target["boxes"] = t_target["boxes"]
 
        return t_image, t_target

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        scales = cfg.get("scales", [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            scales=scales,
        )


@registry.register_processor("blip2_image_pose_qa_train")
class Blip2ImagePoseQATrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, data_cfg,
    ):
        # self.dataset_name = data_cfg['dataset_name']
        self.mean = data_cfg['mean']
        self.std = data_cfg['std']

        if data_cfg['simp_aug']:
            pipeline = [
                Tp.TopDownRandomFlip(flip_prob=0.5),
                Tp.Resize(data_cfg['image_size']),
                Tp.ToUNTensor(),
                Tp.NormalizeTensor(mean=self.mean, std=self.std),
                Tp.Collect(keys=['image', 'joints_3d', 'joints_3d_visible', 'center', 'scale'])
            ]

        self.pipeline = Tp.ComposeX(pipeline)

        self.ann_info = {}
        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['all_keypoint_dict'] = dict(data_cfg['all_keypoint_dict'])
        self.ann_info['num_joints'] = len(self.ann_info['all_keypoint_dict'])
        self.ann_info['flip_pairs'] = data_cfg['flip_pairs']
        # self.ann_info['upper_body_ids'] = data_cfg['upper_body_ids']
        # self.ann_info['lower_body_ids'] = data_cfg['lower_body_ids']

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        # aspect_ratio = self.ann_info['image_size'][0] / self.ann_info['image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        # if np.random.rand() < 0.3:
        #     center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        # if w > aspect_ratio * h:
        #     h = w * 1.0 / aspect_ratio
        # elif w < aspect_ratio * h:
        #     w = h * aspect_ratio

        scale = np.array([w, h], dtype=np.float32)
        scale = scale * 1.25
        return center, scale
    
    def __call__(self, image, target):
        with_bbox_tag = len(target['bbox']) != 0
        if with_bbox_tag:
            bbox = target['bbox']
            #clean bbox
            if isinstance(image, np.ndarray):
                height, width, _ = image.shape
            else:
                width, height = image.size
            x, y, w, h = bbox
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if x2 >= x1 and y2 >= y1:
                bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                print("bbox size error")
                print(target['image_id'])
                # raise ValueError("bbox size error")
            center, scale = self._xywh2cs(*bbox)
        else:
            bbox=[]
            center= np.array([0, 0], dtype=np.float32)
            scale = np.array([0, 0], dtype=np.float32)

        visible= np.expand_dims(np.array(target['vis']), axis=1)
        keypoints_position = np.array(target['points']).reshape((-1,2))
        joints_3d = np.zeros((self.ann_info['num_joints'], 3), dtype=np.float32) 
        joints_3d[:, :2] = keypoints_position[:, :2]
        joints_3d_visible = np.zeros((self.ann_info['num_joints'], 3), dtype=np.float32) 
        joints_3d_visible[:, :2] = visible.clip(min=0, max=1)

        assert len(joints_3d_visible) == len(joints_3d)
        input = {
                'image': image,
                'center': center,
                'scale': scale,
                'bbox': np.array(bbox),
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible
            }
        input['ann_info'] = self.ann_info

        output = self.pipeline(input)
        idx_vis_keypoint_processed, = np.nonzero(output['joints_3d_visible'][:,0])
        idx_vis_keypoint_processed = idx_vis_keypoint_processed.tolist()
        output_points = output['joints_3d'][:, :2] / self.ann_info['image_size']

        t_target = {} 
        t_target['joints_3d_and_visible'] = output['joints_3d']
        t_target['joints_3d_and_visible'][:,2] = output['joints_3d_visible'][:,0]
        
        t_target['points'] = [output_points[i] for i in idx_vis_keypoint_processed]
        t_target['class'] = [self.ann_info['all_keypoint_dict'][i] for i in idx_vis_keypoint_processed]
        t_target['size'] = self.ann_info['image_size']

        if with_bbox_tag:
            cx, cy = output['center']
            w, h = output['scale']
            bbox = np.array([cx - w * 0.5,cy - h * 0.5, w, h])
            t_target['center'] = output['center']
            t_target['scale'] = output['scale']
            t_target['bbox'] = bbox / [t_target['size'][0], t_target['size'][1], t_target['size'][0], t_target['size'][1]]
        else:
            # x = t_target['joints_3d_and_visible'][:,0]
            # y = t_target['joints_3d_and_visible'][:,1]
            # idx = np.nonzero(t_target['joints_3d_and_visible'][:,2])[0]
            # vis_x = x[idx]
            # vis_y = y[idx]
            # x0, x1, y0, y1 = np.min(vis_x), np.max(vis_x), np.min(vis_y), np.max(vis_y)
            # bbox = np.array([x0, y0, x1 - x0, y1 - y0])
            # t_target['bbox'] = bbox / [t_target['size'][0], t_target['size'][1], t_target['size'][0], t_target['size'][1]]
            # t_target['center'] = np.array([0, 0], dtype=np.float32)
            # t_target['scale'] = np.array([0, 0], dtype=np.float32)

            t_target['bbox'] = []
            t_target['center'] = np.array([0, 0], dtype=np.float32)
            t_target['scale'] = np.array([0, 0], dtype=np.float32)

        ### for extra pose prediction module
        t_target['all_points'] = output_points.flatten().clip(min=0.0, max=1.0)
        t_target['all_vis'] = t_target['joints_3d_and_visible'][:, 2]
        ###
 
        t_image = output['image']

        return t_image, t_target

    @classmethod
    def from_config(cls, cfg=None):
        data_cfg = cfg.get("data_cfg", None)

        return cls(data_cfg)

@registry.register_processor("blip2_image_pose_qa_eval")
class Blip2ImagePoseQAEvalProcessor(Blip2ImagePoseQATrainProcessor):
    def __init__(
        self, data_cfg,
    ):
        self.mean = data_cfg['mean']
        self.std = data_cfg['std']
        pipeline = [
                Tp.Resize(data_cfg['image_size']),
                Tp.ToUNTensor(),
                Tp.NormalizeTensor(mean=self.mean, std=self.std),
                Tp.Collect(keys=['image', 'joints_3d', 'joints_3d_visible', 'center', 'scale'])
            ]

        self.pipeline = Tp.ComposeX(pipeline)

        self.ann_info = {}
        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['all_keypoint_dict'] = dict(data_cfg['all_keypoint_dict'])
        self.ann_info['num_joints'] = len(self.ann_info['all_keypoint_dict'])
        # self.ann_info['flip_pairs'] = data_cfg['flip_pairs']
        # self.ann_info['upper_body_ids'] = data_cfg['upper_body_ids']
        # self.ann_info['lower_body_ids'] = data_cfg['lower_body_ids']


@registry.register_processor("blip2_image_parsing_qa_train")
class Blip2ImageParsingQATrainProcessor(BlipImageBaseProcessor):
    def __init__(
         self, data_cfg,   
    ):
        self.mean = data_cfg['mean']
        self.std = data_cfg['std']
        self.is_flip = data_cfg.get("is_flip", False)
        self.left_right_pairs = data_cfg.get("left_right_pairs", None)

        self.transforms = Ts.Compose([
            Ts.Hflip(self.is_flip, self.left_right_pairs),
            Ts.Resize_image(data_cfg['image_size']),
            # TODO: add multi-scale transform
            # Ts.Rotate(data_cfg.get("is_rotate", False), degree=data_cfg.get("degree", 30),
            #                     p=data_cfg.get("possibility", 0.6), pad_val=data_cfg.get("pad_val", 0),
            #                     seg_pad_val=data_cfg.get("ignore_value", 255)),
            Ts.ToTensor(),
            Ts.Normalize(self.mean, self.std),
        ])

        self.all_parts_list = data_cfg['all_parts_list']
    
    def __call__(self, image, target):
        '''
            image(np.array): input image

            target(Dict):
                "segmentation": list of segmentation, each item is a list that contains all the polygon masks in one body part
                "bbox": [x, y, w, h], list
                "class": list of body parts, corresponding to each item in "segmentation"
        '''
        # convert segmentation polygon masks to np.array
        new_segmentation = []
        for part in target['segmentation']:
            new_part = []
            for t in part:
                t = np.array(t, dtype=np.float32).reshape(-1, 2)
                new_part.append(t)
            new_segmentation.append(new_part)
        target['segmentation'] = new_segmentation
        target['class'] = self.all_parts_list

        # convert bounding box to np.array
        if len(target['bbox']) > 0:
            target['bbox'] = np.array(target['bbox'])

        # data augmentations
        t_image, t_target = self.transforms(image, target)
        _, h, w = t_image.shape

        keep_index = [i for i, x in enumerate(target['segmentation']) if len(x) > 0]
        t_target['class'] = [x for i, x in enumerate(t_target['class']) if i in keep_index]
        t_target['segmentation'] = [[poly / np.array([w, h]) for poly in x] for i, x in enumerate(t_target['segmentation']) if i in keep_index]
        if len(target['bbox']) > 0:
            t_target['bbox'] = target['bbox'] / np.array([w, h, w, h])
        
        return t_image, t_target

    @classmethod
    def from_config(cls, cfg=None):
        data_cfg = cfg.get("data_cfg", None)

        return cls(data_cfg)

@registry.register_processor("blip2_image_parsing_qa_eval")
class Blip2ImageParsingQAEvalProcessor(Blip2ImageParsingQATrainProcessor):
    def __init__(
         self, data_cfg,   
    ):
        self.mean = data_cfg['mean']
        self.std = data_cfg['std']

        self.transforms = Ts.Compose([
            Ts.Resize_image(data_cfg['image_size']),
            Ts.ToTensor(),
            Ts.Normalize(self.mean, self.std),
        ])

        self.all_parts_list = data_cfg['all_parts_list']