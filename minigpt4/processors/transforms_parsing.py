import os
import cv2
import torch
import torchvision.transforms.functional as F
from torch.utils import data
import random
import inspect
import numpy as np
from typing import Tuple
import skimage.measure as skm

def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):
    xmax = xmin + w
    ymax = ymin + h

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax

def affine_points(pt, t) -> np.ndarray:
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T

def reform_polygon(poly, size):
    """
        reform the polygons out of image after crop/rotate augmentations
    """
    w, h = size
    out_of_width = (poly[:, 0] < 0) | (poly[:, 0] >= w)
    out_of_height = (poly[:, 1] < 0) | (poly[:, 1] >= h)
    if out_of_width.all() or out_of_height.all():
        return None

    # move coordinates out of image boundary onto the boundary
    poly[:, 0] = poly[:, 0].clip(min=0, max=w - 1)
    poly[:, 1] = poly[:, 1].clip(min=0, max=h - 1)

    # remove redundant points on boundary
    poly = np.unique(poly, axis=0)
    poly = skm.approximate_polygon(poly, tolerance=0.0)

    return poly


class AffineTransform(object):
    """scale+rotation"""
    def __init__(self,
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35)
                 rotation: Tuple[int, int] = None,   # e.g. (-45, 45)
                 fixed_size: Tuple[int, int] = (384,384)):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size
        # self.negative_solver = np.frompyfunc(lambda x: 0 if x < 0 else x, 1, 1)

    def __call__(self, img, target):
        src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["bbox"], fixed_size=self.fixed_size)
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin
        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2])
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0])  # top middle
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2])  # right middle

        if self.scale is not None:
            scale = random.uniform(*self.scale)
            src_w = src_w * scale
            src_h = src_h * scale
            src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
            src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        # todo : check src_p2 > 0 and src_p3 > 0
        
        # if self.rotation is not None:
        #     angle = random.randint(*self.rotation)  # 角度制
        #     angle = angle / 180 * math.pi  # 弧度制
        #     src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
        #     src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵

        reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

        # 对图像进行仿射变换
        resize_img = cv2.warpAffine(img,
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR)

        if "keypoints" in target:
            kps = target["keypoints"]
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            # kps[mask] = self.negative_solver(affine_points(kps[mask], trans))
            kps[mask] = affine_points(kps[mask], trans)
            target["keypoints"] = kps
        
        if "segmentation" in target:
            polys = []
            for point in target["segmentation"]:
                # polys.append(self.negative_solver(affine_points(point, trans)))
                polys.append(affine_points(point, trans))

            target['segmentation'] = polys 


        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return resize_img, target

class Compose:
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, data, label=None):
        for a in self.augs:
            # import pdb;pdb.set_trace()
            data, label = a(data, label)

        return data, label

    def __repr__(self):
        return 'Compose'


class Hflip:
    def __init__(self, p=False, left_right_pairs=None):
        self.p = p
        self.left_right_pairs = left_right_pairs

    def __call__(self, image, target):
        if not self.p:
            return image, target

        assert len(target), "hflip parsing needs label to map left and right pairs"
        flip = random.randrange(2) * 2 -1

        if isinstance(image, np.ndarray):
            image = image[:, ::flip, :]
            width = image.shape[1]
        else:
            if flip == -1:
                image = F.hflip(image)
            width = image.size[0]

        if flip == -1 and len(target['bbox']) > 0:
            bbox = target['bbox']
            bbox[0] = width - bbox[0] - bbox[2]
            target['bbox'] = bbox

        if flip == -1:
            # flip the column coordinates of targets
            new_segmentation = []
            for part in target['segmentation']:
                new_part = []
                for t in part:
                    t[:, 0] = width - 1 - t[:, 0]
                    new_part.append(t)
                new_segmentation.append(new_part)
            if self.left_right_pairs is not None:
                # exchange left and right pairs
                for lr_pair in self.left_right_pairs:
                    li, ri = lr_pair[0], lr_pair[1]
                    new_segmentation[li], new_segmentation[ri] = new_segmentation[ri], new_segmentation[li]
            target['segmentation'] = new_segmentation

        return image, target

    def __repr__(self):
        return f'Hflip with {self.left_right_pairs}'

class Resize_image:
    def __init__(self, crop_size):
        self.size = crop_size

    def __call__(self, image, target):
        if isinstance(image, np.ndarray):
            oh, ow, _ = image.shape
            image = cv2.resize(image, tuple(self.size), interpolation=cv2.INTER_LINEAR)
        else:
            ow, oh = image.size
            image = F.resize(image, self.size, interpolation=F.InterpolationMode.BICUBIC)
        nw, nh = self.size
        rw = float(nw) / ow
        rh = float(nh) / oh
        
        if len(target['bbox']) > 0:
            target['bbox'] = target['bbox'] * np.array([rw, rh, rw, rh])
            
        new_segmentation = []
        for part in target['segmentation']:
            new_part = []
            for t in part:
                t = t * np.array([rw, rh])
                new_part.append(t)
            new_segmentation.append(new_part)
        target['segmentation'] = new_segmentation
        
        return image, target

    def __repr__(self):
        return f"Resize with {self.size}"

class Resize_image_eval:
    def __init__(self, crop_size):
        self.size = crop_size

    def __call__(self, image, label=None):
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, tuple(self.size), interpolation=cv2.INTER_LINEAR)
        else:
            image = F.resize(image, self.size, interpolation=F.InterpolationMode.BICUBIC)
        label = cv2.resize(label, (1000, 1000), interpolation = cv2.INTER_LINEAR_EXACT)
        return image, label

    def __repr__(self):
        return f"Resize_eval with {self.size}"

class Multi_scale:
    def __init__(self, is_multi_scale, scale_factor=11,
                 center_crop_test=False, base_size=480,
                 crop_size=(480, 480),
                 ignore_label=-1):
        self.is_multi_scale = is_multi_scale
        self.scale_factor = scale_factor
        self.center_crop_test = center_crop_test
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

    def multi_scale_aug(self, image, label=None,
            rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label

    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label

    def center_crop(self, image, label):
        h, w = image.shape[:2]
        x = int(round((w - self.crop_size[1]) / 2.))
        y = int(round((h - self.crop_size[0]) / 2.))
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def __call__(self, image, label=None):
        if self.is_multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        # if center_crop_test:
        #     image, label = self.image_resize(image,
        #                                      self.base_size,
        #                                      label)
        #     image, label = self.center_crop(image, label)

        return image, label

# class Normalize:
#     def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#         self.mean = mean
#         self.std = std

#     def __call__(self, image, label = None):
#         image = image.astype(np.float32)
#         image = image / 255.0
#         image -= self.mean
#         image /= self.std
#         return image, label

#     def __repr__(self):
#         return f"Normalize with {self.mean} and {self.std}"

class Transpose:
    def __repr__(self):
        return 'Transpose'

    def __call__(self, image, label=None):
        return image.transpose((2,0,1)), label

class Rotate:

    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }

    def __init__(self, is_rotate=False, degree=0, p=0.5, pad_val=0, seg_pad_val=255,
                 center=None, auto_bound=False):
        self.is_rotate = is_rotate
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        self.p = p
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound=auto_bound

    def __call__(self, image, target):
        if not self.is_rotate or random.random() < self.p:
            return image, target
        degree = random.uniform(min(*self.degree), max(*self.degree))
        image = self._rotate(
            image,
            angle=degree,
            border_value=self.pad_val,
            center=self.center,
            auto_bound=self.auto_bound)
        return image, target

    def _rotate(self, img, target, angle, center=None, scale=1.0, border_value=0, interpolation='bilinear', auto_bound=False):
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated_image = cv2.warpAffine(
            img,
            matrix, (w, h),
            flags=self.cv2_interp_codes[interpolation],
            borderValue=border_value)
        
        new_segmentation = []
        for part in target['segmentation']:
            new_part = []
            for t in part:
                rotated_t = affine_points(t, matrix)
                rotated_t = reform_polygon(rotated_t, (w, h))
                if rotated_t is not None:
                    new_part.append(rotated_t)
            new_segmentation.append(new_part)
        target['segmentation'] = new_segmentation
        return rotated_image, target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str

class ToTensor(object):
    def __call__(self, img, target):
        if isinstance(img, np.ndarray):
            img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)),dtype=torch.float32)
        else:
            return F.to_tensor(img), target
        return img, target
        

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

# copied from https://github.com/open-mmlab/mmsegmentation/blob/2d66179630035097dcae08ee958f60d4b5a7fcae/mmseg/datasets/pipelines/transforms.py
class PhotoMetricDistortion:
    def __init__(self,
                 is_PhotoMetricDistortio=False,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.is_PhotoMetricDistortio = is_PhotoMetricDistortio

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0,1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0,1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0,1):
            img = bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0,1):
            img = bgr2hsv(img)
            img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = hsv2bgr(img)
        return img

    def __call__(self, img, label=None):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if not self.is_PhotoMetricDistortio:
            return img, label
        img = img
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0,1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        # results['img'] = img
        return img, label

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return

def convert_color_factory(src: str, dst: str):

    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img: np.ndarray) -> np.ndarray:
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.
    Args:
        img (ndarray or str): The input image.
    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

bgr2hls = convert_color_factory('bgr', 'hls')

hls2bgr = convert_color_factory('hls', 'bgr')