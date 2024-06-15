"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import json
import warnings
import re
import copy
import torch.distributed as dist
from collections import OrderedDict
import numpy as np
import cv2
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import torch
import logging
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.common.utils import BBOX_TEMPLATE, CLASSES_PLACEHOLDER
from minigpt4.common.dist_utils import main_process
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.base_task import BaseTask
from minigpt4.processors.transforms_post import get_warp_matrix


@registry.register_task("pose_eval")
class PoseEvalTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, image_size,sigmas,use_area, report_metric=True, quantize_bins=0, image_root=None, all_keypoint_dict={}, skeleton=None):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.quantize_bins = quantize_bins
        self.image_root = image_root
        self.report_metric = report_metric
        self.image_size = np.array(image_size)

        rev_keypoint_dict = dict([val,key] for key,val in all_keypoint_dict.items())  
        new_all_keypoint_dict = {}
        for k in rev_keypoint_dict.keys():
            new_all_keypoint_dict[k.replace('_','')]=rev_keypoint_dict[k]

        self.all_keypoint_dict = new_all_keypoint_dict 
        self.skeleton = skeleton

        self.sigmas = np.array(sigmas)
        self.use_area = use_area

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        dataset_cfg = cfg.datasets_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len

        # only support evaluating one dataset at one time
        image_root = None
        for k in dataset_cfg.keys():
            if 'val' in dataset_cfg[k].build_info.ann_path.keys():
                all_keypoint_dict = dataset_cfg[k].vis_processor.eval.data_cfg['all_keypoint_dict']
                skeleton = dataset_cfg[k].vis_processor.eval.data_cfg['skeleton']
                sigmas = dataset_cfg[k].vis_processor.eval.data_cfg.get('sigmas', None)
                use_area = dataset_cfg[k].vis_processor.eval.data_cfg['use_area']
                image_root = dataset_cfg[k].build_info.image_path
                image_size = dataset_cfg[k].vis_processor.eval.data_cfg['image_size']
        quantize_bins = cfg.model_cfg.get("quantize_bins", 100)
        report_metric = run_cfg.get("report_metric", True)
        
        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            image_size=image_size,
            quantize_bins=quantize_bins,
            image_root=image_root,
            sigmas=sigmas,
            use_area=use_area,
            report_metric=report_metric,
            all_keypoint_dict=all_keypoint_dict,
            skeleton=skeleton,

        )

    def convert_text_to_keypoint(self, image_size, caption):
        joints_pos = np.zeros((len(self.all_keypoint_dict), 2), dtype=np.float32)
        vis_list = np.zeros(len(self.all_keypoint_dict))
        w, h = image_size
        keypoint_captions = caption.split('\n')
        keypoint_pattern = r"<p>(.*?)<\/p>{<(\d+)><(\d+)>}"
        matches_keypoint = []
        for kc in keypoint_captions:
            matches_keypoint.extend(re.findall(keypoint_pattern, kc))

        for name, x, y in matches_keypoint:
            name = name.replace('_','')
            x = float(x) / self.quantize_bins * w
            y = float(y) / self.quantize_bins * h
            if name in self.all_keypoint_dict.keys():
                idx = self.all_keypoint_dict[name]
                if vis_list[idx] == 1:
                    warnings.warn('Duplicate keypoint!')
                vis_list[idx] = 1
                joints_pos[idx] = [x, y]
        return joints_pos.astype(np.int32).tolist(), vis_list.astype(np.int32).tolist()

    
    def valid_step(self, model, samples):
        results = []
        gt_results = []

        # predict all keypoints at one time
        images = samples["image"]
        texts = samples["instruction_input"] if "instruction_input" in samples.keys() else samples["conv_q"]
        img_ids = samples["image_id"]
        instance_ids = samples["instance_id"].cpu().numpy().tolist()
        gts = samples["answer"] if "answer" in samples.keys() else samples["conv_a"]
        # gts = samples['target']
        orig_img_sizes = samples["orig_image_size"]
        img_sizes = samples["image_size"]
        centers = samples['center'].cpu().numpy().tolist()
        scales = samples['scale'].cpu().numpy().tolist()
        bboxes = samples['bbox'].cpu().numpy().tolist() if len(samples['bbox']) > 0 else [[] for _ in range(len(gts))]

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            images,
            texts,
            max_new_tokens=self.max_len,
            num_beams=self.num_beams,
        )
        print('image: ', img_ids)
        print('input: ', texts)
        # print('gt: ', gts)
        # print('output: ', captions)
        
        orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        img_w = img_sizes[0].cpu().numpy().tolist()
        img_h = img_sizes[1].cpu().numpy().tolist()
        for caption, img_id, instance_id, ow, oh, w, h, bbox in zip(captions, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h, bboxes):
            joints_pos, vis_list = self.convert_text_to_keypoint((ow, oh), caption)
            if len(bbox)>0:
                bbox = [bbox[0]*ow, bbox[1]*oh, bbox[2]*ow, bbox[3]*oh]
            results.append({"raw_caption": caption, "keypoints_pos": (joints_pos, vis_list), "image": img_id, "id": instance_id, "height": oh, "width": ow, "bbox": bbox})
        for gt, img_id, instance_id, ow, oh, w, h, bbox in zip(gts, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h, bboxes):
            joints_pos, vis_list = self.convert_text_to_keypoint((ow, oh), gt)
            if len(bbox)>0:
                bbox = [bbox[0]*ow, bbox[1]*oh, bbox[2]*ow, bbox[3]*oh]
            gt_results.append({"keypoints_pos": (joints_pos, vis_list), "image": img_id, "id": instance_id, "height": oh, "width": ow, "bbox": bbox})
        
        # visualize pose results
        self._visualize_pose_results(results, gt_results)
        
        return results, gt_results

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []
        gt_results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            # samples = prepare_sample(samples, cuda_enabled=False)

            eval_output, gt_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)
            gt_results.extend(gt_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return (results, gt_results)
    
    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        val_result, gt_result = val_result
        eval_result_file, eval_result = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )
        gt_result_file, gt_result = self.save_result(
            result=gt_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_gt_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )


        if self.report_metric:
            metrics = self._report_coco_pose_metrics(epoch, eval_result, gt_result, split_name)
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_coco_pose_metrics(self, epoch, pred_results, gts, split_name):
        #ToDo
        gt_res = {"images": [], "annotations": [], "categories": [{"supercategory": "person","id": 1,"name": "person"}]}

        image_id = 0
        instance_id = 0
        img2id = {}
        for item in gts:
            image = item["image"]
            img2id[image] = image_id
            gt_res["images"].append({"file_name": image, "id": instance_id})
            joints_pos, vis_list = item["keypoints_pos"]
            joints_pos = np.array(joints_pos)
            vis_list = np.array(vis_list)
            keypoints = np.concatenate((joints_pos,vis_list.reshape((-1,1))), axis=1).flatten().tolist()
            
            bbox = item['bbox']
            if len(bbox) > 0:
                bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
            else:
                x = joints_pos[:,0].astype(np.int32)
                y = joints_pos[:,1].astype(np.int32)
                idx = np.nonzero(vis_list)[0]
                vis_x = x[idx]
                vis_y = y[idx]
                x0,x1,y0,y1 = np.min(vis_x), np.max(vis_x), np.min(vis_y), np.max(vis_y)
                bbox = [int(x0),int(y0),int(x1-x0),int(y1-y0)]
            area = bbox[2] * bbox[3]

            res = {
                'image_id': instance_id,
                'keypoints': keypoints,
                # 'num_keypoints': len(keypoints),
                'category_id': 1,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
                'id': instance_id
            }
            gt_res["annotations"].append(res)
            instance_id += 1
            image_id += 1
        gt_file_name = "{}_coco_gt.json".format(split_name)
        gt_file_path = os.path.join(registry.get_path("result_dir"), gt_file_name)
        with open(gt_file_path, 'w') as f:
            json.dump(gt_res, f)

        dt_res = []
        instance_id=0
        for item in pred_results:
            image = item["image"]
            joints_pos, vis_list = item["keypoints_pos"]
            joints_pos = np.array(joints_pos)
            vis_list = np.array(vis_list)
            keypoints = np.concatenate((joints_pos,vis_list.reshape((-1,1))), axis=1).flatten().tolist()
            
            # bbox = item['bbox']
            # if len(bbox) > 0:
            #     bbox = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
            # else:
            #     x = joints_pos[:,0]
            #     y = joints_pos[:,1]
            #     idx = np.nonzero(vis_list)[0]
            #     vis_x = x[idx]
            #     vis_y = y[idx]
            #     x0,x1,y0,y1 = np.min(vis_x), np.max(vis_x), np.min(vis_y), np.max(vis_y)
            #     bbox = [x0,y0,x1-x0,y1-y0]
            # area = bbox[2] * bbox[3]

            res = {
                'image_id': instance_id,
                'keypoints': keypoints,
                # 'num_keypoints': len(keypoints),
                'category_id': 1,
                # 'bbox': bbox,
                # 'area': area,
                'score': 1.0,
                # 'id': instance_id
            }
            dt_res.append(res)
            instance_id += 1

        coco_api = COCO(gt_file_path)
        coco_dt = coco_api.loadRes(dt_res)
        # coco_gt = coco_api.loadRes(gt_res["annotations"])
        coco_eval = COCOeval(coco_api, coco_dt,'keypoints', self.sigmas, use_area=self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]
        res ={}
        metric_results = {k: v for k, v in zip(stats_names, coco_eval.stats)}
        res["agg_metrics"] = metric_results["AP"].tolist()
        return res

    def _visualize_pose_results(self, pred_result, gt_result):
        save_dir = os.path.join(registry.get_path("output_dir"), "vis")
        os.makedirs(save_dir, exist_ok=True)

        for pred, gt in zip(pred_result, gt_result):
            pred_joints_pos, pred_vis_list = pred['keypoints_pos']
            gt_joints_pos, gt_vis_list = gt['keypoints_pos']

            # pred_joints_pos = pred_joints_pos.astype(np.int32).tolist()
            # pred_vis_list = pred_vis_list.astype(np.int32).tolist()
            # gt_joints_pos = gt_joints_pos.astype(np.int32).tolist()
            # gt_vis_list = gt_vis_list.astype(np.int32).tolist()
            
            assert pred['id'] == gt['id']
            assert pred["image"] == gt["image"]

            pred_vis_idx, = np.nonzero(pred_vis_list)
            gt_vis_idx, = np.nonzero(gt_vis_list)

            image_path = os.path.join(self.image_root, pred["image"])
            
            show_img = cv2.imread(image_path)
            
            # trans = get_warp_matrix(0, center * 2.0, self.image_size - 1.0, scale * 200.0)

            # show_img = cv2.warpAffine(
            #     show_img,
            #     trans, (int(self.image_size[0]), int(self.image_size[1])),
            #     flags=cv2.INTER_LINEAR)

            for idx in pred_vis_idx:
                show_img = cv2.circle(show_img, (pred_joints_pos[idx][0], pred_joints_pos[idx][1]), 5,(0,255,0),-1)

            for idx in gt_vis_idx:
                # print((gt_joints_pos[idx][0], gt_joints_pos[idx][1]))
                show_img = cv2.circle(show_img, (gt_joints_pos[idx][0], gt_joints_pos[idx][1]), 5,(0,0,255),-1)

            for line_pair in self.skeleton:
                if line_pair[0] in pred_vis_idx and line_pair[1] in pred_vis_idx:
                    show_img = cv2.line(show_img, (pred_joints_pos[line_pair[0]][0], pred_joints_pos[line_pair[0]][1]), (pred_joints_pos[line_pair[1]][0], pred_joints_pos[line_pair[1]][1]), color=(0,255,0),thickness=2)

                if line_pair[0] in gt_vis_idx and line_pair[1] in gt_vis_idx:
                    show_img = cv2.line(show_img, (gt_joints_pos[line_pair[0]][0], gt_joints_pos[line_pair[0]][1]), (gt_joints_pos[line_pair[1]][0], gt_joints_pos[line_pair[1]][1]), color=(0,0,255),thickness=2)
            
            save_path = str(pred['id']) + '_' + os.path.basename(image_path)
            save_file = os.path.join(save_dir, save_path)
            cv2.imwrite(save_file, show_img)
    
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file, result
