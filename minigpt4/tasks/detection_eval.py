"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import json
import re
import copy
import torch.distributed as dist
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import logging
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.common.utils import BBOX_TEMPLATE
from minigpt4.common.dist_utils import main_process
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.base_task import BaseTask
from minigpt4.metrics.detection_metric import PedDetEvaluator


@registry.register_task("detection_eval")
class DetectionEvalTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, eval_gt_ann=None, 
                 eval_task_type=None, quantize_bins=0, image_root=None, bbox_thr=0.05, use_decoder=False):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.eval_gt_ann = eval_gt_ann
        self.eval_task_type = eval_task_type # 'detection', 'pose_estimation', ...
        self.quantize_bins = quantize_bins
        self.image_root = image_root

        self.report_metric = report_metric

        self.bbox_thr = bbox_thr
        self.use_decoder = use_decoder
    
    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        dataset_cfg = cfg.datasets_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        # only support evaluating one dataset at one time
        eval_gt_ann = None
        eval_task_type = None
        image_root = None
        bbox_thr = None
        for k in dataset_cfg.keys():
            build_info = dataset_cfg[k].build_info 
            if 'val' in build_info.ann_path.keys():
                eval_gt_ann = build_info.ann_path.val
                image_root = build_info.image_path
                eval_task_type = dataset_cfg[k].task
                bbox_thr = dataset_cfg[k].get('bbox_thr', 0.05)
        quantize_bins = cfg.model_cfg.get("quantize_bins", 0)
        use_decoder = cfg.model_cfg.get("use_decoder", False)

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            eval_gt_ann=eval_gt_ann,
            eval_task_type=eval_task_type,
            quantize_bins=quantize_bins,
            image_root=image_root,
            bbox_thr = bbox_thr,
            use_decoder=use_decoder,
        )

    def convert_text_to_bbox(self, image_size, caption):
        w, h = image_size
        re_bbox = [r"(\d+)"] * 4
        pattern = re.compile(r"{}".format(BBOX_TEMPLATE.format(*re_bbox)))
        bbox_list = re.findall(pattern, caption)
        bbox_list = np.array([list(map(float, x)) for x in bbox_list])
        if len(bbox_list) > 0:
            bbox_list = np.round(bbox_list / self.quantize_bins * np.array([w, h, w, h]), 2)
        
        return bbox_list.tolist()

    def bbox_nms(self, bbox_list, overlap_threshold=0.7):
        # NMS
        sorted_boxes = sorted(bbox_list, key=lambda x: x[3], reverse=True)
        selected_boxes = []
        while len(sorted_boxes) > 0:
            current_box = sorted_boxes[0]
            selected_boxes.append(current_box)

            x1_current, y1_current, x2_current, y2_current = current_box
            area_current = (x2_current - x1_current + 1) * (y2_current - y1_current + 1)
            overlap_scores = []

            for box in sorted_boxes[1:]:
                x1, y1, x2, y2 = box
                x1_overlap = max(x1_current, x1)
                y1_overlap = max(y1_current, y1)
                x2_overlap = min(x2_current, x2)
                y2_overlap = min(y2_current, y2)

                width_overlap = max(0, x2_overlap - x1_overlap + 1)
                height_overlap = max(0, y2_overlap - y1_overlap + 1)

                area_overlap = width_overlap * height_overlap
                iou = area_overlap / (area_current + (x2 - x1 + 1) * (y2 - y1 + 1) - area_overlap)

                overlap_scores.append(iou)

            indices_to_keep = [i + 1 for i, overlap in enumerate(overlap_scores) if overlap < overlap_threshold]
            sorted_boxes = [box for i, box in enumerate(sorted_boxes) if i in indices_to_keep]

        return selected_boxes
    
    def valid_step(self, model, samples):
        results = []
        gt_results = []

        images = samples["image"]
        texts = samples["instruction_input"] if "instruction_input" in samples.keys() else samples["conv_q"]
        img_ids = samples["image_id"]
        # gts = samples["text_gt"]
        gts = samples["answer"] if "answer" in samples.keys() else samples["conv_a"]
        orig_img_sizes = samples["orig_image_size"]
        img_sizes = samples["image_size"]

        # run_cfg = slf.cfg.run_cfg
        # captions = model.generate(
        #     images,
        #     texts,
        #     max_new_tokens=self.max_len,
        #     num_beams=self.num_beams,
        # )
        captions, _ = model.generate(
            images,
            texts,
            max_new_tokens=self.max_len,
            num_beams=self.num_beams,
        )
        orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        img_w = img_sizes[0].cpu().numpy().tolist()
        img_h = img_sizes[1].cpu().numpy().tolist()
        for caption, img_id, ow, oh, w, h in zip(captions, img_ids, orig_img_w, orig_img_h, img_w, img_h):
            bbox_list = self.convert_text_to_bbox((ow, oh), caption)
            nms_bbox_list = self.bbox_nms(bbox_list, overlap_threshold=0.7)
            results.append({"raw_caption": caption, "bbox": nms_bbox_list, "image": img_id, "height": h, "width": w})
        for gt_caption, img_id, ow, oh, w, h in zip(gts, img_ids, orig_img_w, orig_img_h, img_w, img_h):
            gt_caption = self.convert_text_to_bbox((ow, oh), gt_caption)
            gt_results.append({"bbox": gt_caption, "image": img_id, "height": h, "width": w})
        
        # visualize detection results
        self._visualize_detection_results(results, gt_results)
        
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
            remove_duplicate="image",
        )
        gt_result_file, gt_result = self.save_result(
            result=gt_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_gt_epoch{}".format(split_name, epoch),
            remove_duplicate="image",
        )


        if self.report_metric:
            # metrics = self._report_detection_metrics(epoch, eval_result, gt_result, split_name)
            metrics = self._report_CrowdHuman_detection_metrics(epoch, eval_result, gt_result, split_name)
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    # CrowdHuman Metric
    @main_process
    def _report_CrowdHuman_detection_metrics(self, epoch, pred_results, gts, split_name):
        log_stats = {}
        # for llm output
        for dt in pred_results:
            bbox_list = dt['bbox']
            class_bbox_list = [x + [1.0] for x in bbox_list]
            dt['pred'] = class_bbox_list
        det_metric = PedDetEvaluator()
        llm_metric_results = det_metric.evaluate(gts, pred_results)
        log_stats[split_name] = {k: v for k, v in llm_metric_results.items()}

        with open(
            os.path.join(registry.get_path("output_dir"), f"evaluate{epoch}.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {k: v for k, v in llm_metric_results.items()}
        res["agg_metrics"] = llm_metric_results["mAP"]

        return res
    
    # # COCO Metric
    # @main_process
    # def _report_detection_metrics(self, epoch, pred_result, gt, split_name):
    #     bbox_pattern = re.compile(r'\[(\d+\.\d+|\d+),\s(\d+\.\d+|\d+),\s(\d+\.\d+|\d+),\s(\d+\.\d+|\d+)\]')

    #     gt_res = {"images": [], "annotations": [], "categories": [{"supercategory": "person","id": 1,"name": "person"}]}
    #     image_id = 0
    #     instance_id = 0
    #     img2id = {}
    #     for item in gt:
    #         image = item["image"]
    #         img2id[image] = image_id
    #         gt_res["images"].append({"file_name": image, "id": image_id})
    #         caption = item["caption"]
    #         bbox_list = re.findall(bbox_pattern, caption)
    #         bbox_list = [list(map(float, x)) for x in bbox_list]
    #         for bbox in bbox_list:
    #             res = {
    #                 'image_id': image_id,
    #                 'bbox': bbox,
    #                 'category_id': 1,
    #                 'iscrowd': 0,
    #                 'area': bbox[2] * bbox[3],
    #                 'id': instance_id
    #             }
    #             gt_res["annotations"].append(res)
    #             instance_id += 1
    #         image_id += 1
    #     gt_file_name = "{}_coco_gt.json".format(split_name)
    #     gt_file_path = os.path.join(registry.get_path("result_dir"), gt_file_name)
    #     with open(gt_file_path, 'w') as f:
    #         json.dump(gt_res, f)

    #     def unihcp_nms(bbox_list, overlap_threshold=0.7):
    #         # NMS
    #         # convert xywh to x1y1x2y2
    #         for i in range(len(bbox_list)):
    #             bbox_list[i][2] += bbox_list[i][0]
    #             bbox_list[i][3] += bbox_list[i][1]
    #         sorted_boxes = sorted(bbox_list, key=lambda x: x[3], reverse=True)
    #         selected_boxes = []
    #         while len(sorted_boxes) > 0:
    #             current_box = sorted_boxes[0]
    #             selected_boxes.append(current_box)
    #             x1_current, y1_current, x2_current, y2_current, _ = current_box
    #             area_current = (x2_current - x1_current + 1) * (y2_current - y1_current + 1)
    #             overlap_scores = []

    #             for box in sorted_boxes[1:]:
    #                 x1, y1, x2, y2, _ = box
    #                 x1_overlap = max(x1_current, x1)
    #                 y1_overlap = max(y1_current, y1)
    #                 x2_overlap = min(x2_current, x2)
    #                 y2_overlap = min(y2_current, y2)

    #                 width_overlap = max(0, x2_overlap - x1_overlap + 1)
    #                 height_overlap = max(0, y2_overlap - y1_overlap + 1)

    #                 area_overlap = width_overlap * height_overlap
    #                 iou = area_overlap / (area_current + (x2 - x1 + 1) * (y2 - y1 + 1) - area_overlap)

    #                 overlap_scores.append(iou)

    #             indices_to_keep = [i + 1 for i, overlap in enumerate(overlap_scores) if overlap < overlap_threshold]
    #             sorted_boxes = [box for i, box in enumerate(sorted_boxes) if i in indices_to_keep]
            
    #         # convert x1y1x2y2 to xywh
    #         for i in range(len(selected_boxes)):
    #             selected_boxes[i][2] -= selected_boxes[i][0]
    #             selected_boxes[i][3] -= selected_boxes[i][1]

    #         return selected_boxes


    #     coco_api = COCO(gt_file_path)
    #     dt_res = []
    #     unihcp_dt_res = []
    #     for item in pred_result:
    #         image = item["image"]
    #         caption = item["caption"]
    #         bbox_list = re.findall(bbox_pattern, caption)
    #         bbox_list = [list(map(float, x)) for x in bbox_list]
    #         for bbox in bbox_list:
    #             res = {
    #                 'image_id': img2id[image],
    #                 'bbox': bbox,
    #                 'score': 1.0,
    #                 'category_id': 1,
    #                 'id': instance_id
    #             }
    #             dt_res.append(res)

    #         unihcp_class_bbox = item["output_class_bbox"]
    #         # print(unihcp_class_bbox)
    #         # unihcp_class_bbox = unihcp_nms(unihcp_class_bbox)
    #         for i in range(len(unihcp_class_bbox)):
    #             unihcp_res = {
    #             'image_id': img2id[image],
    #             'bbox': unihcp_class_bbox[i][:4],
    #             'score': unihcp_class_bbox[i][4],
    #             'category_id': 1,
    #             'id': instance_id
    #             }
    #             unihcp_dt_res.append(unihcp_res)
            
    #     # coco_dt = coco_api.loadRes(dt_res)

    #     # coco_eval = COCOeval(coco_api, coco_dt, iouType='bbox')
    #     # coco_eval.evaluate()
    #     # coco_eval.accumulate()
    #     # coco_eval.summarize()

    #     # coco_metric_names = {
    #     #         'mAP': 0,
    #     #         'mAP_50': 1,
    #     #         'mAP_75': 2,
    #     #         'mAP_s': 3,
    #     #         'mAP_m': 4,
    #     #         'mAP_l': 5,
    #     #         'AR@100': 6,
    #     #         'AR@300': 7,
    #     #         'AR@1000': 8,
    #     #         'AR_s@1000': 9,
    #     #         'AR_m@1000': 10,
    #     #         'AR_l@1000': 11
    #     #     }
    #     # metric_results = {}
    #     # for k in coco_metric_names:
    #     #     metric_results[k] = coco_eval.stats[coco_metric_names[k]].round(5)
    #     # log_stats = {split_name: metric_results}

    #     # with open(
    #     #     os.path.join(registry.get_path("output_dir"), f"evaluate{epoch}.txt"), "a"
    #     # ) as f:
    #     #     f.write(json.dumps(log_stats) + "\n")

    #     # res = metric_results
    #     # res["agg_metrics"] = metric_results["mAP"]

    #     unihcp_coco_api = COCO(gt_file_path)
    #     unihcp_coco_dt = unihcp_coco_api.loadRes(unihcp_dt_res)

    #     unihcp_coco_eval = COCOeval(unihcp_coco_api, unihcp_coco_dt, iouType='bbox')
    #     unihcp_coco_eval.params.maxDets = [10,100,500]
    #     unihcp_coco_eval.evaluate()
    #     unihcp_coco_eval.accumulate()
    #     unihcp_coco_eval.summarize()

    #     unihcp_coco_metric_names = {
    #             'mAP': 0,
    #             'mAP_50': 1,
    #             'mAP_75': 2,
    #             'mAP_s': 3,
    #             'mAP_m': 4,
    #             'mAP_l': 5,
    #             'AR@100': 6,
    #             'AR@300': 7,
    #             'AR@1000': 8,
    #             'AR_s@1000': 9,
    #             'AR_m@1000': 10,
    #             'AR_l@1000': 11
    #         }
    #     unihcp_metric_results = {}
    #     for k in unihcp_coco_metric_names:
    #         unihcp_metric_results[k] = unihcp_coco_eval.stats[unihcp_coco_metric_names[k]].round(5)
    #     unihcp_log_stats = {split_name: unihcp_metric_results}

    #     with open(
    #         os.path.join(registry.get_path("output_dir"), f"evaluate{epoch}.txt"), "a"
    #     ) as f:
    #         f.write(json.dumps(unihcp_log_stats) + "\n")

    #     unihcp_res = unihcp_metric_results
    #     unihcp_res["agg_metrics"] = unihcp_metric_results["mAP"]

    #     return unihcp_res

    def _visualize_detection_results(self, pred_result, gt_result):
        save_dir = os.path.join(registry.get_path("output_dir"), "vis")
        os.makedirs(save_dir, exist_ok=True)

        for pred, gt in zip(pred_result, gt_result):
            pred_bbox_list = pred['bbox']
            gt_bbox_list = gt['bbox']

            assert pred["image"] == gt["image"]
            image_path = os.path.join(self.image_root, pred["image"])
            image = cv2.imread(image_path)

            for bbox in gt_bbox_list:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            for bbox in pred_bbox_list:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            save_file = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_file, image)
    
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

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
