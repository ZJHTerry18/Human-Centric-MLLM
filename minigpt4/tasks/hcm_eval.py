"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import re
import numpy as np
import cv2
import torch.distributed as dist
import matplotlib.pyplot as plt
import logging
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.utils import BBOX_TEMPLATE
from minigpt4.common.logger import MetricLogger
from minigpt4.common.registry import registry
from minigpt4.common.dist_utils import main_process
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.common.eval_utils import computeIoU
from minigpt4.tasks.base_task import BaseTask


@registry.register_task("hcm_cap_val")
class HCMCaptionEvalTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, eval_gt_ann=None, 
                 eval_task_type=None, quantize_bins=0, image_root=None):
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
        for k in dataset_cfg.keys():
            build_info = dataset_cfg[k].build_info 
            if 'val' in build_info.ann_path.keys():
                eval_gt_ann = build_info.ann_path.val
                image_root = build_info.image_path
                eval_task_type = dataset_cfg[k].task
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
        )
    
    def valid_step(self, model, samples):
        results = []
        gt_results = []

        images = samples["image"]
        texts = samples["instruction_input"] if "instruction_input" in samples.keys() else samples["conv_q"]
        img_ids = samples["image_id"]
        instance_ids = samples["instance_id"]
        bboxes = samples["bbox"]
        # gts = samples["text_gt"]
        gts = samples["answer"] if "answer" in samples.keys() else samples["conv_a"]
        orig_img_sizes = samples["orig_image_size"]
        img_sizes = samples["image_size"]

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            images,
            texts,
            max_new_tokens=self.max_len,
            num_beams=self.num_beams,
        )

        orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        img_w = img_sizes[0].cpu().numpy().tolist()
        img_h = img_sizes[1].cpu().numpy().tolist()
        for instruction, caption, img_id, instance_id, bbox, ow, oh, w, h in zip(texts, captions, img_ids, instance_ids, bboxes, orig_img_w, orig_img_h, img_w, img_h):
            bbox = bbox.cpu().numpy()* [ow, oh, ow, oh]
            bbox[2:4] += bbox[0:2]
            bbox = bbox.astype(int).tolist()
            instance_id = instance_id.item()
            results.append({"image": img_id, "instance_id": instance_id, "bbox": bbox, "height": oh, "width": ow, "question": instruction, "answer": caption})
        for instruction, gt_caption, img_id, instance_id, bbox, ow, oh, w, h in zip(texts, gts, img_ids, instance_ids, bboxes, orig_img_w, orig_img_h, img_w, img_h):
            bbox = bbox.cpu().numpy().astype(int) * [ow, oh, ow, oh]
            bbox[2:4] += bbox[0:2]
            bbox = bbox.tolist()
            instance_id = instance_id.item()
            gt_results.append({"image": img_id, "instance_id": instance_id, "bbox": bbox, "height": oh, "width": ow, "question": instruction, "answer": gt_caption})
        self._visualize_caption_results(results, gt_results)

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
            remove_duplicate="instance_id",
        )
        gt_result_file, gt_result = self.save_result(
            result=gt_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_gt_epoch{}".format(split_name, epoch),
            remove_duplicate="instance_id",
        )

        metrics = {"agg_metrics": 0.0}

        return metrics
    
    def _visualize_caption_results(self, pred_result, gt_result):
        save_dir = os.path.join(registry.get_path("output_dir"), "vis")
        os.makedirs(save_dir, exist_ok=True)

        for pred, gt in zip(pred_result, gt_result):
            bbox = pred['bbox']

            image_path = os.path.join(self.image_root, pred["image"])
            image = cv2.imread(image_path)
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            
            save_file = os.path.join(save_dir, str(pred["instance_id"]) + '_' + os.path.basename(image_path))
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

@registry.register_task("hcm_grounding_val")
class HCMGroundingEvalTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, eval_gt_ann=None, 
                 eval_task_type=None, quantize_bins=0, image_root=None):
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
        for k in dataset_cfg.keys():
            build_info = dataset_cfg[k].build_info 
            if 'val' in build_info.ann_path.keys():
                eval_gt_ann = build_info.ann_path.val
                image_root = build_info.image_path
                eval_task_type = dataset_cfg[k].task
        quantize_bins = cfg.model_cfg.get("quantize_bins", 0)

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
        )

    def extract_object_bbox(self, image_size, caption):
        w, h = image_size
        bbox = ["(\d+)"] * 4
        bbox_pattern = BBOX_TEMPLATE.format(*bbox)
        object_pattern = re.compile("<p>(.*?)</p>" + bbox_pattern)
        object_list = re.findall(object_pattern, caption)
        name_list = [n[0] for n in object_list]
        bbox_list = np.array([list(map(float, x[1:])) for x in object_list])
        if len(bbox_list) > 0:
            bbox_list = np.round(bbox_list / self.quantize_bins * np.array([w, h, w, h]), 2)
        
        return name_list, bbox_list.tolist()
    
    def valid_step(self, model, samples):
        results = []
        gt_results = []

        images = samples["image"]
        texts = samples["instruction_input"] if "instruction_input" in samples.keys() else samples["conv_q"]
        img_ids = samples["image_id"]
        instance_ids = samples["instance_id"]
        # gts = samples["text_gt"]
        gts = samples["answer"] if "answer" in samples.keys() else samples["conv_a"]
        orig_img_sizes = samples["orig_image_size"]
        img_sizes = samples["image_size"]

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            images,
            texts,
            max_new_tokens=self.max_len,
            num_beams=self.num_beams,
        )

        orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        img_w = img_sizes[0].cpu().numpy().tolist()
        img_h = img_sizes[1].cpu().numpy().tolist()
        for instruction, caption, img_id, instance_id, ow, oh, w, h in zip(texts, captions, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            names, bboxes = self.extract_object_bbox((ow, oh), caption)
            instance_id = instance_id.item()
            results.append({"image": img_id, "instance_id": instance_id, "height": h, "width": w, "question": instruction, "answer": caption, "names": names, "bboxes": bboxes})
        for instruction, gt_caption, img_id, instance_id, ow, oh, w, h in zip(texts, gts, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            names, bboxes = self.extract_object_bbox((ow, oh), gt_caption)
            instance_id = instance_id.item()
            gt_results.append({"image": img_id, "instance_id": instance_id, "height": h, "width": w, "question": instruction, "answer": gt_caption, "names": names, "bboxes": bboxes})
        
        self._visualize_grounding_results(results, gt_results)

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
            remove_duplicate="instance_id",
        )
        gt_result_file, gt_result = self.save_result(
            result=gt_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_gt_epoch{}".format(split_name, epoch),
            remove_duplicate="instance_id",
        )

        metrics = {"agg_metrics": 0.0}

        return metrics

    def _visualize_grounding_results(self, pred_result, gt_result):
        save_dir = os.path.join(registry.get_path("output_dir"), "vis")
        os.makedirs(save_dir, exist_ok=True)

        for pred, gt in zip(pred_result, gt_result):
            pred_bbox_list = pred['bboxes']
            pred_name_list = pred['names']

            image_path = os.path.join(self.image_root, pred["image"])
            image = cv2.imread(image_path)
            
            colors_list = plt.cm.gist_rainbow(np.linspace(0, 1, len(pred_bbox_list)))
            for i, (bbox, name) in enumerate(zip(pred_bbox_list, pred_name_list)):
                color = [int(x * 255) for x in colors_list[i]]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
            
            save_file = os.path.join(save_dir, str(pred["instance_id"]) + '_' + os.path.basename(image_path))
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

@registry.register_task("hcm_rec_val")
class HCMRECEvalTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, eval_gt_ann=None, 
                 eval_task_type=None, quantize_bins=0, image_root=None):
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
        for k in dataset_cfg.keys():
            build_info = dataset_cfg[k].build_info 
            if 'val' in build_info.ann_path.keys():
                eval_gt_ann = build_info.ann_path.val
                image_root = build_info.image_path
                eval_task_type = dataset_cfg[k].task
        quantize_bins = cfg.model_cfg.get("quantize_bins", 0)

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
        )

    def extract_bbox(self, image_size, caption):
        w, h = image_size
        bbox = ["(\d+)"] * 4
        bbox_pattern = BBOX_TEMPLATE.format(*bbox)
        bbox_list = re.findall(bbox_pattern, caption)
        bbox_list = np.array([list(map(float, x)) for x in bbox_list])
        if len(bbox_list) > 0:
            bbox_list = np.round(bbox_list / self.quantize_bins * np.array([w, h, w, h]), 2)
        
        return bbox_list.tolist()
    
    def valid_step(self, model, samples):
        results = []
        gt_results = []

        images = samples["image"]
        texts = samples["instruction_input"] if "instruction_input" in samples.keys() else samples["conv_q"]
        img_ids = samples["image_id"]
        instance_ids = samples["instance_id"]
        # gts = samples["text_gt"]
        gts = samples["answer"] if "answer" in samples.keys() else samples["conv_a"]
        orig_img_sizes = samples["orig_image_size"]
        img_sizes = samples["image_size"]

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            images,
            texts,
            max_new_tokens=self.max_len,
            num_beams=self.num_beams,
        )

        orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        img_w = img_sizes[0].cpu().numpy().tolist()
        img_h = img_sizes[1].cpu().numpy().tolist()
        for instruction, caption, img_id, instance_id, ow, oh, w, h in zip(texts, captions, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            bboxes = self.extract_bbox((ow, oh), caption)
            instance_id = instance_id.item()
            results.append({"image": img_id, "instance_id": instance_id, "height": h, "width": w, "question": instruction, "answer": caption, "bboxes": bboxes})
        for instruction, gt_caption, img_id, instance_id, ow, oh, w, h in zip(texts, gts, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            bboxes = self.extract_bbox((ow, oh), gt_caption)
            instance_id = instance_id.item()
            gt_results.append({"image": img_id, "instance_id": instance_id, "height": h, "width": w, "question": instruction, "answer": gt_caption, "bboxes": bboxes})
        
        self._visualize_rec_results(results, gt_results)

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
            remove_duplicate="instance_id",
        )
        gt_result_file, gt_result = self.save_result(
            result=gt_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_gt_epoch{}".format(split_name, epoch),
            remove_duplicate="instance_id",
        )

        if self.report_metric:
            metrics = self._report_rec_metrics(epoch, eval_result, gt_result, split_name)
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics
    
    @main_process
    def _report_rec_metrics(self, epoch, pred_result, gt_result, split_name, iou_threshold=0.5):
        count = 0
        total = len(pred_result)
        for pred, gt in zip(pred_result, gt_result):
            if len(pred['bboxes']) == 0:
                continue
            pred_bbox = pred['bboxes'][0]
            gt_bbox = gt['bboxes'][0]
            iou_score = computeIoU(pred_bbox, gt_bbox)
            if iou_score > iou_threshold:
                count += 1
        accuracy = count / total * 100
        print('Accuracy: {:.2f}%'.format(accuracy))
        metric_result = {"agg_metrics": accuracy}

        return metric_result


    def _visualize_rec_results(self, pred_result, gt_result):
        save_dir = os.path.join(registry.get_path("output_dir"), "vis")
        os.makedirs(save_dir, exist_ok=True)

        for pred, gt in zip(pred_result, gt_result):
            pred_bbox_list = pred['bboxes']
            gt_bbox_list = gt['bboxes']

            image_path = os.path.join(self.image_root, pred["image"])
            image = cv2.imread(image_path)
            
            for bbox in pred_bbox_list:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            for bbox in gt_bbox_list:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            
            save_file = os.path.join(save_dir, str(pred["instance_id"]) + '_' + os.path.basename(image_path))
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