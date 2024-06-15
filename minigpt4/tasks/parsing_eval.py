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
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import logging
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.common.utils import BBOX_TEMPLATE, CLASSES_PLACEHOLDER
from minigpt4.common.dist_utils import main_process
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.base_task import BaseTask
from minigpt4.metrics.parsing_metric import HumParEvaluator


@registry.register_task("parsing_eval")
class ParsingEvalTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, eval_gt_ann=None, 
                 eval_task_type=None, quantize_bins=0, image_root=None, use_decoder=False):
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
            use_decoder=use_decoder,
        )

    def convert_text_to_polygon(self, image_size, caption):
        w, h = image_size
        seg_caption = caption.split('\n')
        class_seg_pattern = r"<p>(.*?)<\/p>{(.*?)}"
        seg_pattern = r"<(\d+)>"
        matches_class_seg = []
        for c in seg_caption:
            matches_class_seg.extend(re.findall(class_seg_pattern, c))

        res_dict = {}
        for cls, seg in matches_class_seg:
            if cls in res_dict.keys():
                warnings.warn(f'Duplicate parsings of {cls}!')
            cls = cls.strip()
            poly_vertices = []
            poly_strs = seg.split('<delim>')
            for p in poly_strs:
                poly_list = re.findall(seg_pattern, p)
                poly_list = poly_list[:-1] if len(poly_list) % 2 == 1 else poly_list[:] # TODO: how to deal with odd number of outputs
                polygon = np.array(poly_list).reshape(-1, 2).astype(float)
                polygon = polygon / 100 * np.array([w, h])
                polygon_list = np.round(polygon, 2).flatten().tolist()
                poly_vertices.append(polygon_list)
            res_dict[cls] = poly_vertices

        return res_dict

    
    def valid_step(self, model, samples):
        results = []
        gt_results = []

        # # test each body part separately (for batch size=1)
        # images = samples["image"]
        # texts = samples["instruction_input"] if "instruction_input" in samples.keys() else samples["conv_q"]
        # img_ids = samples["image_id"]
        # instance_ids = samples["instance_id"].cpu().numpy().tolist()
        # gts = samples["answer"] if "answer" in samples.keys() else samples["conv_a"]
        # # gts = samples['target']
        # orig_img_sizes = samples["orig_image_size"]
        # img_sizes = samples["image_size"]

        # classes = [c[0] for c in samples['class']]
        # seg_dict = {}
        # gt_dict = {}
        # result = {}
        # gt_result = {}
        # for i, cls in enumerate(classes):
        #     cls = cls.lower()
        #     texts_2 = [t.replace(CLASSES_PLACEHOLDER, cls) for t in texts]
        #     # run_cfg = slf.cfg.run_cfg
        #     captions = model.generate(
        #         images,
        #         texts_2,
        #         max_new_tokens=self.max_len,
        #         num_beams=self.num_beams,
        #     )
        #     # print('input: ', texts_2)
        #     # print('output: ', captions)
        #     orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        #     orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        #     img_w = img_sizes[0].cpu().numpy().tolist()
        #     img_h = img_sizes[1].cpu().numpy().tolist()
        #     for caption, img_id, instance_id, ow, oh, w, h in zip(captions, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
        #         # seg_dict = self.convert_text_to_polygon((ow, oh), caption)
        #         seg_dict.update(self.convert_text_to_polygon((ow, oh), caption))
        #         # results.append({"raw_caption": caption, "segmentation": seg_dict, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
        #         result.update({"raw_caption": caption, "segmentation": seg_dict, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
        #     for gt, img_id, instance_id, ow, oh, w, h in zip(gts, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
        #         # gt_dict = self.convert_text_to_polygon((ow, oh), gt)
        #         gt_dict.update(self.convert_text_to_polygon((ow, oh), gt))
        #         # gt_results.append({"segmentation": gt_dict, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
        #         gt_result.update({"segmentation": gt_dict, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
        # results.append(result)
        # gt_results.append(gt_result)

        # test all body parts at one time
        images = samples["image"]
        texts = samples["instruction_input"] if "instruction_input" in samples.keys() else samples["conv_q"]
        img_ids = samples["image_id"]
        instance_ids = samples["instance_id"].cpu().numpy().tolist()
        gts = samples["answer"] if "answer" in samples.keys() else samples["conv_a"]
        # gts = samples['target']
        orig_img_sizes = samples["orig_image_size"]
        img_sizes = samples["image_size"]

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            images,
            texts,
            max_new_tokens=self.max_len,
            num_beams=self.num_beams,
        )
        # print('input: ', texts)
        # print('output: ', captions)
        orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        img_w = img_sizes[0].cpu().numpy().tolist()
        img_h = img_sizes[1].cpu().numpy().tolist()
        for caption, img_id, instance_id, ow, oh, w, h in zip(captions, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            seg_dict = self.convert_text_to_polygon((ow, oh), caption)
            results.append({"raw_caption": caption, "segmentation": seg_dict, "image": img_id, "id": instance_id, "height": h, "width": w})
        for gt, img_id, instance_id, ow, oh, w, h in zip(gts, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            gt_dict = self.convert_text_to_polygon((ow, oh), gt)
            gt_results.append({"segmentation": gt_dict, "image": img_id, "id": instance_id, "height": h, "width": w})
        
        # visualize parsing results
        self._visualize_parsing_results(results, gt_results)
        
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
            metrics = self._report_parsing_metrics(epoch, eval_result, gt_result, split_name)
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics
    
    @main_process
    def _report_parsing_metrics(self, epoch, pred_results, gts, split_name):
        log_stats = {}
        with open(self.eval_gt_ann, 'r') as f:
            lines = f.readlines()
            class_list = list(json.loads(lines[0])['segmentation'].keys())
        class_list = [c.lower() for c in class_list]
        num_classes = len(class_list) + 1 # add background class
        class_indexes = list(range(num_classes))
        evaluator = HumParEvaluator(class_names=class_indexes)
        evaluator.reset()

        # convert polygon output into mask
        gt_masks = []
        pt_masks = []
        for gt, pt in zip(gts, pred_results):
            assert gt['id'] == pt['id']
            w = gt['width']
            h = gt['height']
            mask = np.zeros((h, w), dtype=np.uint8)

            for cls, seg in gt['segmentation'].items():
                if len(seg) > 0:
                    cls_label = class_list.index(cls) + 1
                    for s in seg:
                        polygon = np.array(s).reshape(-1, 2)
                        cv2.fillConvexPoly(mask, np.int32([polygon]), cls_label)
            gt_masks.append(mask)

            w = pt['width']
            h = pt['height']
            mask = np.zeros((h, w), dtype=np.uint8)

            for cls, seg in pt['segmentation'].items():
                if len(seg) > 0 and cls in class_list:
                    cls_label = class_list.index(cls) + 1
                    for s in seg:
                        polygon = np.array(s).reshape(-1, 2)
                        cv2.fillConvexPoly(mask, np.int32([polygon]), cls_label)
            pt_masks.append(mask)
        
        evaluator.process(gt_masks, pt_masks)
        results = evaluator.evaluate()
        log_stats[split_name] = results

        with open(
            os.path.join(registry.get_path("output_dir"), f"evaluate{epoch}.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = {k: v for k, v in results.items()}
        res["agg_metrics"] = results["mIoU"]

        return res

    def _visualize_parsing_results(self, pred_result, gt_result):
        save_dir = os.path.join(registry.get_path("output_dir"), "vis")
        os.makedirs(save_dir, exist_ok=True)
        colormap = np.array([[0.        , 0.        , 0.        ],
                            [0.5       , 0.        , 0.        ],
                            [0.99609375, 0.        , 0.        ],
                            [0.        , 0.33203125, 0.        ],
                            [0.6640625 , 0.        , 0.19921875],
                            [0.99609375, 0.33203125, 0.        ],
                            [0.        , 0.        , 0.33203125],
                            [0.        , 0.46484375, 0.86328125],
                            [0.33203125, 0.33203125, 0.        ],
                            [0.        , 0.33203125, 0.33203125],
                            [0.33203125, 0.19921875, 0.        ],
                            [0.203125  , 0.3359375 , 0.5       ],
                            [0.        , 0.5       , 0.        ],
                            [0.        , 0.        , 0.99609375],
                            [0.19921875, 0.6640625 , 0.86328125],
                            [0.        , 0.99609375, 0.99609375],
                            [0.33203125, 0.99609375, 0.6640625 ],
                            [0.6640625 , 0.99609375, 0.33203125],
                            [0.99609375, 0.99609375, 0.        ],
                            [0.99609375, 0.6640625 , 0.        ]])

        for pred in pred_result:
            # draw predicted mask
            imagename = pred['image']
            instance_id = pred['id']
            pred_img = cv2.imread(imagename)
            pred_segs = pred['segmentation']
            for i, (cls, seg) in enumerate(pred_segs.items()):
                color = colormap[i]
                color = [(x * 255) for x in color]
                for polygon in seg:
                    p = np.array(polygon).reshape(-1, 2)
                    center = np.mean(p, axis=0).astype(np.int32)
                    cv2.polylines(pred_img, np.int32([p]), True, color, 2)
                    cv2.putText(pred_img, cls, (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
            save_file = os.path.join(save_dir, os.path.basename(imagename)[:-4] + f'_{instance_id}' + '.jpg')
            cv2.imwrite(save_file, pred_img)
    
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
