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


@registry.register_task("parsing_bbox_eval")
class ParsingBoxEvalTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True, eval_gt_ann=None, 
                 eval_task_type=None, quantize_bins=0, image_root=None, use_decoder=False, all_parts_list=None):
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
        self.all_parts_list = all_parts_list
    
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
        all_parts_list = None
        for k in dataset_cfg.keys():
            build_info = dataset_cfg[k].build_info 
            if 'val' in build_info.ann_path.keys():
                eval_gt_ann = build_info.ann_path.val
                image_root = build_info.image_path
                eval_task_type = dataset_cfg[k].task
                all_parts_list = dataset_cfg[k].vis_processor.eval.data_cfg.all_parts_list
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
            all_parts_list=all_parts_list
        )

    # def convert_text_to_polygon(self, image_size, caption):
    #     w, h = image_size
    #     seg_caption = caption.split('\n')
    #     class_seg_pattern = r"<p>(.*?)<\/p>{(.*?)}"
    #     seg_pattern = r"<(\d+)>"
    #     matches_class_seg = []
    #     for c in seg_caption:
    #         matches_class_seg.extend(re.findall(class_seg_pattern, c))

    #     res_list = [[]] * len(self.all_parts_list)
    #     for cls, seg in matches_class_seg:
    #         if cls in self.all_parts_list:
    #             cls = cls.strip()
    #             index = self.all_parts_list.index(cls)
    #             poly_vertices = []
    #             poly_strs = seg.split('<delim>')
    #             for p in poly_strs:
    #                 poly_list = re.findall(seg_pattern, p)
    #                 poly_list = poly_list[:-1] if len(poly_list) % 2 == 1 else poly_list[:] # TODO: how to deal with odd number of outputs
    #                 polygon = np.array(poly_list).reshape(-1, 2).astype(float)
    #                 polygon = polygon / self.quantize_bins * np.array([w, h])
    #                 polygon_list = np.round(polygon, 2).flatten().tolist()
    #                 poly_vertices.append(polygon_list)
    #             res_list[index] = poly_vertices

    #     return res_list
    def convert_text_to_polygon(self, image_size, caption):
        w, h = image_size
        seg_caption = caption.split('\n')
        class_seg_pattern = r"<p>(.*?)<\/p>{<(\d+)><(\d+)><(\d+)><(\d+)>}"
        matches_class_seg = []
        for c in seg_caption:
            matches_class_seg.extend(re.findall(class_seg_pattern, c))

        res_list = [[]] * len(self.all_parts_list)
        for cls, x1, y1, x2, y2 in matches_class_seg:
            if cls in self.all_parts_list:
                cls = cls.strip()
                index = self.all_parts_list.index(cls)
                poly_vertices = []
                
                polygon = np.array([x1, y1, x2, y2]).reshape(-1, 2).astype(float)
                polygon = polygon / self.quantize_bins * np.array([w, h])
                polygon_list = np.round(polygon, 2).flatten().tolist()
                poly_vertices.append(polygon_list)
                res_list[index] = poly_vertices

        return res_list

    
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
        # seg_list = [[]] * len(self.all_parts_list)
        # gt_list = [[]] * len(self.all_parts_list)
        # result = {}
        # gt_result = {}
        # for i, cls in enumerate(classes):
        #     cls = cls.lower()
        #     texts_2 = [t.replace(CLASSES_PLACEHOLDER, cls) for t in texts]
        #     # run_cfg = slf.cfg.run_cfg
        #     # captions = model.generate(
        #     #     images,
        #     #     texts_2,
        #     #     max_new_tokens=self.max_len,
        #     #     num_beams=self.num_beams,
        #     # )
        #     captions, _ = model.generate(
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
        #         new_seg = self.convert_text_to_polygon((ow, oh), caption)
        #         seg_list = [seg_list[i] + new_seg[i] for i in range(len(seg_list))]
        #         # results.append({"raw_caption": caption, "segmentation": seg_dict, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
        #         result.update({"raw_caption": caption, "segmentation": seg_list, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
        #     for gt, img_id, instance_id, ow, oh, w, h in zip(gts, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
        #         new_seg = self.convert_text_to_polygon((ow, oh), gt)
        #         gt_list = [gt_list[i] + new_seg[i] for i in range(len(gt_list))]
        #         # gt_results.append({"segmentation": gt_dict, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
        #         gt_result.update({"segmentation": gt_list, "image": img_id, "id": instance_id * 20, "height": h, "width": w})
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
        # print('input: ', texts)
        # print('gt: ', gts)
        # print('output: ', captions)
        orig_img_w = orig_img_sizes[0].cpu().numpy().tolist()
        orig_img_h = orig_img_sizes[1].cpu().numpy().tolist()
        img_w = img_sizes[0].cpu().numpy().tolist()
        img_h = img_sizes[1].cpu().numpy().tolist()
        for caption, img_id, instance_id, ow, oh, w, h in zip(captions, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            seg_list = self.convert_text_to_polygon((ow, oh), caption)
            results.append({"raw_caption": caption, "segmentation": seg_list, "image": img_id, "id": instance_id, "height": h, "width": w})
        for gt, img_id, instance_id, ow, oh, w, h in zip(gts, img_ids, instance_ids, orig_img_w, orig_img_h, img_w, img_h):
            gt_list = self.convert_text_to_polygon((ow, oh), gt)
            gt_results.append({"segmentation": gt_list, "image": img_id, "id": instance_id, "height": h, "width": w})
        
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
            metrics = self._report_detection_metrics(epoch, eval_result, gt_result, split_name)
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics
    
    # COCO Metric
    @main_process
    def _report_detection_metrics(self, epoch, pred_result, gt_result, split_name):
        gt_res = {"images": [], "annotations": [], "categories": []}
        for i, cls in enumerate(self.all_parts_list):
            category_info = {"supercategory": cls, "id": i + 1, "name": cls}
            gt_res['categories'].append(category_info)

        image_id = 0
        instance_id = 0
        img2id = {}
        for item in gt_result:
            image = item["image"]
            img2id[image] = image_id
            gt_res["images"].append({"file_name": image, "id": image_id})
            bbox_list = item['segmentation']
            for i, bbox in enumerate(bbox_list):
                if len(bbox) > 0:
                    cat_id = i + 1
                    # if cat_id in [15, 17, 19]:
                    #     cat_id -= 1
                    bbox = bbox[0]
                    bbox[2] = bbox[2] - bbox[0]
                    bbox[3] = bbox[3] - bbox[1]
                    res = {
                        'image_id': image_id,
                        'bbox': bbox,
                        'category_id': cat_id,
                        'iscrowd': 0,
                        'area': bbox[2] * bbox[3],
                        'id': instance_id
                    }
                    gt_res["annotations"].append(res)
                    instance_id += 1
            image_id += 1
        gt_file_name = "{}_coco_gt.json".format(split_name)
        gt_file_path = os.path.join(registry.get_path("result_dir"), gt_file_name)
        with open(gt_file_path, 'w') as f:
            json.dump(gt_res, f)

        coco_api = COCO(gt_file_path)
        dt_res = []
        instance_id = 0
        for item in pred_result:
            image = item["image"]
            bbox_list = item["segmentation"]
            for i, bbox in enumerate(bbox_list):
                if len(bbox) > 0:
                    cat_id = i + 1
                    # if cat_id in [15, 17, 19]:
                    #     cat_id -= 1
                    bbox = bbox[0]
                    bbox[2] = bbox[2] - bbox[0]
                    bbox[3] = bbox[3] - bbox[1]
                    res = {
                        'image_id': img2id[image],
                        'bbox': bbox,
                        'score': 1.0,
                        'category_id': cat_id,
                        'id': instance_id
                    }
                    dt_res.append(res)
                    instance_id += 1
            
        coco_dt = coco_api.loadRes(dt_res)

        
        # per-category evaluation
        for catid in range(len(self.all_parts_list)):
            print(f'Evaluating on {self.all_parts_list[catid]}: ')
            catid += 1
            coco_eval = COCOeval(coco_api, coco_dt, iouType='bbox')
            coco_eval.params.catIds = [catid]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        # all-category evaluation
        print('Evaluating on all categories')
        coco_eval = COCOeval(coco_api, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
        metric_results = {}
        for k in coco_metric_names:
            metric_results[k] = coco_eval.stats[coco_metric_names[k]].round(5)
        log_stats = {split_name: metric_results}

        with open(
            os.path.join(registry.get_path("output_dir"), f"evaluate{epoch}.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        res = metric_results
        res["agg_metrics"] = metric_results["mAP"]

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

        for gt, pred in zip(gt_result, pred_result):
            assert gt['id'] == pred['id']
            imagename = pred['image']
            instance_id = pred['id']
            pred_img = cv2.imread(imagename)
            gt_segs = gt['segmentation']
            for i, seg in enumerate(gt_segs):
                color = colormap[i]
                color = [(x * 255) for x in color]
                for polygon in seg:
                    polygon = list(map(int, polygon))
                    x1, y1, x2, y2 = polygon
                    # cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
                    # cv2.putText(pred_img, cls, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)

            pred_segs = pred['segmentation']
            for i, seg in enumerate(pred_segs):
                cls = self.all_parts_list[i]
                color = colormap[i]
                color = [(x * 255) for x in color]
                for polygon in seg:
                    polygon = list(map(int, polygon))
                    x1, y1, x2, y2 = polygon
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(pred_img, cls, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
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
