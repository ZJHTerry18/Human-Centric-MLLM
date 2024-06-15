import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn

from torch.nn import functional as F

class HumParEvaluator:
    """
    Evaluate human parsing metrics, specifically, for Human3.6M
    """

    def __init__(
            self,
            class_names,
            ignore_label=255,
            distributed=True,
            output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)

        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self._class_names = class_names
        self._num_classes = len(self._class_names)
        self._contiguous_id_to_dataset_id = {i: k for i, k in enumerate(
            self._class_names)}  # Dict that maps contiguous training ids to COCO category ids
        self._ignore_label = ignore_label

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)
        self._predictions = []

    def process(self, gts, preds):
        for _idx, par_pred in enumerate(preds):
            gt = gts[_idx].astype(np.int32)
            par_pred_size = par_pred.shape
            gt_h, gt_w = gt.shape[-2], gt.shape[-1]

            if par_pred_size[-2]!=gt_h or par_pred_size[-1]!=gt_w:
                par_pred = F.upsample(par_pred.unsqueeze(0), (gt_h, gt_w),mode='bilinear')
                output = par_pred[0]
            else:
                output = par_pred

            pred = np.array(output, dtype=np.int32)

            if len(pred.shape)!=2:
                import pdb;
                pdb.set_trace()


            self._conf_matrix += self.get_confusion_matrix(gt, pred, self._num_classes, self._ignore_label).astype(np.int64)


    def get_confusion_matrix(self, seg_gt, seg_pred, num_class, ignore=-1):
        import time
        start = time.time()
        ignore_index = seg_gt != ignore
        seg_gt = seg_gt[ignore_index]
        try:
            seg_pred = seg_pred[ignore_index]
        except:
            import pdb;pdb.set_trace()

        index = (seg_gt * num_class + seg_pred).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((num_class, num_class))

        for i_label in range(num_class):
            for i_pred in range(num_class):
                cur_index = i_label * num_class + i_pred
                if cur_index < len(label_count):
                    confusion_matrix[i_label,
                                     i_pred] = label_count[cur_index]
        return confusion_matrix

    def evaluate(self):
        """
        
        :return: mean_IoU, IoU_array, pixel_acc, mean_acc 
        """
        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal().astype(np.float64)
        pos_gt = np.sum(self._conf_matrix, axis=0).astype(np.float64)
        # class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix, axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        # fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        # res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        # if self._output_dir:
        #     file_path = os.path.join(self._output_dir, "human_parsing_evaluation.pth")
        #     with open(file_path, "wb") as f:
        #         torch.save(res, f)
        for k, v in res.items():
            print(f"{k}: {v}")
        return res
    
    # def encode_json_sem_seg(self, sem_seg, input_file_name):
    #     """
    #     Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
    #     See http://cocodataset.org/#format-results
    #     """
    #     json_list = []
    #     for label in np.unique(sem_seg):
    #         if self._contiguous_id_to_dataset_id is not None:
    #             assert (
    #                 label in self._contiguous_id_to_dataset_id
    #             ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
    #             dataset_id = self._contiguous_id_to_dataset_id[label]
    #         else:
    #             dataset_id = int(label)
    #         mask = (sem_seg == label).astype(np.uint8)
    #         mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
    #         mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
    #         json_list.append(
    #             {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
    #         )
    #     return json_list
