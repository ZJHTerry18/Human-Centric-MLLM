"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask
from minigpt4.tasks.image_text_pretrain import ImageTextPretrainTask, ImageTextPretrainPointRegTask
from minigpt4.tasks.detection_eval import DetectionEvalTask
from minigpt4.tasks.pose_eval import PoseEvalTask
from minigpt4.tasks.pose_reg_eval import PoseRegEvalTask
from minigpt4.tasks.parsing_eval import ParsingEvalTask
from minigpt4.tasks.parsing_bbox_eval import ParsingBoxEvalTask
from minigpt4.tasks.hcm_eval import HCMCaptionEvalTask, HCMGroundingEvalTask, HCMRECEvalTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "ImageTextPretrainTask",
    "ImageTextPretrainPointRegTask",
    "DetectionEvalTask",
    "PoseEvalTask",
    "PoseRegEvalTask",
    "ParsingEvalTask",
    "ParsingBoxEvalTask",
    "HCMCaptionEvalTask",
    "HCMGroundingEvalTask",
    "HCMRECEvalTask",
]
