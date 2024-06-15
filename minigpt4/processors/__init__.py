"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from minigpt4.processors.base_processor import BaseProcessor
from minigpt4.processors.blip_processors import (
    Blip2ImageTrainProcessor,
    Blip2ImageEvalProcessor,
    BlipCaptionProcessor,
    Blip2ImageMultitaskTrainProcessor,
    Blip2ImageMultitaskEvalProcessor
)
from minigpt4.processors.blip_multitask_qa_image_processors import (
    Blip2ImageDetQATrainProcessor,
    Blip2ImageDetQAEvalProcessor,
    Blip2ImagePoseQATrainProcessor,
    Blip2ImagePoseQAEvalProcessor,
    Blip2ImageParsingQATrainProcessor,
    Blip2ImageParsingQAEvalProcessor
)

from minigpt4.processors.blip_multitask_qa_text_processors import (
    Blip2TextDetQATrainProcessor,
    Blip2TextDetQAEvalProcessor,
    Blip2TextPoseQATrainProcessor,
    Blip2TextPoseQAEvalProcessor,
    Blip2TextParsingQATrainProcessor,
    Blip2TextParsingQAEvalProcessor
)

from minigpt4.processors.blip_reg_text_processors import (
    Blip2TextPoseRegTrainProcessor,
    Blip2TextPoseRegEvalProcessor
)

from minigpt4.processors.blip_inst_text_processors import (
    Blip2TextDensecapTrainProcessor,
    Blip2TextDensecapEvalProcessor,
    Blip2TextInstcapTrainProcessor,
    Blip2TextInstcapEvalProcessor
)

from minigpt4.common.registry import registry

__all__ = [
    "BaseProcessor",
    "Blip2ImageTrainProcessor",
    "Blip2ImageEvalProcessor",
    "BlipCaptionProcessor",
    "Blip2ImageDetQATrainProcessor",
    "Blip2ImageDetQAEvalProcessor",
    "Blip2ImagePoseQATrainProcessor",
    "Blip2ImagePoseQAEvalProcessor",
    "Blip2ImageParsingQATrainProcessor",
    "Blip2ImageParsingQAEvalProcessor",
    "Blip2TextDetQATrainProcessor",
    "Blip2TextDetQAEvalProcessor",
    "Blip2TextPoseQATrainProcessor",
    "Blip2TextPoseQAEvalProcessor",
    "Blip2TextParsingQATrainProcessor",
    "Blip2TextParsingQAEvalProcessor",
    "Blip2TextPoseRegTrainProcessor",
    "Blip2TextPoseRegEvalProcessor",
    "Blip2TextInstcapTrainProcessor",
    "Blip2TextInstcapEvalProcessor",
    "Blip2TextDensecapTrainProcessor",
    "Blip2TextDensecapEvalProcessor"

]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
