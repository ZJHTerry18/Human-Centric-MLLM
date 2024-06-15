import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.multitask_qa_datasets import (
    DetQADataset, PoseQADataset, ParsingQADataset, DenseCaptionDataset, InstanceCaptionDataset, InstanceGroundingDataset, InstancePartRECDataset, InstanceRECDataset, ParsingRECDataset, GRITGroundingDataset, InstructTuningDataset
)
from minigpt4.datasets.datasets.coco_caption import COCOCapDataset

class multitaskQABuilder(BaseDatasetBuilder):
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        train_dataset_cls = self.train_dataset_cls
        eval_dataset_cls = self.eval_dataset_cls
        datasets['train'] = train_dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=build_info.image_path,
            ann_paths=build_info.ann_path['train'],
            template_path=build_info.template_path,
            task=self.config.task,
        )
        if build_info.ann_path.get("val", None) is not None:
            datasets['val'] = eval_dataset_cls(
                vis_processor=self.vis_processors["eval"],
                text_processor=self.text_processors["eval"],
                vis_root=build_info.image_path,
                ann_paths=build_info.ann_path['val'],
                template_path=build_info.template_path,
                task=self.config.task,
            )

        return datasets

### instruct_tuning
@registry.register_builder("coco_instruct_tuning")
class COCOInstruct_tuningBuilder(multitaskQABuilder):
    train_dataset_cls = InstructTuningDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct_tuning/coco.yaml",
    }
    
@registry.register_builder("aic_instruct_tuning")
class AICInstruct_tuningBuilder(multitaskQABuilder):
    train_dataset_cls = InstructTuningDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct_tuning/aic.yaml",
    }

@registry.register_builder("ccslaion_instruct_tuning")
class CCSLAIONInstruct_tuningBuilder(multitaskQABuilder):
    train_dataset_cls = InstructTuningDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct_tuning/ccslaion.yaml",
    }



### dense caption builders
@registry.register_builder("coco_dense_caption")
class COCODenseCaptionBuilder(multitaskQABuilder):
    train_dataset_cls = DenseCaptionDataset
    eval_dataset_cls = DenseCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dense_caption/coco.yaml",
    }

@registry.register_builder("aic_dense_caption")
class AICDenseCaptionBuilder(multitaskQABuilder):
    train_dataset_cls = DenseCaptionDataset
    eval_dataset_cls = DenseCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dense_caption/aic.yaml",
    }

@registry.register_builder("cihp_dense_caption")
class CIHPDenseCaptionBuilder(multitaskQABuilder):
    train_dataset_cls = DenseCaptionDataset
    eval_dataset_cls = DenseCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dense_caption/CIHP.yaml",
    }

@registry.register_builder("ccslaion_dense_caption")
class CCSLaionDenseCaptionBuilder(multitaskQABuilder):
    train_dataset_cls = DenseCaptionDataset
    eval_dataset_cls = DenseCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dense_caption/ccslaion.yaml",
    }

@registry.register_builder('coco_inst_pose')
class COCOInstPoseBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/pose/coco_inst_pose.yaml",
    }

@registry.register_builder('coco_inst_app')
class COCOInstAppBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/appearance/coco_inst_app.yaml",
    }

@registry.register_builder('coco_inst_mod')
class COCOInstModBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/modality/coco_inst_mod.yaml",
    }

@registry.register_builder('coco_inst_rel')
class COCOInstRelBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/relation/coco_inst_rel.yaml",
    }

@registry.register_builder('aic_inst_pose')
class AICInstPoseBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/pose/aic_inst_pose.yaml",
    }

@registry.register_builder('aic_inst_app')
class AICInstAppBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/appearance/aic_inst_app.yaml",
    }

@registry.register_builder('aic_inst_mod')
class AICInstModBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/modality/aic_inst_mod.yaml",
    }

@registry.register_builder('aic_inst_rel')
class AICInstRelBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/relation/aic_inst_rel.yaml",
    }

@registry.register_builder('ccslaion_inst_pose')
class CCSLaionInstPoseBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/pose/ccslaion_inst_pose.yaml",
    }

@registry.register_builder('ccslaion_inst_app')
class CCSLaionInstAppBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/appearance/ccslaion_inst_app.yaml",
    }

@registry.register_builder('ccslaion_inst_mod')
class CCSLaionInstModBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/modality/ccslaion_inst_mod.yaml",
    }

@registry.register_builder('ccslaion_inst_rel')
class CCSLaionInstRelBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceCaptionDataset
    eval_dataset_cls = InstanceCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/relation/ccslaion_inst_rel.yaml",
    }
#TODO: add instance caption builders sam
    
@registry.register_builder('coco_inst_grounding')
class COCOInstGroundBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceGroundingDataset
    eval_dataset_cls = InstanceGroundingDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/grounding/coco_inst_grounding.yaml",
    }

@registry.register_builder('coco_inst_rec')
class COCOInstRECBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceRECDataset
    eval_dataset_cls = InstanceRECDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/rec/coco_inst_rec.yaml",
    }

@registry.register_builder('coco_inst_part_rec')
class COCOInstPartRECBuilder(multitaskQABuilder):
    train_dataset_cls = InstancePartRECDataset
    eval_dataset_cls = InstancePartRECDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/rec/coco_inst_part_rec.yaml",
    }

@registry.register_builder('aic_inst_grounding')
class AICInstGroundBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceGroundingDataset
    eval_dataset_cls = InstanceGroundingDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/grounding/aic_inst_grounding.yaml",
    }

@registry.register_builder('aic_inst_rec')
class AICInstRECBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceRECDataset
    eval_dataset_cls = InstanceRECDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/rec/aic_inst_rec.yaml",
    }

@registry.register_builder('aic_inst_part_rec')
class AICInstPartRECBuilder(multitaskQABuilder):
    train_dataset_cls = InstancePartRECDataset
    eval_dataset_cls = InstancePartRECDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/rec/aic_inst_part_rec.yaml",
    }

@registry.register_builder('ccslaion_inst_grounding')
class CCSLaionInstGroundBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceGroundingDataset
    eval_dataset_cls = InstanceGroundingDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/grounding/ccslaion_inst_grounding.yaml",
    }

@registry.register_builder('ccslaion_inst_rec')
class CCSLaionInstRECBuilder(multitaskQABuilder):
    train_dataset_cls = InstanceRECDataset
    eval_dataset_cls = InstanceRECDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/rec/ccslaion_inst_rec.yaml",
    }

@registry.register_builder('ccslaion_inst_part_rec')
class CCSLaionInstPartRECBuilder(multitaskQABuilder):
    train_dataset_cls = InstancePartRECDataset
    eval_dataset_cls = InstancePartRECDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/rec/ccslaion_inst_part_rec.yaml",
    }

@registry.register_builder('cihp_parsing_rec')
class CIHPParsingRECBuilder(multitaskQABuilder):
    train_dataset_cls = ParsingRECDataset
    eval_dataset_cls = ParsingQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instance_tasks/rec/cihp_part.yaml"
    }

@registry.register_builder('grit_grounding')
class GRITGroundingBuilder(multitaskQABuilder):
    train_dataset_cls = GRITGroundingDataset
    eval_dataset_cls = GRITGroundingDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/grit/grit.yaml"
    }

# ### old tasks: detection, pose, parsing

@registry.register_builder('coco_det_qa')
class COCODetQABuilder(multitaskQABuilder):
    train_dataset_cls = DetQADataset
    eval_dataset_cls = DetQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/detection/coco.yaml",
    }

@registry.register_builder('crowdhuman_det_qa')
class crowdhumanDetQABuilder(multitaskQABuilder):
    train_dataset_cls = DetQADataset
    eval_dataset_cls = DetQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/detection/crowdhuman.yaml",
    }

@registry.register_builder('cityperson_det_qa')
class citypersonDetQABuilder(multitaskQABuilder):
    train_dataset_cls = DetQADataset
    eval_dataset_cls = DetQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/detection/cityperson.yaml",
    }

@registry.register_builder('ECP_det_qa')
class ECPDetQABuilder(multitaskQABuilder):
    train_dataset_cls = DetQADataset
    eval_dataset_cls = DetQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/detection/ECP.yaml",
    }

@registry.register_builder('WiderPerson_det_qa')
class WiderPersonDetQABuilder(multitaskQABuilder):
    train_dataset_cls = DetQADataset
    eval_dataset_cls = DetQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/detection/WiderPerson.yaml",
    }

@registry.register_builder('widerpedestrian_det_qa')
class widerpedestrianDetQABuilder(multitaskQABuilder):
    train_dataset_cls = DetQADataset
    eval_dataset_cls = DetQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/detection/widerpedestrian.yaml",
    }

@registry.register_builder("coco_pose_qa")
class COCOPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/coco.yaml",
    }

@registry.register_builder("aic_pose_qa")
class AIChallengerPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/ai_challenger.yaml",
    }

@registry.register_builder("PoseTrack_pose_qa")
class PoseTrackPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/PoseTrack.yaml",
    }

@registry.register_builder("MHP_pose_qa")
class MHPPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/MHP.yaml",
    }

@registry.register_builder("3DPW_pose_qa")
class _3DPWPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/3DPW.yaml",
    }

@registry.register_builder("halpe_pose_qa")
class halpePoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/halpe.yaml",
    }

@registry.register_builder("penn_action_pose_qa")
class PennActionPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/penn_action.yaml",
    }
    
@registry.register_builder("JRDB_pose_qa")
class JRDBPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/JRDB.yaml",
    }

@registry.register_builder("human36m_pose_qa")
class human36mPoseQABuilder(multitaskQABuilder):
    train_dataset_cls = PoseQADataset
    eval_dataset_cls = PoseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/pose_estimation/human36m.yaml",
    }


@registry.register_builder("cihp_parsing_qa")
class CIHPParsingQABuilder(multitaskQABuilder):
    train_dataset_cls = ParsingQADataset
    eval_dataset_cls = ParsingQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/parsing/cihp.yaml",
    }

@registry.register_builder("lip_parsing_qa")
class LIPParsingQABuilder(multitaskQABuilder):
    train_dataset_cls = ParsingQADataset
    eval_dataset_cls = ParsingQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/parsing/lip.yaml",
    }

@registry.register_builder("vip_parsing_qa")
class VIPParsingQABuilder(multitaskQABuilder):
    train_dataset_cls = ParsingQADataset
    eval_dataset_cls = ParsingQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/parsing/vip.yaml",
    }

@registry.register_builder("deepfashion2_parsing_qa")
class DeepFashion2ParsingQABuilder(multitaskQABuilder):
    train_dataset_cls = ParsingQADataset
    eval_dataset_cls = ParsingQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/parsing/deepfashion2.yaml",
    }

@registry.register_builder("modanet_parsing_qa")
class ModanetParsingQABuilder(multitaskQABuilder):
    train_dataset_cls = ParsingQADataset
    eval_dataset_cls = ParsingQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_qa/parsing/modanet.yaml",
    }