import json
import numpy as np

# detection datasets
json_list = {
    'COCO Train': '/data/datasets/dataset_pedestrian_detection/coco/annotations/coco_detection_train_bbox.jsonl',
    'COCO Val': '/data/datasets/dataset_pedestrian_detection/coco/annotations/coco_detection_val_bbox.jsonl',
    'Crowdhuman Train': '/data/datasets/dataset_pedestrian_detection/crowdhuman/crowdhuman_detection_train_bbox.jsonl',
    'Crowdhuman Val': '/data/datasets/dataset_pedestrian_detection/crowdhuman/crowdhuman_detection_val_bbox.jsonl',
    'ECP Train': '/data/datasets/dataset_pedestrian_detection/ECP/ECP_detection_train_bbox.jsonl',
    'ECP Val': '/data/datasets/dataset_pedestrian_detection/ECP/ECP_detection_val_bbox.jsonl',
    'Widerperson Train': '/data/datasets/dataset_pedestrian_detection/WiderPerson/WiderPerson_detection_train_bbox.jsonl',
    'Widerperson Val': '/data/datasets/dataset_pedestrian_detection/WiderPerson/WiderPerson_detection_val_bbox.jsonl',
    'Widerpedestrian Train': '/data/datasets/dataset_pedestrian_detection/widerpedestrian/widerpedestrian_detection_train_bbox.jsonl',
    'Widerpedestrian Val': '/data/datasets/dataset_pedestrian_detection/widerpedestrian/widerpedestrian_detection_val_bbox.jsonl',
    'Citypersons Train': '/data/datasets/dataset_pedestrian_detection/cityperson/cityperson_detection_train_bbox.jsonl',
    'Citypersons Val': '/data/datasets/dataset_pedestrian_detection/cityperson/cityperson_detection_val_bbox.jsonl'
}

max_limit = 50
for k, ann in json_list.items():
    dats = []
    imgs = []
    with open(ann, 'r') as f:
        for line in f.readlines():
            dats.append(json.loads(line))
            imgs.append(json.loads(line)['image'])
    dataset_name = k
    num_samples = len(dats)
    imgs = set(imgs)
    num_imgs = len(imgs)
    # box_num = np.array([len(d['boxes']) for d in dats])
    # oversize_portion = float(np.sum(box_num > max_limit)) / len(box_num)
    print("{} dataset: {} samples, {} images".format(dataset_name, num_samples, num_imgs))

# parsing datasets
json_list = {
    'CIHP Train': '/data/datasets/dataset_parsing/CIHP/instance-level_human_parsing/CIHP_train_polygon.jsonl',
    'CIHP Val': '/data/datasets/dataset_parsing/CIHP/instance-level_human_parsing/CIHP_val_polygon.jsonl',
    'DeepFashion2 Train': '/data/datasets/dataset_parsing/DeepFashion2/deepfashion2_train_polygon.jsonl',
    'DeepFashion2 Val': '/data/datasets/dataset_parsing/DeepFashion2/deepfashion2_train_polygon.jsonl',
    'LIP Train': '/data/datasets/dataset_parsing/LIP/TrainVal_parsing_annotations/LIP_train_polygon.jsonl',
    'LIP Val': '/data/datasets/dataset_parsing/LIP/TrainVal_parsing_annotations/LIP_train_polygon.jsonl',
    'VIP Train': '/data/datasets/dataset_parsing/VIP_new/VIP_Fine/annotations/VIP_train_polygon.jsonl',
    'VIP Val': '/data/datasets/dataset_parsing/VIP_new/VIP_Fine/annotations/VIP_val_polygon.jsonl'
}

max_limit = 200
for k, ann in json_list.items():
    dats = []
    imgs = []
    with open(ann, 'r') as f:
        for line in f.readlines():
            dats.append(json.loads(line))
            imgs.append(json.loads(line)['image'])
    dataset_name = k
    num_samples = len(dats)
    imgs = set(imgs)
    num_imgs = len(imgs)
    poly_lens = []
    # for d in dats:
    #     l = 0
    #     for p, v in d['segmentation'].items():
    #         v_flat = [item for sublist in v for item in sublist]
    #         l += len(v_flat)
    #     poly_lens.append(l)
    # poly_lens = np.array(poly_lens)
    # oversize_portion = float(np.sum(poly_lens > max_limit)) / len(poly_lens)
    print("{} dataset: {} samples, {} images".format(dataset_name, num_samples, num_imgs))

# pose datasets
json_list = {
    '3DPW train': '/data/datasets/dataset_pose/3DPW/3DPW_pose_train_bbox.jsonl',
    '3DPW val': '/data/datasets/dataset_pose/3DPW/3DPW_pose_validation_bbox.jsonl',
    'aiChallenger train': '/data/datasets/dataset_pose/aic/aic_annotations/aic_pose_train_bbox.jsonl',
    'aiChallenger val': '/data/datasets/dataset_pose/aic/aic_annotations/aic_pose_val_bbox.jsonl',
    'COCO train': '/data/datasets/dataset_pose/coco/annotations/coco_pose_train_bbox.jsonl',
    'COCO val': '/data/datasets/dataset_pose/coco/annotations/coco_pose_val_bbox.jsonl',
    'halpe train': '/data/datasets/dataset_pose/halpe/halpe_pose_train_bbox.jsonl',
    'halpe val': '/data/datasets/dataset_pose/halpe/halpe_pose_val_bbox.jsonl',
    'human3.6m train': '/data/datasets/dataset_pose/human3.6m/h36m_pose_train_bbox.jsonl',
    'human3.6m val': '/data/datasets/dataset_pose/human3.6m/h36m_pose_valid_bbox.jsonl',
    'JRDB train': '/data/datasets/dataset_pose/JRDB/JRDB_pose_train_bbox.jsonl',
    'MHP train': '/data/datasets/dataset_pose/MHP/MHP_pose_train_bbox.jsonl',
    'MHP val': '/data/datasets/dataset_pose/MHP/MHP_pose_val_bbox.jsonl',
    'pennAction train': '/data/datasets/dataset_pose/PennAction/Penn_Action_pose_train_bbox.jsonl',
    'PoseTrack train': '/data/datasets/dataset_pose/PoseTrack/PoseTrack_pose_train_bbox.jsonl',
    'PoseTrack val': '/data/datasets/dataset_pose/PoseTrack/PoseTrack_pose_val_bbox.jsonl'
}

for k, ann in json_list.items():
    dats = []
    imgs = []
    with open(ann, 'r') as f:
        for line in f.readlines():
            dats.append(json.loads(line))
            imgs.append(json.loads(line)['image'])
    dataset_name = k
    num_samples = len(dats)
    num_imgs = len(set(imgs))
    print("{} dataset: {} samples, {} imgs".format(dataset_name, num_samples, num_imgs))