import json
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm

new_write_root = './filtered_annotations'
os.makedirs(new_write_root, exist_ok=True)

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

image_root = {
    '3DPW train': '/data/datasets/dataset_pose/3DPW',
    '3DPW val': '/data/datasets/dataset_pose/3DPW',
    'aiChallenger train': '/data/datasets/dataset_pose/aic',
    'aiChallenger val': '/data/datasets/dataset_pose/aic',
    'COCO train': '/data/datasets/dataset_pose/coco',
    'COCO val': '/data/datasets/dataset_pose/coco',
    'halpe train': '/data/datasets/dataset_pose/halpe',
    'halpe val': '/data/datasets/dataset_pose/halpe',
    'human3.6m train': '/data/datasets/dataset_pose/human3.6m',
    'human3.6m val': '/data/datasets/dataset_pose/human3.6m',
    'JRDB train': '/data/datasets/dataset_pose/JRDB',
    'MHP train': '/data/datasets/dataset_pose/MHP',
    'MHP val': '/data/datasets/dataset_pose/MHP',
    'pennAction train': '/data/datasets/dataset_pose/PennAction',
    'PoseTrack train': '/data/datasets/dataset_pose/PoseTrack',
    'PoseTrack val': '/data/datasets/dataset_pose/PoseTrack'
}

for dataset_name, json_path in json_list.items():
    print("{} dataset: ".format(dataset_name))
    datalist = []
    with open(json_path, 'r') as f:
        for line in f.readlines():
            datalist.append(json.loads(line))
    new_datalist = []
    for dat in tqdm(datalist):
        image_path = osp.join(image_root[dataset_name], dat['image'])
        try:
            image = Image.open(image_path).convert('RGB')
            assert image is not None
            new_datalist.append(dat)
        except:
            print("{} not readable".format(image_path))
    
    write_path = osp.join(new_write_root, osp.basename(json_path))
    with open(write_path, 'w') as f:
        for l in new_datalist:
            f.write(json.dumps(l) + '\n')


# parsing datasets
json_list = {
    'CIHP Train': '/data/datasets/dataset_parsing/CIHP/instance-level_human_parsing/CIHP_train_polygon.jsonl',
    'CIHP Val': '/data/datasets/dataset_parsing/CIHP/instance-level_human_parsing/CIHP_val_polygon.jsonl',
    'Deepfashion2 Train': '/data/datasets/dataset_parsing/DeepFashion2/deepfashion2_train_polygon.jsonl',
    'Deepfashion2 Val': '/data/datasets/dataset_parsing/DeepFashion2/deepfashion2_val_polygon.jsonl',
    'LIP Train': '/data/datasets/dataset_parsing/LIP/TrainVal_parsing_annotations/LIP_train_polygon.jsonl',
    'LIP Val': '/data/datasets/dataset_parsing/LIP/TrainVal_parsing_annotations/LIP_val_polygon.jsonl',
    'VIP Train': '/data/datasets/dataset_parsing/VIP_new/VIP_Fine/annotations/VIP_train_polygon.jsonl',
    'VIP Val': '/data/datasets/dataset_parsing/VIP_new/VIP_Fine/annotations/VIP_val_polygon.jsonl',
}

image_root = {
    'CIHP Train': '/data/datasets/dataset_parsing/CIHP/instance-level_human_parsing',
    'CIHP Val': '/data/datasets/dataset_parsing/CIHP/instance-level_human_parsing',
    'Deepfashion2 Train': '/data/datasets/dataset_parsing/DeepFashion2',
    'Deepfashion2 Val': '/data/datasets/dataset_parsing/DeepFashion2',
    'LIP Train': '/data/datasets/dataset_parsing/LIP/data',
    'LIP Val': '/data/datasets/dataset_parsing/LIP/data',
    'VIP Train': '/data/datasets/dataset_parsing/VIP_new/VIP_Fine/Images',
    'VIP Val': '/data/datasets/dataset_parsing/VIP_new/VIP_Fine/Images',
}

for dataset_name, json_path in json_list.items():
    print("{} dataset: ".format(dataset_name))
    datalist = []
    with open(json_path, 'r') as f:
        for line in f.readlines():
            datalist.append(json.loads(line))
    new_datalist = []
    for dat in tqdm(datalist):
        image_path = osp.join(image_root[dataset_name], dat['image'])
        try:
            image = Image.open(image_path).convert('RGB')
            assert image is not None
            new_datalist.append(dat)
        except:
            print("{} not readable".format(image_path))
    
    write_path = osp.join(new_write_root, osp.basename(json_path))
    with open(write_path, 'w') as f:
        for l in new_datalist:
            f.write(json.dumps(l) + '\n')


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
    'Widerpedestrian Val': '/data/datasets/dataset_pedestrian_detection/widerpedestrian/widerpedestrian_detection_train_bbox.jsonl',
    'Citypersons Train': '/data/datasets/dataset_pedestrian_detection/cityperson/cityperson_detection_train_bbox.jsonl',
    'Citypersons Val': '/data/datasets/dataset_pedestrian_detection/cityperson/cityperson_detection_val_bbox.jsonl'
}

image_root = {
    'COCO Train': '/data/datasets/dataset_pedestrian_detection/coco',
    'COCO Val': '/data/datasets/dataset_pedestrian_detection/coco',
    'Crowdhuman Train': '/data/datasets/dataset_pedestrian_detection/crowdhuman',
    'Crowdhuman Val': '/data/datasets/dataset_pedestrian_detection/crowdhuman',
    'ECP Train': '/data/datasets/dataset_pedestrian_detection/ECP',
    'ECP Val': '/data/datasets/dataset_pedestrian_detection/ECP',
    'Widerperson Train': '/data/datasets/dataset_pedestrian_detection/WiderPerson',
    'Widerperson Val': '/data/datasets/dataset_pedestrian_detection/WiderPerson',
    'Widerpedestrian Train': '/data/datasets/dataset_pedestrian_detection/widerpedestrian',
    'Widerpedestrian Val': '/data/datasets/dataset_pedestrian_detection/widerpedestrian',
    'Citypersons Train': '/data/datasets/dataset_pedestrian_detection/cityperson',
    'Citypersons Val': '/data/datasets/dataset_pedestrian_detection/cityperson'
}

for dataset_name, json_path in json_list.items():
    print("{} dataset: ".format(dataset_name))
    datalist = []
    with open(json_path, 'r') as f:
        for line in f.readlines():
            datalist.append(json.loads(line))
    new_datalist = []
    for dat in tqdm(datalist):
        image_path = osp.join(image_root[dataset_name], dat['image'])
        try:
            image = Image.open(image_path).convert('RGB')
            assert image is not None
            new_datalist.append(dat)
        except:
            print("{} not readable".format(image_path))
    
    write_path = osp.join(new_write_root, osp.basename(json_path))
    with open(write_path, 'w') as f:
        for l in new_datalist:
            f.write(json.dumps(l) + '\n')