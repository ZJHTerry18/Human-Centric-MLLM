import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class ReferCOCOSegDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, dataset='refcoco', splitBy='unc'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.refer = REFERSeg(ann_path, vis_root, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split="train")

        self.instruction_pool = [
            "[refer segmentation] {}",
            "[refer segmentation] give me the boundary points of {}",
            "[refer segmentation] where is {} ? Output the location in the format of boundary points",
            "[refer segmentation] from this image, tell me the location of {} in boundary points",
            "[refer segmentation] the boundary points of {} is",
            "[refer segmentation] could you tell me the boudnary points for {} ?",
            "[refer segmentation] where can I locate the {} ? Give the boundary points of the location",
        ]

        self.quantize_bins = 100
        self.max_seg_len = 200


    def __len__(self):
        return len(self.ref_ids)

    def preprocess(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        image_file = 'COCO_train2014_{:0>12}.jpg'.format(ref["image_id"])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image_orig_size = image.size
        image = self.vis_processor(image)
        image_new_size = [image.shape[1], image.shape[2]]

        image_new_size = [self.quantize_bins, self.quantize_bins]

        sample_sentence = random.choice(ref['sentences'])['raw']
        refer_sentence = self.text_processor(sample_sentence)
    
        segmentation = self.refer.getRefSeg(ref['ref_id'])
        seg_strs = []
        seg_len = 0
        for seg in segmentation:
            poly = np.array(seg).reshape(-1, 2).astype(float)
            poly = poly / np.array([image_orig_size[0], image_orig_size[1]]) * image_new_size
            poly = poly.flatten().tolist()
            poly = [int(x) for x in poly]
            seg_len += len(poly)
            if seg_len > self.max_seg_len:
                break
            poly_template = "{{" + "<{}>" * len(poly) + "}}"
            poly_str = poly_template.format(*poly)
            seg_strs.append(poly_str)
        seg_str = '<delim>'.join(seg_strs)

        return {
            "image": image,
            "refer_sentence": refer_sentence,
            "segmentation": seg_str,
            "image_id": ref['image_id'],
        }


    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(data['refer_sentence'])

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        # # debug
        # import cv2
        # import re
        # image = data['image']
        # show_img = image.permute(1,2,0).contiguous().numpy()
        # show_img = (show_img * 255).astype('uint8')
        # show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
        # import scipy.io as scio
        # colormap = scio.loadmat('/data/datasets/dataset_parsing/CIHP/human_colormap.mat')['colormap']
        # seg_caption = data['segmentation']
        # seg_pattern = r"<(\d+)>"
        # H, W, _, = show_img.shape
        # part = seg_caption.split('<delim>')
        # for p in part:
        #     poly_list = re.findall(seg_pattern, p)
        #     poly = np.array(poly_list).reshape(-1, 2).astype(float) / self.quantize_bins * np.array([W, H])
        #     cv2.polylines(show_img, np.int32([poly]), True, (0, 255, 0), 2)
        # cv2.imwrite(os.path.join('debug', 'refseg', str(data['image_id']) + '.jpg'), show_img)
        # print(data['image_id'])
        # print(instruction)
        # print(data['segmentation'])

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['segmentation'],
            "image_id": data['image_id'],
        }


class InvReferCOCOSegDataset(ReferCOCOSegDataset):
    def __init__(self, *args, **kwargs):
        super(InvReferCOCOSegDataset, self).__init__(*args, **kwargs)

        self.instruction_pool = [
            "[identify segmentation] {}",
            "[identify segmentation] what object is in this location {}",
            "[identify segmentation] identify the object present at this location {}",
            "[identify segmentation] what is it in {}",
            "[identify segmentation] describe this object in {}",
            "[identify segmentation] this {} is",
            "[identify segmentation] the object in {} is",
            ]

    def __getitem__(self, index):
        data = self.preprocess(index)

        instruction = random.choice(self.instruction_pool).format(data['segmentation'])

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        # # debug
        # import cv2
        # import re
        # image = data['image']
        # show_img = image.permute(1,2,0).contiguous().numpy()
        # show_img = (show_img * 255).astype('uint8')
        # show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)
        # import scipy.io as scio
        # colormap = scio.loadmat('/data/datasets/dataset_parsing/CIHP/human_colormap.mat')['colormap']
        # seg_caption = data['segmentation']
        # seg_pattern = r"<(\d+)>"
        # H, W, _, = show_img.shape
        # part = seg_caption.split('<delim>')
        # for p in part:
        #     poly_list = re.findall(seg_pattern, p)
        #     poly = np.array(poly_list).reshape(-1, 2).astype(float) / self.quantize_bins * np.array([W, H])
        #     cv2.polylines(show_img, np.int32([poly]), True, (0, 255, 0), 2)
        #     cv2.imwrite(os.path.join('debug', 'refseg', str(data['image_id']) + '.jpg'), show_img)
        # print(data['image_id'])
        # print(instruction)
        # print(self.text_processor(data['refer_sentence']))
        
        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": self.text_processor(data['refer_sentence']),
            "image_id": data['image_id'],
        }

class RefCOCOSegEvalData(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['img_id']
        sent = data['sents']
        image_path = os.path.join(self.root_path, f'{img_id[:27]}.jpg')#os.path.join(self.root_path, f'{img_id[:27]}.jpg'.split('_')[-1]) for various styles of img_path between json and exact files
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[refer segmentation] give me the boundary points of {sent}"
        return image, question, img_id

class REFERSeg:
    def __init__(self, data_root, vis_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        dataset = dataset.split('inv')[-1]  # inv dataset is stored in the same path as normal dataset
        print('loading dataset %s into memory...' % dataset)
        self.ann_dir = os.path.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.vis_root = vis_root
        elif dataset == 'refclef':
            raise 'No RefClef image data'
        else:
            raise 'No refer dataset is called [%s]' % dataset

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = os.path.join(self.ann_dir, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        # load annotations from data/dataset/instances.json
        instances_file = os.path.join(self.ann_dir, 'instances_poly.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if
                            split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    raise 'No such split [%s]' % split
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]
    
    def getRefSeg(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['segmentation']

    def showRef(self, ref, seg_box='box'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(os.path.join(self.vis_root, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                raise NotImplementedError('RefClef is not downloaded')
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)
