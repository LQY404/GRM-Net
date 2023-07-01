import copy
import logging
import os.path as osp

import numpy as np
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt

import cv2

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .utils.comm import encode, decode, display_instances, tokenize
from pycocotools import mask as MASK
import pickle
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

import imgviz
from PIL import Image
import os


CATEGORY_refcoco = {}
CATE = {
    0: 77, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
    13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23,
    27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34,
    41: 35, 42: 36, 43: 37, 44: 38, 46: 39, 47: 40, 48: 41, 49: 42, 50: 43, 51: 44, 52: 45, 53: 46, 54: 47, 55: 48, 56: 49, 57: 50, 58: 51, 59: 52, 60: 53, 61: 54, 62: 55, 63: 56, 64: 57, 65: 58,
    67: 59,
    70: 60,
    72: 61, 73: 62, 74: 63, 75: 64, 76: 65, 77: 66, 78: 67, 79: 68,
    81: 69, 82: 70,
    84: 71, 85: 72, 86: 73, 87: 74, 88: 75,
    90: 76
}
for i in range(1, 91):
    if i <= 11:
        CATEGORY_refcoco[i] = i - 1
        continue

    if i >= 13 and i <= 25:
        CATEGORY_refcoco[i] = i - 2
        continue

    if i == 27 or i == 28:
        CATEGORY_refcoco[i] = i - 3
        continue

    if 31 <= i <= 39:
        CATEGORY_refcoco[i] = i - 5
        continue
    
    if 41 <= i <= 44:
        CATEGORY_refcoco[i] = i - 6
        continue

    if 46 <= i <= 65:
        CATEGORY_refcoco[i] = i - 7
        continue

    if i == 67:
        CATEGORY_refcoco[i] = 59
        continue

    if i == 70:
        CATEGORY_refcoco[i] = 60
        continue

    if 72 <= i <= 79:
        CATEGORY_refcoco[i] = i - 11
        continue

    if i == 81 or i == 82:
        CATEGORY_refcoco[i] = i - 12
        continue
    
    if 84 <= i <= 88:
        CATEGORY_refcoco[i] = i - 13
        continue

    if i == 90:
        CATEGORY_refcoco[i] = 76
        continue

ICATE = {}
for k, v in CATE.items():
    ICATE[v] = k

# from pytorch_pretrained_bert.tokenization import BertTokenizer

class RefcocoDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # self.augmentation = build_augmentation(cfg, is_train)
        self.augmentation = None # 不使用数据增强
        print("不使用数据增强") if self.augmentation is None else print("使用数据增强")

        

        self.split = ('train' if is_train else 'val')
        print("训练模式") if is_train else print("测试模式")

        # self.ref_root = "/home/lingpeng/project/demo/data/"/
        self.ref_root = "/nfs/demo/data/"
        self.ref_type = ["refcoco", "refcoco+", "refcocog", "refclef"][0]
        self.ref_split = {
            "refcoco": ["unc", "google"],
            "refcoco+": ["unc"],
            "refcocog": ['umd', "google"],
            "refclef": ['unc', "berkeley"]
        }[self.ref_type][0]

        # vocab_file = os.path.join(self.ref_root, self.ref_type, 'picked_vocab_train.json')
        vocab_file = os.path.join("/nfs/crefs/", "dict", self.ref_type, 'picked_c_vocab.json')

        self.simple = False
        easy = True
        begin = True
        use_attn = True
        use_iter = False
        # if self.simple:
        #     self.file_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_101_1x_iep_ref_simple" + ('_easyconcat' if easy else "") + ("_atend" if not begin else "_atbegin") + ("_useattn" if use_attn else "")
        # else:
        #     self.file_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_101_1x_iep_ref" + ('_easyconcat' if easy else "") + ("_atend" if not begin else "_atbegin") + ("_useattn" if use_attn else "")
        
        self.file_root = "/home/lingpeng/project/Adet/inference_dir/mrcnn_R_101_1x_refcoco_unc_v4_pick_0.8"  \
                                                        +  ("_simple" if self.simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_simple" if use_attn else "")  \
                                                        + ("_use_1iter_withmattn_newversion" if use_iter else "")
        # self.file_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_50_1x_iep_ref_v2"  + ("_simple" if self.simple else "") + ("_atend" if not begin else "") + "_newmethod"
      
        os.makedirs(self.file_root, exist_ok=True)
            
        # self.coco_image_path = "/home/lingpeng/project/demo/data/images/mscoco/images/train2014/"
        self.coco_image_path = "/nfs/demo/data/images/mscoco/images/train2014/"
        # self.saiapr_tc_12_image_path = "/home/lingpeng/project/demo/data/images/saiapr_tc-12/"
        self.saiapr_tc_12_image_path = "/nfs/demo/data/images/saiapr_tc-12/"
        # self.image_root = 'data/clevr_ref+_1.0/images/' + 'train' 
        # self.image_name = 'CLEVR_' + (self.split if self.split == 'train' else 'val') + '_%s.png'

        self.height, self.width = 512, 512
        # 因为coco中图片尺寸不一致，所以在运行时确定
        # self.scale_factor_x = self.width*1.0 / 480
        # self.scale_factor_y = self.height*1.0 / 320
        
        # self.max_len = 23 + 2
        self.max_len = {
            'refcoco': 24,
            'refcocog': 35,
            'refcoco+': 24,
            'refclef': 28
        }[self.ref_type] + 3

        self.build_index()
        self.build_vocab(vocab_file)

        self.use_bert = False
        if self.use_bert:
            self.max_len += 1
            self.tokenizer = BertTokenizer('/nfs/crefs/bert/bert-base-uncased-vocab.txt')

    
    def get_targets(self): #导入组合后的数据，注意的是，我们生成的数据为列表，列表的值为列表，长度为2/3或者4，都是原数据的ref_id
        gref_name = "ref_res_" + self.ref_type + "_" + self.ref_split
        grefs = json.load(open("/home/lingpeng/project/iep-ref-master/" + gref_name + '.json', 'r'))[self.ref_type]

        refs_useful = []

        for gref_idss in grefs:
            for gref_ids in gref_idss:

                for index, gref_id in enumerate(gref_ids):
                    if gref_id in refs_useful:
                        continue

                    refs_useful.append(gref_id)
        
        self.refs_useful = refs_useful

    def build_index(self):

        # self.get_targets()

        ref_path = os.path.join(self.ref_root, self.ref_type)
        ref_file = os.path.join(ref_path, "refs(" + self.ref_split + ").p")
        instances_file = os.path.join(ref_path, "instances.json")

        refs = pickle.load(open(ref_file, 'rb'))
        instances = json.load(open(instances_file, 'r'))

        data = {}
        data['refs'] = refs
        data['images'] = instances['images']
        data['annotations'] = instances['annotations']
        data['categories'] = instances['categories']

        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in data['images']:
            Imgs[img['id']] = img
        for cat in data['categories']:
            Cats[cat['id']] = cat['name']
        
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.imgToAnns = imgToAnns

        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in data['refs']:
            # ids
            ref_id = ref['ref_id']
            # if ref_id not in self.refs_useful:
            #     continue

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

        self.Refs = Refs
        self.imgToRefs = imgToRefs
        self.refToAnn = refToAnn

    def build_vocab(self, vocab_file):
        vocab2idx = {}
        idx2vocab = {}
        with open(vocab_file) as f:
            vocab = json.load(f)['refexp_token_to_idx']
        for k, v in vocab.items():
            vocab2idx[k] = int(v)
            idx2vocab[int(v)] = k

        self.word2idx = vocab2idx
        self.idx2word = idx2vocab

    def getMask(self, ann, image):
        # return mask, area and mask-center
        # ann = refToAnn[ref['ref_id']]
        # image = Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon  crowd = 0
            rle = MASK.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']
        # rle = bytes(rle).encode('utf-8')
        m = MASK.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(MASK.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}

    def get_image(self, image_id, height=-1, width=-1):
            
        # image_id = '{:0=6}'.format(int(image_id))
        image_name = self.Imgs['file_name']
        image_path = os.path.join(self.image_root, image_name)
        im = Image.open(image_path).convert('RGB')
        im = np.array(im)  # 0-225
        im_type = im.dtype
        # 此时的image的尺寸不一定是320 x 320，但是mask的尺寸已经固定了
        if height != -1 and width != -1:
            im = cv2.resize(im.astype(np.float32), (width, height)).astype(im_type)
        
        return im

    def get_sent_encode(self, tokens):

        refexp_encoded = []
        for token in tokens:
            refexp_encoded.append(self.word2idx[token])
        
        
        while len(refexp_encoded) < self.max_len:
            refexp_encoded.append(self.word2idx['<NULL>'])


        assert len(refexp_encoded) == self.max_len, len(refexp_encoded)

        return np.array(refexp_encoded, dtype=np.long)

    def from_encode2sentence(self, encode_sent):
        ori_sentence = decode(encode_sent, self.idx2word)

        return ori_sentence

    def saveMask(self, mask, save_dir):
        lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())

        lbl_pil.save(save_dir)

    def __call__(self, dataset_dict):

        # print(dataset_dict)
        # assert 1 == 0

        dataset_dicts = {}

        # print(dataset_dict)
        # print(dataset_dict.keys())
        # print(dataset_dict.values())
        
        # 只适用于batch size为1
        cref_id = list(dataset_dict.keys())[-1]
        gref_ids = list(dataset_dict.values())[-1]


        # assert 1 == 0

        image_id = self.Refs[gref_ids[-1]]['image_id']
        # assert int(image_id) == 458969

        image = self.Imgs[image_id]
        image_name = image['file_name']
        # print(image_name)
        image_file = os.path.join(self.coco_image_path, image_name) if self.ref_type != 'refclef' else os.path.join(self.saiapr_tc_12_image_path, image_name)
        # print(image_file)
        img = Image.open(image_file).convert('RGB')
        img = np.array(img)
        # print(img.shape)
        oheight, owidth = img.shape[: -1]
        img = cv2.resize(img.astype(np.float32), (self.width, self.height)).astype(img.dtype)
        

        image_bbox = img.copy()
        scale_factor_x, scale_factor_y = self.width*1.0 / owidth, self.height*1.0 / oheight
        bmask = np.zeros((self.height, self.width), dtype=np.uint8)
        smask = np.zeros((self.height, self.width))
        single_masks = []
        boxes = []
        classes = []
        sentence = None

        tokens = []
        # if self.use_bert:
            # tokens.append('[CLS]')
        
        for gref_id in gref_ids:
            gref = self.Refs[gref_id]
            gimage_id = gref['image_id']
            assert gimage_id == image_id

            gchoice = np.random.choice(len(gref['sentences']))
            # print(gref['sentences'][gchoice])
            gsent = gref['sentences'][gchoice]['raw']
            while '?' in gsent:
                gchoice = np.random.choice(len(gref['sentences']))
                gsent = gref['sentences'][gchoice]['raw']

            gtokens = gref['sentences'][gchoice]['tokens']

            gsent = gsent.lower()
            if sentence is None:
                sentence = gsent
            else:
                if self.use_bert:
                    sentence = sentence + ' [SEP] ' + gsent
                else:
                    sentence = sentence + '.' + gsent

            # if tokens is None:
            #     tokens = gtokens.copy()
                
            # else:
            #     tokens.append('.')
            for gtoken in gtokens:
                tokens.append(gtoken)
            
            tokens.append('.')
        
            
            ann = self.refToAnn[gref_id]
            if self.ref_type != 'refclef':
                assert ann['iscrowd'] == 0

            ocategory_id = gref['category_id']
            category_name = self.Cats[ocategory_id]
            # category_id = CATEGORY[category_id]
            category_id = CATEGORY_refcoco[ocategory_id]

            # assert category_id <= 75
            assert category_id <= 77
            classes.append(category_id)

            mask_area = self.getMask(ann, image)
            mask, area = mask_area['mask'], mask_area['area']

            mask = cv2.resize(mask.astype(np.float32), (self.width, self.height)).astype(np.uint8)
            mask[mask > 1] = 0
            assert mask.sum() > 0
            bmask = np.where((bmask == 0) & (mask != 0), bmask + mask, bmask)
            smask = np.where((smask == 0) & (mask != 0), smask + mask * (category_id if self.split=='train' else ocategory_id), smask)
            single_masks.append(mask)
            box = ann['bbox']
            x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]

            x0 *= scale_factor_x
            x1 *= scale_factor_x
            y0 *= scale_factor_y
            y1 *= scale_factor_y
            boxes.append([x0, y0, x1, y1])

            color = [
                        int(1.5*category_id if self.split=='train' else ocategory_id), 
                        int(0.5*category_id if self.split=='train' else ocategory_id), 
                        int(4.5*category_id if self.split=='train' else ocategory_id)]

            image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
            text_category = str(category_id if self.split=='train' else ocategory_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            image_bbox = cv2.putText(image_bbox, text_category, (int(x1)-40, int(y1)-10), font, 0.5, color, 2)
        
        if self.use_bert:
            sentence = sentence + ' [SEP]'
        # sentence = sentence.lower()
        # sentence = "man with yellow helmet. man with blue helmet."
        # tokens = ["man", "with", "yellow", "helmet", ".", "man", "with", "blue", "helmet", "."]
        # sentence = "tv screen on left."
        # tokens = ["tv", "screen","on", "left", "."]
        # print(sentence)
        # print(tokens)
        # print(len(tokens))
        
        if self.use_bert:
            # print(sentence)
            tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence)
            while len(tokens) < self.max_len:
                tokens = tokens + ['[PAD]']
            
            if len(tokens) > self.max_len:
                tokens = tokens[: self.max_len]

            sent_encode = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            sent_encode = self.get_sent_encode(tokens)
        # print(self.tokenizer.convert_ids_to_tokens(sent_encode))
        # print(self.from_encode2sentence(sent_encode))
        dataset_dicts['sent_encode'] = torch.as_tensor(sent_encode, dtype=torch.long)
        
        
        # assert 1 == 0
        

        # plt.imshow(image_bbox)
        # plt.show()

        # assert 1 == 0
 
        if self.split != 'train':
            
            gt_sentence_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_sentence.txt")
            gt_sem_mask_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_semantic_mask.png")
            gt_image_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_image_withbbox.png")

            with open(gt_sentence_file, 'w') as f:
                f.write(str(image_file) + ": " + str(sentence))

            self.saveMask(smask, gt_sem_mask_file)
            Image.fromarray(image_bbox.astype(np.uint8)).save(gt_image_file)


        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height
        dataset_dicts['image_id'] = image_id
        dataset_dicts['cref_id'] = cref_id  #唯一识别标志
        dataset_dicts['file_name'] = image_file

        # sem_seg_gt = smask.astype(np.long)
        sem_seg_gt = None   # 暂时不需要
        
        boxes = np.array(boxes)
        # print('The unm_bbox of the display image is:', len(boxes)) 

        # print(boxes)
        # print(classes)

        image_shape = img.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dicts["image"] = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        )

        if sem_seg_gt is not None:
            dataset_dicts["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))


        if not self.is_train:
            dataset_dicts.pop("obj_bbox", None)
            dataset_dicts.pop('obj_mask', None)
            dataset_dicts.pop("pano_seg_file_name", None)

            # return dataset_dicts
        
        # 最后一步：创建instance
        target = Instances(image_shape)
        target.gt_boxes = Boxes(list(boxes))
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in single_masks])
            )
        target.gt_masks = masks
        # print("#" * 20)
        # print("查看instance信息")
        # print(target)

        instances = target
        dataset_dicts["instances"] = utils.filter_empty_instances(instances)
        # print("输出输入的数据情况")
        # print(dataset_dicts.keys())
        return dataset_dicts
