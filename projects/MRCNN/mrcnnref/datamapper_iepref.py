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


class IEPDatasetMapperWithBasis(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # self.augmentation = build_augmentation(cfg, is_train)
        self.augmentation = None # 不使用数据增强
        print("不使用数据增强") if self.augmentation is None else print("使用数据增强")

        self.split = ('train' if is_train else 'val')
        # self.split = 'train'
        self.training = is_train
        # self.training = False

        print("训练模式") if is_train else print("测试模式")

        if cfg.MODEL.NUM_CLASSES == 46:
            self.data_name = 'sketch'

        elif cfg.MODEL.NUM_CLASSES == 1272:
            self.data_name = 'phrasecut'

        elif cfg.MODEL.NUM_CLASSES == 48:
            self.data_name = 'iepref'
        
        else:
            raise
        
        self.simple = False
        easy = True
        begin = True
        use_attn = True
        use_iter = False
        self.file_root = "/home/lingpeng/project/Adet/inference_dir/mrcnn_R_101_1x_refcoco_unc_v4_pick_0.8"  \
                                                        +  ("_simple" if self.simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_simple" if use_attn else "")  \
                                                        + ("_use_1iter_withmattn_newversion" if use_iter else "") + '_' + self.data_name
        os.makedirs(self.file_root, exist_ok=True)

        # self.ref_root = '/nfs/iep-ref-master/data'
        self.ref_root = '/home/lingpeng/project/iep-ref-master/data3'

        self.scene_file = osp.join(self.ref_root, 'clevr_ref+_1.0/scenes/clevr_ref+_' + (self.split if self.split == 'train' else 'val') + '_scenes.json')
        self.vocab_file = osp.join(self.ref_root, "referring_rubber/picked_vocab_" + (self.split if self.split == 'train' else 'val') + ".json")

        self.image_root = osp.join(self.ref_root, 'clevr_ref+_1.0/images/' + (self.split if self.split == 'train' else 'val'))
        # self.image_root = 'data/clevr_ref+_1.0/images/' + 'train' 
        self.image_name = 'CLEVR_' + (self.split if self.split == 'train' else 'val') + '_%s.png'

        self.height, self.width = (320, 480)
        # self.height, self.width = (128, 128) if self.is_train else (512, 512)
        self.scale_factor_x = self.width*1.0 / 480
        self.scale_factor_y = self.height*1.0 / 320
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.224])

        self.max_len = 14 + 2

        self.bad_ref_ids = []
        self.build_vocab(self.vocab_file)
    
    def get_image(self, image_id, height=-1, width=-1):
            
        image_id = '{:0=6}'.format(int(image_id))
        image_path = os.path.join(self.image_root, self.image_name % (image_id))
        im = Image.open(image_path).convert('RGB')
        im = np.array(im)  # 0-225
        im_type = im.dtype
        # 此时的image的尺寸不一定是320 x 320，但是mask的尺寸已经固定了
        if height != -1 and width != -1:
            im = cv2.resize(im.astype(np.float32), (width, height)).astype(im_type)
        
        return im

    def from_imgdensestr_to_imgarray(self, imgstr):
            img = []
            cur = 0
            for num in imgstr.split(','):
                num = int(num)
                img += [cur] * num
                cur = 1-cur
            img = np.asarray(img).reshape((320,480))
            return img

    def saveMask(self, mask, save_dir):
        lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())

        lbl_pil.save(save_dir)

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

    def from_encode2sentence(self, encode_sent):
        ori_sentence = decode(encode_sent, self.idx2word)

        return ori_sentence

    def get_sent_encode(self, sentence):

        refexp_tokens = tokenize(sentence,
                        punct_to_keep=[';', ','],
                        punct_to_remove=['?', '.'])

        refexp_encoded = encode(refexp_tokens,
                         self.word2idx,
                         allow_unk=False)
        
        while len(refexp_encoded) < self.max_len:
            refexp_encoded.append(self.word2idx['<NULL>'])

        assert len(refexp_encoded) == self.max_len

        return np.array(refexp_encoded)


    def __call__(self, dataset_dict):  #输入的就是一个一个的json数据
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_dicts = {}

        image_id = dataset_dict['image_index']
        # assert image_id == 666  # 自己输入测试用



        cref_id = dataset_dict['cref_id']

        image_name = dataset_dict['image_filename']
        image_file = osp.join(self.image_root, image_name)


        img = Image.open(image_file).convert('RGB')
        img = np.array(img)
        oheight, owidth = img.shape[: -1]
        img = cv2.resize(img.astype(np.float32), (self.width, self.height)).astype(img.dtype)

        sentence = dataset_dict['refexp']

        # sentence = "the purple cylinder in the right front of the brown sphere"
        if not self.training:
            print(sentence)

        
        sentence = sentence.lower()
        sent_encode = self.get_sent_encode(sentence)
        dataset_dicts['sent_encode'] = torch.as_tensor(sent_encode, dtype=torch.long)

        image_bbox = img.copy()
        # scale_factor_x, scale_factor_y = self.scale_factor_x, self.scale_factor_y
        bmask = np.zeros((self.height, self.width), dtype=np.uint8)
        smask = np.zeros((self.height, self.width))
        single_masks = []
        boxes = []
        classes = []



        for index, objid in enumerate(dataset_dict['objlist']):
            obj_mask = dataset_dict['obj_mask'][str(objid)]

            mask_img = self.from_imgdensestr_to_imgarray(obj_mask)
            mask_img = cv2.resize(mask_img.astype(np.float32), (self.width, self.height))
            mask_img[mask_img > 0] = 1
            mask_img = mask_img.astype(np.uint8)
            # assert not np.all(mask_img == 0), 'problem refexp  index ' + str(refexp["refexp_index"])
            assert mask_img.sum() != 0, 'problem scene  index ' + str(dataset_dict['image_index'])

            bmask = np.where(bmask == 0 & (mask_img != 0), bmask + mask_img, bmask)

            category_id = dataset_dict['category_id'][int(objid)]
            tcategory_id = category_id + 1
            smask = np.where((smask == 0) & (mask_img != 0) , smask + mask_img * tcategory_id, smask)

            single_masks.append(mask_img)
            classes.append(category_id)
            # bbox
            box = dataset_dict['obj_bbox'][str(objid)]
            x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]

            x0 *= self.scale_factor_x
            x1 *= self.scale_factor_x
            y0 *= self.scale_factor_y
            y1 *= self.scale_factor_y

            boxes.append([x0, y0, x1, y1])
            image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
        
        if not self.training:
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


        if not self.training:
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


