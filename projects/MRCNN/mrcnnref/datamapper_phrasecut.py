from numpy.lib.shape_base import split
import scipy.io
import scipy.ndimage
import os
import os.path as osp
import json
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode
from PIL import Image
import cv2
import numpy as np
from .utils.comm import encode, decode, display_instances, tokenize
import imgviz
import torch
import matplotlib.pyplot as plt
from typing import Any, Iterator, List, Union
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
import pycocotools.mask as mask_util
from .phrasecut.refvg_loader import RefVGLoader
# from pytorch_pretrained_bert.tokenization import BertTokenizer
import imgviz
from PIL import Image
import os
from .phrasecut.visualize_utils import plot_refvg, plot_refvg_v2, visualize_colors
from .phrasecut.data_transfer import polygons_to_mask, xywh_to_xyxy

from .phrasecut.file_paths import img_fpath
from matplotlib.patches import Rectangle, Polygon

from pycocotools import mask as MASK

PCATE = {}  # 映射
for i in range(1, 1273):
    PCATE[i] = i - 1

ICATE_PHRASE = {}  #映射回初始类id
for oi, ni in PCATE.items():
    ICATE_PHRASE[ni] = oi




class PhraseCutDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        print("for phrasecut")
        # Rebuild augmentations
        

        self.split = 'train' if is_train else 'test'
        # self.split = 'train'
        self.is_train = is_train

        print("训练模式") if is_train else print("测试模式")


        if cfg.MODEL.NUM_CLASSES == 46:
            self.data_name = 'sketch'

        elif cfg.MODEL.NUM_CLASSES == 1272:
            self.data_name = 'phrasecut'

        elif cfg.MODEL.NUM_CLASSES == 48:
            self.data_name = 'iepref'
        
        else:
            raise

        # self.refvg_loader = RefVGLoader(split=self.split)
        self.simple = False
        easy = True
        begin = True
        use_attn = True
        use_iter = False
        self.file_root = "/home/lingpeng/project/Adet/inference_dir_phrasecut/mrcnn_R_101_1x_refcoco_unc_v4_pick_0.8"  \
                                                        +  ("_simple" if self.simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_simple" if use_attn else "")  \
                                                        + ("_use_1iter_withmattn_newversion" if use_iter else "") + '_' + self.data_name
        os.makedirs(self.file_root, exist_ok=True)
          
        # vocab_file = os.path.join(self.ref_root, 'vocab.json')
        # self.height, self.width = 640, 640
        # self.height, self.width = (512, 512) if self.is_train else (786, 786)
        self.height, self.width = (512, 512)
        # self.height, self.width = (768, 768)
    
        self.max_len = 32

        self.use_bert = cfg.MODEL.USE_BERT
        
        self.use_roberta_base = cfg.MODEL.USE_ROBERTA

        self.use_clip = cfg.MODEL.USE_CLIP
        if self.use_bert:
            print("使用bert")
            # self.max_len += 2
            # self.tokenizer = BertTokenizer('/nfs/crefs/bert/bert-base-uncased-vocab.txt') 
        if self.use_clip:
            print("使用CLIP")
            
        if self.use_roberta_base:
            pass
            assert not self.use_bert
            print("使用roberta")
            # self.max_len += 2   # 前后各加上一个pad，总长度为4，
            # self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            
        vocab_file = "/nfs/crefs/dict/phrasecut/dict.json"
        self.build_vocab(vocab_file)
    
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

    def get_sent_encode(self, tokens):

        refexp_encoded = []
        for token in tokens:
            # if token == '' or token == ' ' or token == '  ':
            #     continue
            refexp_encoded.append(self.word2idx[token])

        while len(refexp_encoded) < self.max_len:
            refexp_encoded.append(self.word2idx['<NULL>'])


        assert len(refexp_encoded) == self.max_len, len(refexp_encoded)

        return np.array(refexp_encoded, dtype=np.long)

    def from_encode2sentence(self, encode_sent):

        if self.use_bert:
            ori_sentence = self.tokenizer.convert_ids_to_tokens(encode_sent)
        else:
            ori_sentence = decode(encode_sent, self.idx2word)

        token = []
        for x in ori_sentence:
            if x == '<NULL>' or x == '[PAD]':
                break
            token.append(x)

        return token

    def saveMask(self, mask, save_dir):
        lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())

        lbl_pil.save(save_dir)

    def showGT(self, img_ref_data):
        img_id = img_ref_data['image_id']
        fig_size = (self.height/10, self.width/10)

        # print(len(img_ref_data['task_ids']))
        task_id = img_ref_data['task_id']
        phrase = img_ref_data['phrase']
        gt_boxes = img_ref_data['gt_boxes']
        gt_Polygons = img_ref_data['gt_Polygons']
        plot_refvg(fig_size=fig_size, font_size=12, img_id=img_id, title=phrase,
                gt_Polygons=gt_Polygons, gt_boxes=gt_boxes, gray_img=False, save_path="/home/lingpeng/project/SparseR-CNN-main/t.png")

        # fig.show()

        # assert 1 == 0

    def __call__(self, data_dict):
        dataset_dicts = {}

        img_id = data_dict['image_id']
        cref_id = data_dict['task_id']

        phrase = data_dict['phrase']
        gt_boxes = data_dict['gt_boxes']

        # gt_boxes = xywh_to_xyxy(gt_boxes)
        gt_Polygons = data_dict['gt_Polygons']
        class_id = data_dict["class_ids"]  # 该数据集中必然同类instance
        class_name = data_dict['class_name']

        oimg = Image.open(os.path.join(img_fpath, '%d.jpg' % img_id)).convert('RGB')
        # print(oimg.size)
        img = np.array(oimg)
        # print(img.shape)
        ori_h, ori_w = img.shape[: -1]
        if self.width != ori_w or self.height != ori_h:
            img = cv2.resize(img.astype(np.float32), (self.width, self.height)).astype(img.dtype)
        # print(img.shape)
        # bmasks = []
        single_masks = []
        boxes = []
        classes = []
        scale_factor_x, scale_factor_y = self.width*1.0 / ori_w, self.height*1.0 / ori_h
        bmask = np.zeros((self.height, self.width), dtype=np.uint8)
        smask = np.zeros((self.height, self.width))

        def modify_color(d):
            colors = visualize_colors.copy()
            if d is None:
                return colors
            for name, color in d.items():
                colors[name] = color
            return colors

        colors = modify_color(None)

        if not self.is_train:
            image_bbox = img.copy()

        color = colors['gt_polygons']

        # print(len(gt_Polygons), len(gt_boxes))
        category_id = PCATE[class_id]
        for p, box in zip(gt_Polygons, gt_boxes):
            # print("category: ", class_id)
            # print(type(p))  # <class 'list'>
            # print(p)
            # po = Polygon(p, fill=True, alpha=0.5, color=color)
            
            classes.append(category_id)

            x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
            x0 *= scale_factor_x
            x1 *= scale_factor_x
            y0 *= scale_factor_y
            y1 *= scale_factor_y
            boxes.append([x0, y0, x1, y1])

            color = [
                    int(1.5*category_id if self.is_train else class_id), 
                    int(0.5*category_id if self.is_train else class_id), 
                    int(4.5*category_id if self.is_train else class_id)
                ]

            if self.is_train:
                image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)

                # without class_name
                # text_category = str(category_id if self.is_train else class_id)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # image_bbox = cv2.putText(image_bbox, text_category, (int(x1)-40, int(y1)-10), font, 0.5, color, 2)


            mask = polygons_to_mask(p, ori_w, ori_h) + 0  # 0/1 narray
            # return {'mask': m, 'area': area}
            # if ori_w != self.width or self.ori_h != self.height:
            mask = cv2.resize(mask.astype(np.float32), (self.width, self.height)).astype(mask.dtype)
            # mask[mask > 1] = 0  # 也可以设为1
            assert mask.all() <= 1

            # if not (mask.sum() > 0):
            #     plt.title(phrase)
            #     print(img_id, len(gt_boxes))
                # plt.imshow(img)
                # plt.show()

                # plt.imshow(image_bbox)
                # plt.show()

                # plt.imshow(mask)
                # plt.show()

                # assert 1 == 0, str(img_id) + str(cref_id) + str(class_name)
            # assert mask.sum() > 0, str(img_id) + str(class_name) + str(box)
            bmask = np.where((bmask == 0) & (mask != 0), bmask + mask, bmask)

            smask = np.where((smask == 0) & (mask != 0), smask + mask * (category_id if self.is_train else class_id), smask)
            single_masks.append(mask)


            

        plt.title(phrase)
        plt.imshow(image_bbox)
        plt.show()

        plt.imshow(bmask)
        plt.show()
        plt.close()
        assert 1 == 0

        sentence = phrase.lower()
        sentence = ' '.join(sentence.split())
        # print(sentence)
        tokens = sentence.split(" ")
        # print(tokens)
        # sentence = "the purple cylinder in the right front of the brown sphere"
        if not self.is_train:
            print(sentence)
        
        sent_encode = self.get_sent_encode(tokens)
        dataset_dicts['sent_encode'] = torch.as_tensor(sent_encode, dtype=torch.long)

        if not self.is_train: # 自然图像可以可视化

            file_root_per_image = os.path.join(self.file_root, str(img_id))
            os.makedirs(file_root_per_image, exist_ok=True)

            # gt_inst_file = osp.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_gt_image_inst.png")


            gt_sentence_file = osp.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_gt_sentence.txt")
            gt_sem_mask_file = osp.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_gt_semantic_mask.png")
            gt_image_file = osp.join(file_root_per_image, str(img_id) + '_' + str(cref_id) + "_gt_image_withbbox.png")

            with open(gt_sentence_file, 'w') as f:
                f.write(str(sentence))

            self.saveMask(smask, gt_sem_mask_file)
            Image.fromarray(image_bbox.astype(np.uint8)).save(gt_image_file)

            # assert 1 == 0

        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height
        dataset_dicts['image_id'] = img_id
        dataset_dicts['cref_id'] = cref_id  #唯一识别标志
        dataset_dicts['raw_sentence'] = sentence
        
        sem_seg_gt = None   # 暂时不需要
        
        boxes = np.array(boxes)
        image_shape = img.shape[:-1]  # h, w
        dataset_dicts["image"] = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        )

        if sem_seg_gt is not None:
            dataset_dicts["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

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


        # assert 1 == 0
        return dataset_dicts

    