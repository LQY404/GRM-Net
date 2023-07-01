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
from transformers import RobertaTokenizerFast, BertTokenizer
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



CATE = {}  # 映射
for i in range(1, 79):
    CATE[i] = i - 1

ICATE_SKETCH = {}  #映射回初始类id
for oi, ni in CATE.items():
    ICATE_SKETCH[ni] = oi

# from pytorch_pretrained_bert.tokenization import BertTokenizer
from .utils.sketch_visualizer import visualize_sem_inst_mask

# sketchscene
SKETCH_CLASS_NAME = {
    0: 'background',
    1: "airplane",
    2: "apple",
    3: "balloon",
    4: "banana",
    5: "basket",
    6: "bee",
    7: "bench",
    8: "bicycle",
    9: "bird", 
    10: "bottle",
    11: "bucket",
    12: "bus",
    13: "butterfly",
    14: "car",
    15: "cat",
    16: "chair",
    17: "chicken",
    18: "cloud",
    19: "cow", 
    20: "cup",
    21: "dinnerware",
    22: "dog",
    23: "duck",
    24: "fence",
    25: "flower",
    26: "grape",
    27: "grass",
    28: "horse",
    29: "house",
    30: "moon",
    31: "mountain",
    32: "people",
    33: "picnic rug",
    34: "pig",
    35: "rabbit", 
    36: "road",
    37: "sheep",
    38: "sofa",
    39: "star",
    40: "street lamp",
    41: "sun",
    42: "table",
    43: "tree",
    44: "truck",
    45: "umbrella",
    46: "others"
}

SKETCH_CLASS_NAME_ = {
    0: 'background',
    1: "bench",
    2: "bird", 
    3: "bus",
    4: "butterfly",
    5: "car",
    6: "cat",
    7: "chick",
    8: "cloud",
    9: "cow",
    10: "dog",
    11: "duck",
    12: "grass",
    13: "horse",
    14: "house",
    15: "moon",
    16: "person",
    17: "pig",
    18: "rabbit",
    19: "road",
    20: "sheep",
    21: "star",
    22: "sun",
    23: "tree",
    24: "truck"
}

class SketchDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        print("for sketch")
        # Rebuild augmentations
        

        self.split = 'train' if is_train else 'test'
        # self.split = 'train'
        # self.split = 'val'
        # self.split = 'train'
        self.is_train = is_train

        print("训练模式") if is_train else print("测试模式")

        self.data_name = 'sketch' 
        '''
        if cfg.MODEL.NUM_CLASSES == 46:
            self.data_name = 'sketch'

        elif cfg.MODEL.NUM_CLASSES == 1272:
            self.data_name = 'phrasecut'

        elif cfg.MODEL.NUM_CLASSES == 48:
            self.data_name = 'iepref'
        
        else:
            raise
        '''
        # self.ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/" # share
        self.ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax

        # self.image_root = "/nfs/SketchyScene-pytorch/data/" + self.split + "/DRAWING_GT"
        self.image_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + self.split + "/DRAWING_GT"


        # self.semantic_root = "/nfs/SketchyScene-pytorch/data/" + self.split + "/CLASS_GT"
        self.semantic_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + self.split + "/CLASS_GT"
        # self.instance_root = "/nfs/SketchyScene-pytorch/data/" + self.split + "/INSTANCE_GT"
        self.instance_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + self.split + "/INSTANCE_GT"

        self.simple = False
        easy = True
        begin = True
        use_attn = True
        use_iter = False
        self.file_root = "/home/lingpeng/project/Adet_2022/effe_inference_dir_new/mrcnn_R_101_1x_refcoco_unc_v4_pick_0.8"  \
                                                        +  ("_simple" if self.simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_simple" if use_attn else "")  \
                                                        + ("_use_1iter_withmattn_newversion" if use_iter else "") + '_' + self.data_name \
                                                        + "_onenewgraph"
        os.makedirs(self.file_root, exist_ok=True)
          
        vocab_file = os.path.join(self.ref_root, 'vocab.json')
        # self.height, self.width = 640, 640
        # self.height, self.width = (512, 512) if self.is_train else (786, 786)
        self.height, self.width = (512, 512)
        # self.height, self.width = (786, 786)
        # self.height, self.width = (768, 768)
        # self.height, self.width = (770, 770)
    
        self.max_len = 15

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
            
        self.build_vocab(vocab_file)
        
        self.inference_root = "/home/lingpeng/project/SparseR-CNN-main/prepare_for_teaser"
    
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

    def get_gt_new(self, image_id, selected_instance_ids, height=-1, width=-1):
        if not self.is_train:
            print("get gt from image: ", image_id)

        
        image_name = "L0_sample" + str(image_id) + '.png'
        semantic_name = "sample_" + str(image_id) + "_class"
        instance_name = "sample_" + str(image_id) + "_instance"

        image_path = os.path.join(self.image_root, image_name)
        # image_path = os.path.join(self.inference_root, image_name)
        
        semantic_path = os.path.join(self.semantic_root, semantic_name)
        instance_path = os.path.join(self.instance_root, instance_name)

        # 载入图像
        # print(image_path)
        # print(self.split)
        im = Image.open(image_path).convert('RGB')
        im = np.array(im)  # 0-225
        im_type = im.dtype
        ori_h, ori_w = im.shape[: -1]

        oim = None
        if height != ori_h or width != ori_w:
            assert height != -1 and width != -1
            if not self.is_train:
                oim = cv2.resize(im.astype(np.float32), (768, 768)).astype(im_type)

            im = cv2.resize(im.astype(np.float32), (width, height)).astype(im_type)
        
        if selected_instance_ids is None:
            print("for inference")
            return oim, im, [], None, [], None

        assert type(selected_instance_ids) is list
        selected_instance_ids_ = [item for item in selected_instance_ids]

        # 载入mask
        INSTANCE_GT = scipy.io.loadmat(instance_path)['INSTANCE_GT']
        INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
        CLASS_GT = scipy.io.loadmat(semantic_path)['CLASS_GT']  # (750, 750)
        instance_count = np.bincount(INSTANCE_GT.flatten())  # 图中所有的实例 [inst_id: inst_pixel]


        instance_count = instance_count[1:]  # e.g. shape=(101,)
        # print(instance_count)


        bmasks = []
        classes = []

        smask = None

        real_instanceIdx = 0
        for i in range(instance_count.shape[0]):
            if instance_count[i] == 0:
                continue
            
            instanceIdx = i + 1

            if real_instanceIdx in selected_instance_ids:
                selected_instance_ids_.remove(real_instanceIdx)
                mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
                mask[INSTANCE_GT == instanceIdx] = 1

                assert np.sum(mask) != 0

                class_gt_filtered = CLASS_GT * mask
                class_gt_filtered = np.bincount(class_gt_filtered.flatten())
                class_gt_filtered = class_gt_filtered[1:]
                class_id = np.argmax(class_gt_filtered) + 1

                nclass_id = CATE[class_id]
                classes.append(nclass_id)


                if height != -1 and width != -1:
                    if self.is_train:
                        mask = cv2.resize(mask.astype(np.float32), (width, height))
                    else:
                        mask = cv2.resize(mask.astype(np.float32), (768, 768))


                if smask is None:
                    smask = mask * class_id
                else:
                    smask = np.where((smask == 0) & (mask > 0), smask+mask * class_id, smask)


                bmasks.append(mask)

            real_instanceIdx += 1
        
        # 提取box
        boxes = []
        for m in bmasks:
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
                # print("some problems happened during getting bbox")
                assert 1 == 0
            # boxes.append(np.array([y1, x1, y2, x2]))
            boxes.append(np.array([x1, y1, x2, y2]))


        return oim, im, bmasks, smask, classes, boxes

    def get_gt(self, image_id, selected_instance_ids, height=-1, width=-1):
        if not self.is_train:
            print("get gt from image: ", image_id)

        assert type(selected_instance_ids) is list
        selected_instance_ids_ = [item for item in selected_instance_ids]

        image_name = "L0_sample" + str(image_id) + '.png'
        semantic_name = "sample_" + str(image_id) + "_class"
        instance_name = "sample_" + str(image_id) + "_instance"

        image_path = os.path.join(self.image_root, image_name)
        semantic_path = os.path.join(self.semantic_root, semantic_name)
        instance_path = os.path.join(self.instance_root, instance_name)

        # 载入图像
        print(image_path)
        im = Image.open(image_path).convert('RGB')
        im = np.array(im)  # 0-225
        im_type = im.dtype
        if height != -1 and width != -1:
            im = cv2.resize(im.astype(np.float32), (width, height)).astype(im_type)
        

        # 载入mask
        INSTANCE_GT = scipy.io.loadmat(instance_path)['INSTANCE_GT']
        INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
        CLASS_GT = scipy.io.loadmat(semantic_path)['CLASS_GT']  # (750, 750)

        instance_count = np.bincount(INSTANCE_GT.flatten())  # 图中所有的实例 [inst_id: inst_pixel]

        instance_count = instance_count[1:]  # e.g. shape=(101,)
        # print(instance_count)
        nonzero_count = np.count_nonzero(instance_count)

        selected_mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.int32)
        mask_set = np.zeros([len(selected_instance_ids), INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
        # mask_set = np.zeros([nonzero_count, INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.int32)
        # class_id_set = np.zeros([nonzero_count], dtype=np.int32)
        class_id_set = np.zeros([len(selected_instance_ids)], dtype=np.int32)


        real_instanceIdx = 0
        # print("遍历")
        for i in range(instance_count.shape[0]):
            if instance_count[i] == 0:
                continue
            
            # print("get mask")
            instanceIdx = i + 1

            if real_instanceIdx in selected_instance_ids:
                selected_mask[INSTANCE_GT == instanceIdx] = 1
                selected_instance_ids_.remove(real_instanceIdx)

                mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
                mask[INSTANCE_GT == instanceIdx] = 1

                assert np.sum(mask) != 0
                mask_set[real_instanceIdx] = mask

                class_gt_filtered = CLASS_GT * mask
                class_gt_filtered = np.bincount(class_gt_filtered.flatten())
                class_gt_filtered = class_gt_filtered[1:]
                class_id = np.argmax(class_gt_filtered) + 1
                class_id_set[real_instanceIdx] = class_id

                if len(selected_instance_ids_) == 0:
                    # print('done')
                    break
            
            real_instanceIdx = real_instanceIdx + 1

        assert np.sum(selected_mask) != 0

        # mask_set = np.array(mask_set, dtype=np.uint8)
        # class_id_set = np.array(class_id_set, dtype=np.uint8)

        mask_set = np.transpose(mask_set, (1, 2, 0))  # [H, W, nInst]
        if mask_set.shape[0] != height:
            scale = height / mask_set.shape[0]
            mask_set = scipy.ndimage.zoom(mask_set, zoom=[scale, scale, 1], order=0)
            mask_set = np.array(mask_set, dtype=np.uint8)

            selected_mask = scipy.ndimage.zoom(selected_mask, zoom=[scale, scale], order=0)
            selected_mask = np.array(selected_mask, dtype=np.int32)

        bboxes = self.extract_bboxes(mask_set)  # [nInst, (y1, x1, y2, x2)]
        # print("return gt")
        mask_set = np.transpose(mask_set, (2, 0, 1))

        assert mask_set.shape[0] == bboxes.shape[0]
        return im, selected_mask, mask_set, class_id_set, bboxes

    def extract_bboxes(self, mask):
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
                # print("some problems happened during getting bbox")
            boxes[i] = np.array([y1, x1, y2, x2])

        return boxes.astype(np.int32)

    def get_sent_encode(self, tokens):

        refexp_encoded = []
        for token in tokens:
            if token not in self.word2idx:
                refexp_encoded.append(self.word2idx['<UNK>'])
            else: 
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

    def __call_old(self, data_dict):
        # print(data_dict)
        dataset_dicts = {}
        image_id = data_dict['image_id']
        cref_id = data_dict['ref_id']
        selected_instance_ids = data_dict['inst_ids'] if 'inst_ids' in data_dict else None

        # print("get gt")
        # image, selected_mask, masks, class_list, bboxes = self.get_gt(image_id, selected_instance_ids, self.height, self.width)
        oimage, image, selected_masks, smask, class_list, bboxes= self.get_gt_new(image_id, selected_instance_ids, self.height, self.width)


        # print("add bbox to image")
        
        # if not self.is_train:
        #     image_bbox = image.copy()
        #     for bbox, category_id in zip(bboxes, class_list):

                # print("add bbox to image")
                # print(bbox)

                # x0, y0, x1, y1 = bbox[1], bbox[0], bbox[3], bbox[2]
        #        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

        #        ocate = ICATE_SKETCH[category_id]
                # color = [int(1.5*ocate), int(0.5*ocate), int(4.5*ocate)]
        #        color = [0, 255, 255]

        #        image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
        #        text_category = str(ocate)
        #        font = cv2.FONT_HERSHEY_SIMPLEX
        #        image_bbox = cv2.putText(image_bbox, text_category, (int(x1)-40, int(y1)-10), font, 0.5, (0, 255, 255), 2)
        
        # print(data_dict['raw_sent'])
        # plt.imshow(image_bbox)
        # plt.show()


        # assert 1 == 0

        sentence = data_dict['raw_sent']
        # sentence = "the three sheep behind of the bus"
        # sentence = "the two sheep on the left back of the tree"
        # sentence = "the two sheep on the left front of the houses"
        # sentence = "the two trees on the right of the dogs"
        # sentence = "the two ducks on the right"
        # sentence = "the left two ducks"
        # sentence = "the house and the person"
        # sentence = "the leftmost tree"
        # sentence = "the rightmost butterfly"
        # sentence = "the top cat"
        # sentence = "the leftmost cow"
        # sentence = "the two people on the left"
        # sentence = 'the horse on the rightmost'
        # sentence = "the two chickens on the right"
        # sentence = "the horse on the right"
        # sentence = "the two trees on the left of the house"
    
        if self.is_train is False:    
            print(sentence)

        sentence = sentence.lower()
        tokens = sentence.split(" ")
        up_tokens = []
        for token in tokens:
            if token == '' or token == ' ':
                continue
            up_tokens.append(token)

        # print(len(tokens), tokens)
        sent_encode = self.get_sent_encode(up_tokens)
        # print(sent_encode.shape)
        # print(self.from_encode2sentence(sent_encode))
        # assert 1 == 0

        if self.use_bert:
            # sentence += " [SEP]"
            tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']
            # tokens = self.tokenizer.tokenize(sentence)
            # print(tokens)
            mask_attention = [1 for _ in range(len(tokens))]
            while len(tokens) < self.max_len:
                tokens = tokens + ['[PAD]']
                mask_attention.append(0)

            # print(tokens)

            assert len(tokens) == self.max_len

            sent_encode = self.tokenizer.convert_tokens_to_ids(tokens)
            mask_attention = np.array(mask_attention)
            # print(sent_encode)

            # assert 1 == 0
            sent_encode = np.array(sent_encode)
        # print(self.tokenizer.convert_ids_to_tokens(sent_encode))
        # print(self.from_encode2sentence(sent_encode))
        # assert 1 == 0
        # print(sent_encode.shape)
        # assert 1 == 0
        if not self.use_roberta_base:
            dataset_dicts['sent_encode'] = torch.as_tensor(sent_encode, dtype=torch.long)
            
        if self.use_bert:
            dataset_dicts['mask_attention'] = torch.as_tensor(mask_attention, dtype=torch.long)
        
        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height
        dataset_dicts['image_id'] = image_id
        dataset_dicts['cref_id'] = cref_id  #唯一识别标志
        dataset_dicts['file_name'] = "L0_sample" + str(image_id) + '.png'
        
        dataset_dicts['raw_sentence'] = sentence

        sem_seg_gt = None   # 暂时不需要
        boxes = np.array(bboxes)
        # print(image.shape)
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dicts["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if not self.is_train:
            dataset_dicts["oimage"] = torch.as_tensor(
                np.ascontiguousarray(oimage.transpose(2, 0, 1))
            )

        
        # print(dataset_dicts["image"].shape)
        # print(dataset_dicts["image"])
        # assert 1 == 0
        # image_bin = np.zeros([self.height, self.width], dtype=np.long)
        # image_bin[dataset_dicts["image"][0, :, :] == 0] = 1

        # dataset_dicts['sketch_bin'] = torch.as_tensor(image_bin, dtype=torch.long)

        if not self.is_train and 1==0:
            print("===========================")
            print("保存GT")
            print("===========================")
            # plt.imshow(image_bbox)
            # plt.show()
            #gt_sentence_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_sentence.txt")
            #gt_sem_mask_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_semantic_mask.png")
            #gt_image_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_image_withbbox.png")

            file_root_per_image = os.path.join(self.file_root, str(image_id))
            os.makedirs(file_root_per_image, exist_ok=True)

            gt_inst_file = osp.join(file_root_per_image, str(image_id) + '_' + str(cref_id) + "_gt_image_inst.png")

            # with open(gt_sentence_file, 'w') as f:
            #     f.write(str(image_id) + '_' + str(cref_id) + ":\n" + str(sentence))

            # self.saveMask(smask, gt_sem_mask_file)
            # Image.fromarray(image_bbox.astype(np.uint8)).save(gt_image_file)


            # print(image.shape)
            # print(smask.shape)
            visualize_sem_inst_mask(oimage, smask, np.array(bboxes), np.array(selected_masks), np.array([ICATE_SKETCH[x] for x in class_list]), class_names=list(SKETCH_CLASS_NAME.values()), sent=sentence, save_path=gt_inst_file, gt=True)

        if len(class_list) == 0:
            return dataset_dicts
        
        if sem_seg_gt is not None:
            dataset_dicts["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        
        # 最后一步：创建instance
        target = Instances(image_shape)
        target.gt_boxes = Boxes(list(boxes))
        classes = torch.tensor(class_list, dtype=torch.int64)
        target.gt_classes = classes

        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in selected_masks])
        )
        target.gt_masks = masks
        # print("#" * 20)
        # print("查看instance信息")
        # print(target)

        instances = target
        dataset_dicts["instances"] = utils.filter_empty_instances(instances)

        return dataset_dicts

    def __call__(self, data_dict):
        # print(data_dict)
        dataset_dicts = {}
        image_id = data_dict['image_id']
        cref_id = data_dict['ref_id']
        selected_instance_ids = data_dict['inst_ids'] if 'inst_ids' in data_dict else None

        # print("get gt")
        # image, selected_mask, masks, class_list, bboxes = self.get_gt(image_id, selected_instance_ids, self.height, self.width)
        oimage, image, selected_masks, smask, class_list, bboxes= self.get_gt_new(image_id, selected_instance_ids, self.height, self.width)


        # print("add bbox to image")
        
        # if not self.is_train:
        #     image_bbox = image.copy()
        #     for bbox, category_id in zip(bboxes, class_list):

                # print("add bbox to image")
                # print(bbox)

                # x0, y0, x1, y1 = bbox[1], bbox[0], bbox[3], bbox[2]
        #        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]

        #        ocate = ICATE_SKETCH[category_id]
                # color = [int(1.5*ocate), int(0.5*ocate), int(4.5*ocate)]
        #        color = [0, 255, 255]

        #        image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
        #        text_category = str(ocate)
        #        font = cv2.FONT_HERSHEY_SIMPLEX
        #        image_bbox = cv2.putText(image_bbox, text_category, (int(x1)-40, int(y1)-10), font, 0.5, (0, 255, 255), 2)
        
        # print(data_dict['raw_sent'])
        # plt.imshow(image_bbox)
        # plt.show()


        # assert 1 == 0

        # sentence = data_dict['raw_sent']
        sentence = 'both the butterflies'
        # sentence = "the two sheep on the left of the car"
        # sentence = "the two sheep on the left back of the tree"
        # sentence = "the two sheep on the left front of the houses"
        # sentence = "the two trees on the right of the dogs"
        # sentence = "the two ducks on the right"
        # sentence = "the left two ducks"
        # sentence = "the house and the person"
        # sentence = "the leftmost tree"
        # sentence = "the rightmost butterfly"
        # sentence = "the top cat"
        # sentence = "the leftmost cow"
        # sentence = "the two people on the left"
        # sentence = 'the horse on the rightmost'
        # sentence = "the two chickens on the right"
        # sentence = "the horse on the right"
        # sentence = "the two trees on the left of the house"
    
        if not self.is_train:    
            print(sentence)

        sentence = sentence.lower()
        tokens = sentence.split(" ")
        up_tokens = []
        for token in tokens:
            if token == '' or token == ' ':
                continue
            up_tokens.append(token)

        # print(len(tokens), tokens)
        sent_encode = self.get_sent_encode(up_tokens)
        # print(sent_encode.shape)
        # print(self.from_encode2sentence(sent_encode))
        # assert 1 == 0

        if not self.use_bert and not self.use_roberta_base or not self.use_clip:
            dataset_dicts['sent_encode'] = torch.as_tensor(sent_encode, dtype=torch.long)
        
        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height
        dataset_dicts['image_id'] = image_id
        dataset_dicts['cref_id'] = cref_id  #唯一识别标志
        dataset_dicts['file_name'] = "L0_sample" + str(image_id) + '.png'
        
        dataset_dicts['raw_sentence'] = sentence

        sem_seg_gt = None   # 暂时不需要
        boxes = np.array(bboxes)
        # print(image.shape)
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dicts["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if not self.is_train:
            dataset_dicts["oimage"] = torch.as_tensor(
                np.ascontiguousarray(oimage.transpose(2, 0, 1))
            )

        
        # print(dataset_dicts["image"].shape)
        # print(dataset_dicts["image"])
        # assert 1 == 0
        # image_bin = np.zeros([self.height, self.width], dtype=np.long)
        # image_bin[dataset_dicts["image"][0, :, :] == 0] = 1

        # dataset_dicts['sketch_bin'] = torch.as_tensor(image_bin, dtype=torch.long)

        if not self.is_train and 1==0:
            print("===========================")
            print("保存GT")
            print("===========================")
            # plt.imshow(image_bbox)
            # plt.show()
            #gt_sentence_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_sentence.txt")
            #gt_sem_mask_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_semantic_mask.png")
            #gt_image_file = osp.join(self.file_root, str(image_id) + '_' + str(cref_id) + "_gt_image_withbbox.png")

            file_root_per_image = os.path.join(self.file_root, str(image_id))
            os.makedirs(file_root_per_image, exist_ok=True)

            gt_inst_file = osp.join(file_root_per_image, str(image_id) + '_' + str(cref_id) + "_gt_image_inst.png")

            # with open(gt_sentence_file, 'w') as f:
            #     f.write(str(image_id) + '_' + str(cref_id) + ":\n" + str(sentence))

            # self.saveMask(smask, gt_sem_mask_file)
            # Image.fromarray(image_bbox.astype(np.uint8)).save(gt_image_file)


            # print(image.shape)
            # print(smask.shape)
            visualize_sem_inst_mask(oimage, smask, np.array(bboxes), np.array(selected_masks), np.array([ICATE_SKETCH[x] for x in class_list]), class_names=list(SKETCH_CLASS_NAME.values()), sent=sentence, save_path=gt_inst_file, gt=True)

        if len(class_list) == 0:
            return dataset_dicts
        
        if sem_seg_gt is not None:
            dataset_dicts["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        
        # 最后一步：创建instance
        target = Instances(image_shape)
        target.gt_boxes = Boxes(list(boxes))
        classes = torch.tensor(class_list, dtype=torch.int64)
        target.gt_classes = classes

        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in selected_masks])
        )
        target.gt_masks = masks
        # print("#" * 20)
        # print("查看instance信息")
        # print(target)

        instances = target
        dataset_dicts["instances"] = utils.filter_empty_instances(instances)

        return dataset_dicts

    