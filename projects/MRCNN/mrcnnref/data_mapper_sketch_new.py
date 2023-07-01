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
        
        print("-"*20)
        print("新sketch datamapper")
        print("-"*20)
        self.split = 'train' if is_train else 'test'
        # self.split = 'train'
        # self.split = 'val'
        self.is_train = is_train

        print("训练模式") if is_train else print("测试模式")

        self.data_name = 'sketch' 

        if cfg.MODEL.NUM_CLASSES == 46:
            self.data_name = 'sketch'

        elif cfg.MODEL.NUM_CLASSES == 1272:
            self.data_name = 'phrasecut'

        elif cfg.MODEL.NUM_CLASSES == 48:
            self.data_name = 'iepref'
        
        else:
            raise
          
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
        self.file_root = "/home/lingpeng/project/Adet_2022/effe_inference_dir/mrcnn_R_101_1x_refcoco_unc_v4_pick_0.8"  \
                                                        +  ("_simple" if self.simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_simple" if use_attn else "")  \
                                                        + ("_use_1iter_withmattn_newversion" if use_iter else "") + '_' + self.data_name
        os.makedirs(self.file_root, exist_ok=True)
          
        # self.height, self.width = 640, 640
        # self.height, self.width = (512, 512) if self.is_train else (786, 786)
        self.height, self.width = (512, 512)
        # self.height, self.width = (786, 786)
        # self.height, self.width = (768, 768)
        # self.height, self.width = (770, 770)
    
        self.max_len = 15

        self.use_bert = cfg.MODEL.USE_BERT
        
        self.use_roberta_base = False

        if self.use_bert:
            print("使用bert")
            self.max_len += 2

        if self.use_roberta_base:
            assert not self.use_bert
            print("使用roberta")
            self.max_len += 2   # 前后各加上一个pad，总长度为4，

        
        self.inference_root = "/home/lingpeng/project/SparseR-CNN-main/prepare_for_teaser"
        self.data_folder = "/home/lingpeng/project/prepare_sketch_data"
        self.data_prefix = 'train' if is_train else "test"
        os.makedirs(os.path.join(self.data_folder, self.data_prefix), exist_ok=True)
    

    def saveMask(self, mask, save_dir):
        lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())

        lbl_pil.save(save_dir)

    def __call__(self, data_dict):
      
        cref_id = data_dict['ref_id']
        data_file = os.path.join(self.data_folder, self.data_prefix, 'sketch_' + str(cref_id) + '.npz')
        datas = np.load(data_file)
        image_id = data_dict['image_id']
        assert image_id == datas['image_id']
        
        bboxes = datas['bboxes']
        sent_encode = datas['sent_encode']
        mask_attention = datas['mask_attention']
        raw_sentence = datas['raw_sentence']
        image = datas['image']
        
        
        smask = datas['smask']
        
        # print(self.tokenizer.convert_ids_to_tokens(sent_encode))
        # print(self.from_encode2sentence(sent_encode))
        # assert 1 == 0
        # print(sent_encode.shape)
        # assert 1 == 0
        dataset_dicts = {}
        dataset_dicts['sent_encode'] = torch.as_tensor(sent_encode, dtype=torch.long)
        if self.use_bert:
            dataset_dicts['mask_attention'] = torch.as_tensor(mask_attention, dtype=torch.long)
        
        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height
        dataset_dicts['image_id'] = image_id
        dataset_dicts['cref_id'] = cref_id  #唯一识别标志
        dataset_dicts['file_name'] = "L0_sample" + str(image_id) + '.png'
        
        dataset_dicts['raw_sentence'] = raw_sentence
        selected_masks = datas['selected_masks']
        class_list = datas['class_list']
        
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
            oimage = datas['oimage']
            dataset_dicts["oimage"] = torch.as_tensor(
                np.ascontiguousarray(oimage.transpose(2, 0, 1))
            )

        
        # print(dataset_dicts["image"].shape)
        # print(dataset_dicts["image"])
        # assert 1 == 0
        # image_bin = np.zeros([self.height, self.width], dtype=np.long)
        # image_bin[dataset_dicts["image"][0, :, :] == 0] = 1

        # dataset_dicts['sketch_bin'] = torch.as_tensor(image_bin, dtype=torch.long)

        if not self.is_train:
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
            visualize_sem_inst_mask(oimage, smask, np.array(bboxes), np.array(datas['selected_masks']), np.array([ICATE_SKETCH[x] for x in class_list]), class_names=list(SKETCH_CLASS_NAME.values()), sent=sentence, save_path=gt_inst_file, gt=True)

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

    