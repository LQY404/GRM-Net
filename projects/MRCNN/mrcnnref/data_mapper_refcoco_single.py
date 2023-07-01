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
import random


CATEGORY_refcoco = {}

CATE = {}

for i in range(1, 100):
    CATE[i] = i - 1

# CATE[91] = 0

ICATE = {}
for k, v in CATE.items():
    ICATE[v] = k



bad_words = ["le", "inhat", "mon", "reddishbrown", "orangeapple", "rightorange", "nextblue", "wii", ":(", "wplate", 
                "wtf", "mtf", "yep", "boobs", "frint", "1tbsp", "bib", "tattoos", "wqall", "oj", "lshoulder", "ikr", "cam", "ridht",
                "ye", "gy", "ya", "yee", "afto", "fonz", "btw", "ag", "crib", "gal", "rtmiddle", "seatmetal", "yuh", "ppl", "ella", "lbench", "ump", "ndswith", "yeah", "yuh", "gj",
                ""]

replace_words = {
    "ele": ["elephant"], "eleph": ["elephant"], "elep": ["elephant"], "r": ["right"], "rt": ["right"], "bab": ["baby"], "bottome": ["bottom"], "bottommiddle": ['bottom', 'middle'], 
    "frot": ["front"], "fron": ['front'], "fronbt": ['front'], "frt": ['front'], "frnt": ['front'], "theyre": ['they', 'are'], "umb": ['umbrella'], "umbrell": ['umbrella'],
    "umbre": ['umbrella'], "ki": ['kid'], "l": ['left'], "grirl": ['girl'], "girt": ['girl'], "lef": ['left'], "sittin": ['sitting'], "gon": ['on'], "infront": ['in', 'front'],
    "fromleft": ['from', 'left'], "blk": ['black'], "ion": ['on'], "bkac": ['black'], "fromt": ['front'], "labeledgreen": ['labeled', 'green'], "grn": ['green'], "z": ['zebra'],
    "whittish": ['white'], "wite": ['white'], "botttom": ['bottom'], "botom": ['bottom'], "bottomw": ['bottom'], "sppon": ['spoon'], "purplewhite": ['purple', 'white'],
    "pik": ['pink'], "pinnk": ['pink'], "wit": ['white'], "whit": ['white'], 'n': ['on'], "shirtgrey": ['shirt', 'gray'], "mman": ['man'], "mane": ['man'], "gril": ['girl'],
    "cneter": ['center'], "c": ['center'], 'u': ['us'], 'rowright': ['row', 'right'], 'bedhind': ['behind'], "pic": ['picture'], "yellin": ['yelling'], "greenyellow": ['green', 'yellow'],
    "bluewhite": ['blue', 'white'], "frontcenter": ['front', 'center'], "eft": ['left'], "backbutt": ['back', 'butt'], "helf": ['held'], "bluury": ['blurry'], "limo": ['car'], 
    "grrom": ['groom'], "woth": ['with'], "sittin": ['sitting'], "jqacket": ['jacket'], "cynan": ['cyan'], "redblack": ['red', 'black'], "th": ['the'], 'behin': ['behind'], 'ehind': ['behind'],
    "wbeer": ['with', 'beer'], 'unmberlla': ['umbrella'], "um": ['umbrella'], "eleft": ['left'], "umbrebrella": ['umbrella'], "righr": ['right'], 'littl': ['little'], 
    'blak': ['black'], 'midle': ['middle'], 'whearts': ['with', 'hearts'], "rightthanks": ['right'], "biek": ['bike'], "girafe": ['giraffe'], "thw": ['the'], "girraff": ['giraffe'], 
    "risghtside": ['right', 'side'], "froreground": ["foreground"], "prpulish": ['purple'], "girafee": ['giraffe'], "bottomright": ['bottom', 'right'], "topleftmost": ['top', 'leftmost'],
    "flowwer": ['flower'], "onleft": ['on', 'left'], "tha": ["that"], "te": ['the'], "withe": ['white'], "liek": ['like'], "bluw": ['blue'], "giraffee": ['giraffe'], "rght": ['right'],
    "kittty": ['kitty'], "whiteblue": ['white', 'blue'], "pokka": ['polka'], "animalon": ['animal'], "wondow": ['window'], "boar": ['board'], "cse": ['case'], "botom": ['bottom'], 
    "corener": ['corner'], "oman": ['woman'], "sand": ['sandwich'], "coffe": ['coffee'], "anima": ['animal'], "caraffe": ['carafe'], "m": ['man'], "veh": ['vehicle'], "sandwhch": ['sandwich'],
    "ppizza": ['pizza'], "sorry": [""], "whitehorse": ['white', 'horse'], "aaple": ['apple'], "perso": ['person'], "grene": ['green'], "ctr": ['center'], "unbrella": ['umbrella'], "cam": ['camera'],
    "babt": ['baby'], "babby": ['baby'], "bwteen": ['between'], "bluepurple": ['blue', 'purple'], "wih": ['with'], "platebowl": ['plate', 'bowl'], "bluegreen": ['blue', 'green'], 'comp': ['computer'],
    "com": ['computer'], "pizzzza": ['pizza'], "whitered": ['white', 'red'], "botom": ['bottom'], "blackwhite": ['black', 'white'], "bikr": ['bike'], "leftman": ['left', 'man'], "secong": ['second'], 
    "wcolorful": ['with', 'colorful'], "sandwhich": ['sandwich'], "chaiar": ['chair'], "theback": ['the', 'back'], "currrly": ['curly'], "toiletbidet": ['toilet'], "lowerright": ['lower', 'right'],
    "cakepie": ['cake'], "wbackpack": ['with', 'backpack'], "fencerailing": ['fence'], "doghnut": ['donut'], "don": ['donut'], "headright": ['head', 'right'], "bagunder": ['bag', 'under'], "behindbetween": ['behind'],
    "sppooon": ['spoon'], "g": ['girl'], "anial": ['animal'], "rgiht": ['right'], "whiteblack": ['white', 'black'], "leftwindowblack": ['left', 'window', 'black'], "behindslight": ['behind'], "cofee": ['coffee'],
    "blackhaired": ['black', 'haired'], "cente": ['center'], "yellowgreen": ['yellow', 'green'], "plantsflowers": ['plate', 'flowers'], "rightmounted": ['right', 'mounted'], "choc": ['chocolate'], 
    "choco": ['chocolate'], "jocket": ['jacket'], "jockeyt": ['jocket'], "wdark": ['with', 'dark'], "bck": ['back'], "abov": ['above'], "zeb": ["zebra"], "z": ['zebra'], "ffed": ['fed'], "offood": ['of', 'food'],
    "cjild": ['child'], "brownblack": ['brown', 'black'], "leftwhy": ['left', 'why'], "elft": ['left'], "hatmoneky": ['hat', 'monkey'], "brassmetal": ['metal'], "nana": ['banana'], "honeynice": ['honey', 'nice'],
    "midfle": ['middle'], "zebrain": ['zebra', 'in'], "foood": ['food'], "bottomleft": ['bottom', 'left'], "blueblack": ['blue', 'black'], "middlebottom": ['middle', 'bottom'], "yello": ['yellow'], "redorange": ['red', 'orange'],
    "centerbackground": ['center', 'background'], "grayblack": ['gray', 'black'], "gifafe": ['giraffe'], "wglasses": ['with', 'glasses'], "lookalike": ['look', 'alike'], "gall": ['girl'], "ocntainer": ['container'], 
    "wout": ['without'], "bakground": ['background'], "shirtlong": ['shirt', 'long'], "meetter": ['meter'], "ble": ['blue'], "redwhite": ['red', 'white'], "broc": ['broccoli'], "brocli": ['broccoli'], "csout": ['scout'],
    "redyellow": ['red', 'yellow'], "donutleft": ['dount', 'left'], "frm": ['from'], "bottm": ['bottom'], "bluegrey": ['blue', 'gray'], "vehicale": ['vehicle'], "wfruit": ['with', 'fruit'], "sleeveright": ['sleeve', 'right'],
    "middel": ['middle'], "whitepink": ['white', 'pink'], "lol": [''], "owoman": ['woman'], "hatscarf": ['hat', 'scarf'], "animalsheep": ['sheep'], "bottoma": ['bottom'], 
    "broccolli": ['broccoli'], "brocolli": ['broccoli'], "zeeb": ['zebra'], "bluejean": ['blue', 'jean'], "peerson": ['person'], 'lkeft': ['left'], "sk8er": ['skier'], "frme": ['frame'],
    "boi": ['boy'], "girffe": ['giraffe'], "2ndglass": ['second', 'glass'], "sheepgray": ['sheep', 'gray'], "ortange": ['orange'], "gif": ['giraffe'], "zbra": ['zebra'], "choclate": ['chocolate'],
    "donunut": ["doughnut"], "hor": ['horse'], "monnitor": ['monitor'], "dirnk": ['drink'], "slut": ['woman'], "roght": ['right'], "onright": ['on', 'right'], "wht": ['white'], "shrt": ['shirt'],
    "botton": ['bottom'], "jen": ['jeans'], "lowerleft": ['lower', 'left'], "purpleviolet": ['purple'], "chairthat": ['chair', 'that'], "pizzzzzzzzza": ['pizza'], "wo": ['without'],
    "suitcaseblack": ['suitcase', 'black'], "bottombye": ['bottom'], "pantswhite": ['pants', 'white'], "chaise": ['couch'], "lugg": ['luggage'], "middlef": ['middle'], "tannish": ['tan'],
    "wchunk": ['with', 'chunk'], "greenish": ['green'], 'reddish': ['red'], "grifere": ['giraffe'], "glss": ['glass'], "glas": ['glass'], "childe": ['child'], "lefthat": ['left', 'hat'], "gal": ['girl'],
    "visibls": ['visible'], "whitred": ['white', 'red'], "freig": ['fridge'], "middlee": ['middle'], "blackjacket": ['black', 'jacket'], "smallll": ['small'], "frontleft": ['front', 'left'],
    "skiis": ['skis'], "p": [""], "perosn": ['person'], "guyb": ['guy'], "umbrellacanopy": ['umbrella'], "botoom": ['bottom'], "riight": ['right'], "abovebeside": ['above'], "rom": ['from'],
    "empy": ['empty'], "suticase": ['suitcase'], "babe": ['baby'], "leftsunglasses": ['left', 'sunglasses'], "graffe": ['giraffe'], 

    }

bad_sents = ['coke can', "tv not on, right side", 'yes', 'no', 'Nope, it was all me! lol - guy on right with plaid shirt', 'yep', 'The one you just hit', 'on floor', "guy pissed he can't tennis",
        '~titty sprinkles~', "girl's racquet", 'DONUT WITHTONGS TOUCHING IT'
        ]
# from pytorch_pretrained_bert.tokenization import BertTokenizer

class RefcocoSingleDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # self.augmentation = build_augmentation(cfg, is_train)
        self.augmentation = None # 不使用数据增强
        print("不使用数据增强") if self.augmentation is None else print("使用数据增强")

        

        self.split = ('train' if is_train else 'val')
        # self.split = "testA"
        
        print("训练模式") if is_train else print("测试模式")
        self.is_train = is_train

        # self.ref_root = "/home/lingpeng/project/demo/data/"/
        self.ref_root = "/nfs/demo/data/"
        self.ref_type = ["refcoco", "refcoco+", "refcocog", "refclef"][0]
        self.ref_split = {
            "refcoco": ["unc", "google"],
            "refcoco+": ["unc"],
            "refcocog": ['umd', "google"],
            "refclef": ['unc', "berkeley"]
        }[self.ref_type][0]

        vocab_file = os.path.join(self.ref_root, self.ref_type, 'vocab.json')
        # vocab_file = os.path.join("/nfs/crefs/", "dict", self.ref_type, 'picked_c_vocab.json')

        simple = False
        easy = True
        begin = True
        use_attn = True
        use_iter = False
        self.file_root = "/home/lingpeng/project/Adet_2022/effe_inference_dir_new/mrcnn_R_101_1x_refcoco_unc_v4_pick_0.8"  \
                                                        +  ("_simple" if simple else "") + ('_easyconcat' if easy  else "")  \
                                                        + ('_rmi' if not easy and not use_attn else "")  \
                                                        + ("_atend" if not begin else "_atbegin")  \
                                                        + ("_simple" if use_attn else "")  \
                                                        + ("_use_1iter_withmattn_newversion" if use_iter else "") + '_' + "refcoco" \
                                                        + '_onenewgraph'
        # self.file_root = "/home/lingpeng/project/AdelaiDet-master/inference_dir/Condinst_MS_R_50_1x_iep_ref_v2"  + ("_simple" if self.simple else "") + ("_atend" if not begin else "") + "_newmethod"
      
        os.makedirs(self.file_root, exist_ok=True)
            
        # self.coco_image_path = "/home/lingpeng/project/demo/data/images/mscoco/images/train2014/"
        # self.coco_image_path = "/nfs/demo/data/images/mscoco/images/train2014/"
        self.coco_image_path = "/home/lingpeng/project/refcoco/images/train2014/"
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
            'refcoco': 39,
            'refcocog': 35,
            'refcoco+': 24,
            'refclef': 28
        }[self.ref_type] + 2

        # self.build_index()
        self.build_vocab(vocab_file)

        self.use_bert = False
        if self.use_bert:
            self.max_len += 1
            # self.tokenizer = BertTokenizer('/nfs/crefs/bert/bert-base-uncased-vocab.txt')

    
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

    def getMask(self, ann, image_height, image_width):
        # return mask, area and mask-center
        # ann = refToAnn[ref['ref_id']]
        # image = Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon  crowd = 0
            rle = MASK.frPyObjects(ann['segmentation'], image_height, image_width)
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

    def add_random_noisy(self, img, num=200):
        image_shape = img.shape

        assert image_shape[-1] == 3

        for i in range(num):
            hi = random.randint(0, img.shape[0]-1)
            wi = random.randint(0, img.shape[1]-1)
            ci = random.randint(0, 2)

            img[hi, wi, ci] = random.randint(0, 255)
        

        return img


    
    def __call__(self, dataset_dict):

        dataset_dicts = {}
        # print(dataset_dict)
        # assert 1 == 0

        cref_id = dataset_dict['cref_id'],
        ref_id = dataset_dict['ref_id']
        sent_id = dataset_dict['sent_id']

        ann = dataset_dict['ann']
        image_name = dataset_dict['file_name']
        image_id = dataset_dict['image_id']
        ocategory_id = dataset_dict['category_id']

        sentence = dataset_dict['sentence']['raw']
        tokens = dataset_dict['sentence']['tokens']

        image_file = os.path.join(self.coco_image_path, image_name) if self.ref_type != 'refclef' else os.path.join(self.saiapr_tc_12_image_path, image_name)
        # print(image_file)
        # assert 1 == 0
        img = Image.open(image_file).convert('RGB')
        img = np.array(img)

        oheight, owidth = img.shape[: -1]
        img = cv2.resize(img.astype(np.float32), (self.width, self.height)).astype(img.dtype)
        
        scale_factor_x, scale_factor_y = self.width*1.0 / owidth, self.height*1.0 / oheight
        bmask = np.zeros((self.height, self.width), dtype=np.uint8)
        smask = np.zeros((self.height, self.width))
        single_masks = []
        boxes = []
        classes = []

        category_id = CATE[ocategory_id]
        classes.append(category_id)

        
        mask_area = self.getMask(ann, oheight, owidth)
        mask, area = mask_area['mask'], mask_area['area']

        mask = cv2.resize(mask.astype(np.float32), (self.width, self.height)).astype(np.uint8)
        mask[mask > 1] = 0
        assert mask.sum() > 0

        smask = np.where((mask!=0)&(smask==0), smask+mask*ocategory_id, smask)
        single_masks.append(mask)


        box = ann['bbox']
        x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]

        x0 *= scale_factor_x
        x1 *= scale_factor_x
        y0 *= scale_factor_y
        y1 *= scale_factor_y
        boxes.append([x0, y0, x1, y1])




        sent_encode = self.get_sent_encode(tokens)

        dataset_dicts['sent_encode'] = torch.as_tensor(sent_encode, dtype=torch.long)
        
        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height
        dataset_dicts['image_id'] = image_id
        dataset_dicts['cref_id'] = cref_id  #唯一识别标志
        dataset_dicts['file_name'] = image_name
        
        dataset_dicts['raw_sentence'] = sentence

        boxes = np.array(boxes)
        # print(image.shape)
        image_shape = img.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dicts["image"] = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        )

        dataset_dicts["oimage"] = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        )
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


        if self.split != 'train':
            print(sentence)
            file_root_per_image = os.path.join(self.file_root, str(image_id))
            os.makedirs(file_root_per_image, exist_ok=True)
            image_bbox = img.copy()
            color = [
                int(1.5*category_id if self.split=='train' else ocategory_id), 
                int(0.5*category_id if self.split=='train' else ocategory_id), 
                int(4.5*category_id if self.split=='train' else ocategory_id)
            ]

            for mask, box in zip(single_masks, boxes):
                x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
  
                image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
                text_category = str(category_id if self.split=='train' else ocategory_id)
                font = cv2.FONT_HERSHEY_SIMPLEX
                image_bbox = cv2.putText(image_bbox, text_category, (int(x1)-40, int(y1)-10), font, 0.5, color, 2)
        
            # plt.imshow(image_bbox)
            # plt.show()

            # assert 1 == 0

            gt_sentence_file = osp.join(file_root_per_image, str(cref_id) + '_' + str(image_id) + "_gt_sentence.txt")
            gt_sem_mask_file = osp.join(file_root_per_image, str(cref_id) + '_' + str(image_id) + "_gt_semantic_mask.png")
            gt_image_file = osp.join(file_root_per_image, str(cref_id) + '_' + str(image_id) + "_gt_image_withbbox.png")

            with open(gt_sentence_file, 'w') as f:
                f.write(str(image_file) + ": " + str(sentence) + '\n' + str(tokens))

            self.saveMask(smask, gt_sem_mask_file)
            Image.fromarray(image_bbox.astype(np.uint8)).save(gt_image_file)


        return dataset_dicts


