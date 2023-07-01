

from inspect import Parameter
from projects.MRCNN.mrcnnref.utils.comm import tokenize


def t6():
    import torch
    import clip
    import numpy as np
    
    from PIL import Image
    print("测试CLIP------------------------------")
    print(clip.available_models()) # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN101", device=device)

    image_file = [
                "/home/lingpeng/project/SketchyScene-pytorch/data/val/DRAWING_GT/L0_sample2.png",  # complete
                "/home/lingpeng/project/dataset/cat.png",
                "/home/lingpeng/project/dataset/tree.png",
                "/home/lingpeng/project/dataset/sun.png",
                "/home/lingpeng/project/dataset/grass.png",
                "/home/lingpeng/project/dataset/cloud.png"
                ][1]

    image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
    text = clip.tokenize(["the cloud", "the tree", "the left cat", 'the sun', "grass", "the dog"]).to(device)
    
    print(image.shape) # [N, 3, 224, 224]
    # image = torch.randn((3, 3, 224, 224)).to(device)
    # logit_scale = torch.nn.Parameter(torch.ones([])*np.log(1 / 0.07))
    # print(text)
    with torch.no_grad():
        image_features = model.encode_image(image)
        # print(image_features.shape) # [N, 512]

        # assert 1 == 0
        text_features = model.encode_text(text)
        # print(text_features.shape) # [len(text), 512]
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        print(logits_per_image.shape) # [N, len(text)]
        print(logits_per_text.shape)  # [len(text), N]
        print(logits_per_image.softmax(-1), logits_per_text)
        
        print('====================================================')

        # print(lc.shape, rc.shape)

        # print(torch.bmm(lc, rc))

        print(image_features.shape) # [N, DIM]
        print(text_features.shape)  # [len(text), DIM]
        
        logits_per_image, logits_per_text = model(image, text)
        print(logits_per_image.shape) # [N, len(text)]
        print(logits_per_text.shape)  # [len(text), N]
        print(logits_per_image.softmax(-1), logits_per_text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


def t7():
    import json
    from mrcnnref.phrasecut.refvg_loader import RefVGLoader
    from mrcnnref.phrasecut.file_paths import name_att_rel_count_fpath

    splits = ['train', 'test']

    dicts = dict()
    dicts["refexp_token_to_idx"] = {}
    dicts["refexp_token_to_idx"]["<NULL>"] = 0
    dicts["refexp_token_to_idx"]['<START>'] = 1
    dicts["refexp_token_to_idx"]['<END>'] = 2
    dicts["refexp_token_to_idx"]['UNK'] = 3

    info_account = json.load(open(name_att_rel_count_fpath, 'r'))['cat']
    infos = {}
    cates = {}
    for item in info_account:
        c, v = item[0], item[1]

        infos[c] = v

        cates[c] = len(cates) + 1

    max_len = 0
    for split in splits:
        refvg_loader = RefVGLoader(split=split)

        

        for img_id in refvg_loader.img_ids:
            ref = refvg_loader.get_img_ref_data(img_id)

            for task_i, task_id in enumerate(ref['task_ids']):
                ps = ref['p_structures'][task_i]
                # assert ps['type'] == 'name'

                cate_name = ps['name']

                if cate_name not in infos or infos[cate_name] <= 20:  # 控制类别：1272
                    continue

                phrase= ref['phrases'][task_i]

                sent = phrase.lower()
                tokens = sent.split(' ')
                max_len = max(max_len, len(tokens))
                for token in tokens:
                    if token in dicts["refexp_token_to_idx"]:
                        continue

                    dicts["refexp_token_to_idx"][token] = len(dicts["refexp_token_to_idx"])
    
    
    with open("/nfs/crefs/dict/phrasecut/dict_1272.json", 'w') as f:
        json.dump(dicts, f, indent=4)

    print(max_len)


def t8():
    import os
    import numpy as np
    from PIL import Image
    import json
    import cv2
    from mrcnnref.phrasecut.refvg_loader import RefVGLoader
    from mrcnnref.phrasecut.file_paths import name_att_rel_count_fpath, img_fpath

    # info_account = json.load(open(name_att_rel_count_fpath, 'r'))['cat']
    
    new_images = os.path.join("/home/lingpeng/project/PhraseCutDataset/data/VGPhraseCut_v0/images_v1/VG_100k")
    os.makedirs(new_images, exist_ok=True)
    width, height = 768, 768
    for split in ['train', 'test']:
        
        refvg_loader = RefVGLoader(split=split)

        for img_id in refvg_loader.img_ids:
            image_name = os.path.join(img_fpath, '%d.jpg' % img_id)

            oimg = Image.open(image_name).convert('RGB')
            img = np.array(oimg)
            img = cv2.resize(img.astype(np.float32), (width, height)).astype(img.dtype)

            img = Image.fromarray(img)

            nimage_name = os.path.join(new_images, '%d.jpg' % img_id)

            img.save(nimage_name)

            # assert 1 == 0
            print(img_id)


def t9():
    import os
    import os.path as osp
    import sys
    import json
    import time
    import numpy as np
    import h5py
    import cv2
    from PIL import Image
    import scipy.io
    import scipy.ndimage
    # from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser
    # from mrcnnref.data_mapper_sketch import SketchDatasetMapper
    # from train_net import setup

    # args = default_argument_parser().parse_args()
    # cfg = setup(args)

    splits = ['train', 'test']
    ref_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data"

    height, width = 768, 768
    for split in splits:
        cref_name = "sentence_instance_" + split + ".json"
        cref_file = os.path.join(ref_root, cref_name)
        crefs = json.load(open(cref_file, 'r'))
        # np.random.shuffle(crefs)
        data_dict = []

        for cref in crefs:
            cimage_id = int(cref['key'])

            for k, v in cref['sen_instIdx_map'].items():

                ref = {}
                ref['image_id'] = cimage_id

                sent = k
                ins_ids = v

                ref["raw_sent"] = sent
                ref['inst_ids'] = ins_ids
                ref['ref_id'] = len(data_dict)

                data_dict.append(ref)
        
        image_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data/" + split + "/DRAWING_GT"
        semantic_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data/" + split + "/CLASS_GT"
        instance_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data/" + split + "/INSTANCE_GT"

        dataset_h5_name = '/home/lingpeng/project/dataset/sketchyscene_copy1/data/' + split + '.h5'

        f = h5py.File(dataset_h5_name, 'w')
        drawing_set = f.create_dataset("sketch_drawing", (len(data_dict), 768, 768, 3), dtype=np.int32)
        semantic_set = f.create_dataset("semantic", (len(data_dict), 750, 750), dtype=np.int32)
        instance_set = f.create_dataset("instance", (len(data_dict), 750, 750), dtype=np.uint8)


        # dataloader = SketchDatasetMapper(cfg, is_train=split=='train')
        # 先获取所有的数据

        for index, info in enumerate(data_dict):
            image_id = info['image_id']
            cref_id = info['ref_id']
            selected_instance_ids = info['inst_ids']

            image_name = "L0_sample" + str(image_id) + '.png'
            semantic_name = "sample_" + str(image_id) + "_class"
            instance_name = "sample_" + str(image_id) + "_instance"

            image_path = os.path.join(image_root, image_name)
            semantic_path = os.path.join(semantic_root, semantic_name)
            instance_path = os.path.join(instance_root, instance_name)

            im = Image.open(image_path).convert('RGB')
            im = np.array(im)  # 0-225
            im_type = im.dtype
            ori_h, ori_w = im.shape[: -1]
            if height != ori_h or width != ori_w:
                assert height != -1 and width != -1
                im = cv2.resize(im.astype(np.float32), (width, height)).astype(im_type)
        
            # 载入mask
            INSTANCE_GT = scipy.io.loadmat(instance_path)['INSTANCE_GT']
            INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
            CLASS_GT = scipy.io.loadmat(semantic_path)['CLASS_GT']  # (750, 750)


            # print(im.shape)
            # print(INSTANCE_GT.shape)
            # print(CLASS_GT.shape)

            # assert 1 == 0
            drawing_set[index] = im
            semantic_set[index] = CLASS_GT
            instance_set[index] = INSTANCE_GT


            # assert 1 == 0
            print(index)
        print("finish " + split)
        f.close()



def t10():
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import cm
    # %matplotlib inline

    image_dir = '/home/lingpeng/project/dataset/sketchyscene_copy1/data/val/DRAWING_GT'
    image_name = "L0_sample%d.png"

    batch_size = 1

    def edit_images(image1, image2):
        assert image1.shape == image2.shape
        h, w, c = image1.shape
        y, x = np.ogrid[0:h, 0:w]
        mask = (x - w / 2) ** 2 + (y - h / 2) ** 2 > h * w / 9
        result1 = np.copy(image1)
        result1[mask] = image2[mask]
        result2 = np.copy(image2)
        result2[mask] = image1[mask]

        return result1, result2

    def show_images(image_batch):
        columns = 4
        rows = (batch_size + 1) // columns
        fig = plt.figure(figsize=(32, (32 // columns) * rows))
        gs = gridspec.GridSpec(rows, columns)
        for j in range(rows*columns):
            plt.subplot(gs[j])
            plt.axis("off")
            plt.imshow(image_batch.at(j))

    python_function_pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0,
                                    exec_async=False, exec_pipelined=False, seed=99)
    
    with python_function_pipe:
        print("____________________________")
        print(image_dir)
        input1 = fn.readers.file(file_root=image_dir, random_shuffle=True)
        # input2, _ = fn.readers.file(file_root=image_dir, random_shuffle=True)
        im1 = fn.decoders.image(input1, device='mixed', output_type=types.RGB)
        # res1, res2 = fn.resize([im1, im2], resize_x=300, resize_y=300)
        # out1, out2 = fn.python_function(res1, res2, function=edit_images, num_outputs=2)
        print(type(im1))
        python_function_pipe.set_outputs(im1)

    python_function_pipe.build()
    ims1 = python_function_pipe.run()

    show_images(ims1)
    # show_images(ims2)

def t11():
    import os
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types

    image_dir = '/home/lingpeng/project/dataset/sketchyscene_copy1/data/val/DRAWING_GT'
    max_batch_size = 10

    image_name = "L0_sample%d.png"

    @pipeline_def
    def simple_pipeline():
        jpegs  = fn.readers.file(file_list=[os.path.join(image_dir, "L0_sample2.png")][0])
        print(type(jpegs))  # <class 'list'>

        images = fn.decoders.image(jpegs, device='cpu', output_type=types.RGB)
        print(type(images)) # <class 'list'>
        print(len(images))
        print(type(images[0]))
        print(type(images[1]))
        # print(len(images[0]))
        return images[0]
    
    pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    print("ready")
    pipe.build()
    print("running")
    pipe.run()



def t12():
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    tokens = tokenizer("woman with hat", return_tensors="pt")

    print(tokens)  # 前后各加上一个pad，总长度为6，
 
    print(tokens.char_to_token(8)) # 所以当char_to_token(para)的参数超过6后的输出都是None

    tokens_positive = []
    # print(tokens['input_ids'].shape)
    start_pos = 0
    while tokens.char_to_token(start_pos) is not None and start_pos < tokens['input_ids'].shape[-1]:
        end_pos = start_pos + 1
        while tokens.char_to_token(end_pos) is not None:
            end_pos += 1

        if end_pos != start_pos + 1:         
            tokens_positive.append([start_pos, end_pos])
            start_pos = end_pos + 1

    print(tokens_positive)

def t13():
    from transformers import RobertaTokenizerFast
    import json
    import scipy
    import numpy as np
    import os
    from mrcnnref.data_mapper_sketch import SKETCH_CLASS_NAME

    def get_gt_new(semantic_root, instance_root, image_id, selected_instance_ids, height=-1, width=-1):
        assert type(selected_instance_ids) is list
        selected_instance_ids_ = [item for item in selected_instance_ids]

        # image_name = "L0_sample" + str(image_id) + '.png'
        semantic_name = "sample_" + str(image_id) + "_class"
        instance_name = "sample_" + str(image_id) + "_instance"

        instance_path = os.path.join(instance_root, instance_name)
        semantic_path = os.path.join(semantic_root, semantic_name)

        INSTANCE_GT = scipy.io.loadmat(instance_path)['INSTANCE_GT']
        INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
        CLASS_GT = scipy.io.loadmat(semantic_path)['CLASS_GT']  # (750, 750)
        instance_count = np.bincount(INSTANCE_GT.flatten())  # 图中所有的实例 [inst_id: inst_pixel]

        instance_count = instance_count[1:]  # e.g. shape=(101,)
        
        classes = []
        bmasks = []
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

                assert np.sum(mask) != 0

                class_gt_filtered = CLASS_GT * mask
                class_gt_filtered = np.bincount(class_gt_filtered.flatten())
                class_gt_filtered = class_gt_filtered[1:]
                class_id = np.argmax(class_gt_filtered) + 1
                # nclass_id = CATE[class_id]
                classes.append(int(class_id))

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
            boxes.append(np.array([x1, y1, x2-x1, y2-y1]))  # xywh for coco

        return boxes, classes


        

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


    data_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"

    for split in ['val', 'train', 'test']:
        semantic_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/CLASS_GT"
        # self.instance_root = "/nfs/SketchyScene-pytorch/data/" + self.split + "/INSTANCE_GT"
        instance_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/INSTANCE_GT"


        tdata = json.load(open(data_root + "/sentence_instance_" + str(split) + ".json", 'r'))
        convert_sketchcoco = {}
        image_infos = []
        annotations = []

        for td in tdata:
            image_id = int(td['key'])
            image_infos.append({
                "file_name": "L0_sample" + str(image_id) + ".png",
                "height": 768,
                "width": 768,
                "id": image_id
            })
            print(image_id)

            for k, v in td['sen_instIdx_map'].items():
                sent = k
                ins_ids = v
                # break
                boxes, class_ids = get_gt_new(semantic_root, instance_root, image_id, ins_ids)

                # 添加"tokens_positive"字段
                tokens = tokenizer(sent, return_tensors="pt")
                tokens_positive = []

                start_pos = 0
                while tokens.char_to_token(start_pos) is not None and start_pos < tokens['input_ids'].shape[-1]:
                    end_pos = start_pos + 1
                    while tokens.char_to_token(end_pos) is not None:
                        end_pos += 1
                    
                    if end_pos != start_pos + 1:         
                        tokens_positive.append([start_pos, end_pos])
                        start_pos = end_pos + 1
                    else:
                        break
                    
                assert len(tokens_positive) != 0
                tokens_positive_full = []
                # print(len(ins_ids))
                for _ in range(len(ins_ids)):
                    tokens_positive_full.append(tokens_positive.copy())

                assert len(ins_ids) == len(tokens_positive_full)

                class_ids_ = set(class_ids)
                assert len(class_ids_) == 1
                annotations.append({
                    'sentence': sent,
                    'ins_ids': ins_ids,
                    'image_id': image_id, 
                    'id': len(annotations),
                    "tokens_positive": tokens_positive_full,
                    "category_id": class_ids[0],
                    # 'bbox': 
                })

        convert_sketchcoco['images'] = image_infos
        convert_sketchcoco["annotations"] = annotations


        categories = []
        for k, v in SKETCH_CLASS_NAME.items():
            categories.append({
                "supercategory": v,
                "id": k,
                "name": v
            })

        convert_sketchcoco["categories"] = categories

        with open(data_root + "/sentence_instance_" + str(split) + "_coco_v4.json", 'w') as f:
            json.dump(convert_sketchcoco, f, indent=2)
        
        print("processed the " + split)




def t14():
    from transformers import RobertaTokenizerFast
    import json
    import scipy
    import numpy as np


    def get_gt_new(semantic_path, instance_path, image_id, selected_instance_ids, height=-1, width=-1):
        assert type(selected_instance_ids) is list
        selected_instance_ids_ = [item for item in selected_instance_ids]

        image_name = "L0_sample" + str(image_id) + '.png'
        semantic_name = "sample_" + str(image_id) + "_class"
        instance_name = "sample_" + str(image_id) + "_instance"

        INSTANCE_GT = scipy.io.loadmat(instance_path)['INSTANCE_GT']
        INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
        CLASS_GT = scipy.io.loadmat(semantic_path)['CLASS_GT']  # (750, 750)
        instance_count = np.bincount(INSTANCE_GT.flatten())  # 图中所有的实例 [inst_id: inst_pixel]

        instance_count = instance_count[1:]  # e.g. shape=(101,)
        
        classes = []
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

                assert np.sum(mask) != 0

                class_gt_filtered = CLASS_GT * mask
                class_gt_filtered = np.bincount(class_gt_filtered.flatten())
                class_gt_filtered = class_gt_filtered[1:]
                class_id = np.argmax(class_gt_filtered) + 1
                # nclass_id = CATE[class_id]
                classes.append(class_id)
        

        return 


        

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


    data_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"

    for split in ['train', 'test', 'val']:

        semantic_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/CLASS_GT"
        # self.instance_root = "/nfs/SketchyScene-pytorch/data/" + self.split + "/INSTANCE_GT"
        instance_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/INSTANCE_GT"

        tdata = json.load(open(data_root + "/sentence_instance_" + str(split) + ".json", 'r'))
        annotations_with_positive = []
        
        
        for td in tdata:
            image_id = int(td['key'])

            td_wp = {}
            td_wp['key'] = image_id

            sen_instIdx_map = {}
            tokens_positives = {}

            for k, v in td['sen_instIdx_map'].items():
                sent = k
                ins_ids = v
                # print(sent)
                sen_instIdx_map[k] = v
                # break

                # 添加"tokens_positive"字段
                tokens = tokenizer(sent, return_tensors="pt")
                tokens_positive = []

                start_pos = 0
                while tokens.char_to_token(start_pos) is not None and start_pos < tokens['input_ids'].shape[-1]:
                    end_pos = start_pos + 1
                    while tokens.char_to_token(end_pos) is not None:
                        end_pos += 1
                    
                    if end_pos != start_pos + 1:         
                        tokens_positive.append([start_pos, end_pos])
                        start_pos = end_pos + 1
                    else:
                        break
                    
                assert len(tokens_positive) != 0

                tokens_positive_full = []
                # print(len(ins_ids))
                for _ in range(len(ins_ids)):
                    tokens_positive_full.append(tokens_positive.copy())

                assert len(ins_ids) == len(tokens_positive_full)
                # print(tokens_positive)
                # print(tokens_positive_full)
                # assert 1 == 0
                tokens_positives[k] = tokens_positive_full.copy()

            td_wp["tokens_positive"] = tokens_positives.copy()
            td_wp['sen_instIdx_map'] = sen_instIdx_map.copy()

                
            annotations_with_positive.append(td_wp)
        
        with open(data_root + "/sentence_instance_" + str(split) + "_with_positiveMap_v2.json", 'w') as f:
            json.dump({"annotations": annotations_with_positive}, f, indent=2)
        
        print("processed the " + split)



def t15():  # evaluate for the prediction of mdetr
    import json
    import os
    from mrcnnref.ref_evaluator import compute_bbox_iou
    import numpy as np

    def box_iou_xyxy(box1, box2):
        x1min, y1min, x1max, y1max = box1
        x2min, y2min, x2max, y2max = box2
        
        area1 = (y1max - y1min + 1) * (x1max - x1min + 1)
        area2 = (y2max - y2min + 1) * (x2max - x2min + 1)
        
        xmin = np.maximum(x1min, x2min)
        ymin = np.maximum(y1min, y2min)
        
        xmax = np.minimum(x1max, x2max)
        ymax = np.minimum(y1max, y2max)
        
        inter_h = np.maximum(ymax - ymin + 1, 0)
        inter_w = np.maximum(xmax - xmin + 1, 0)
        intersection = inter_h * inter_w
        
        union = area1 + area2 - intersection
    
        iou = intersection / union
        
        return iou
    
    def mask_iou(mask1, mask2):
        # print(mask1.dtype, mask2.dtype)

        # assert 1 == 0
        # mask1 = mask1.astype(np.uint8)
        # mask2 = mask2.astype(np.uint8)

        # intersection = np.sum(np.logical_and(mask1, mask2))
        # union = np.sum(np.logical_or(mask1, mask2))
        mask1 = np.reshape(mask1.astype(np.float32), (-1))
        mask2 = np.reshape(mask2.astype(np.float32), (-1))

        area1 = np.sum(mask1)
        area2 = np.sum(mask2)

        intersection = np.dot(mask1.T, mask2)
        union = area1 + area2 - intersection

        return intersection / union

    def compute_ap(self, gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks=None,
        iou_threshold=0.5, score_threshold=0.0, image_bin=None):
        if isinstance(pred_boxes, list):
            pred_boxes = np.array(pred_boxes)
        
        if isinstance(gt_boxes):
            gt_boxes = np.array(gt_boxes)

        overlaps = np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)  # 以mask为基础计算，在sketch中会存在一些问题

        for pi in range(pred_boxes.shape[0]):

            pred_bbox = pred_boxes[pi]
            pred_class = pred_class_ids[pi]
            pred_mask = pred_masks[pi] if pred_masks is not None else None
            # print(pred_mask)

            for gi in range(gt_boxes.shape[0]):
                gt_bbox = gt_boxes[gi]
                gt_class = gt_class_ids[gi]
                gt_mask = gt_masks[gi]

                if pred_mask is None:
                    overlaps[pi, gi] = box_iou_xyxy(pred_bbox, gt_bbox)
                else:
                    overlaps[pi, gi] = mask_iou(pred_mask, gt_mask)
            
        match_count = 0
        # pred_match = -1 * np.ones([pred_boxes.shape[0]])
        pred_match = np.zeros([pred_boxes.shape[0]])
        # gt_match = -1 * np.ones([gt_boxes.shape[0]])
        gt_match = np.zeros([gt_boxes.shape[0]])

        for i in range(pred_boxes.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            # low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            # if low_score_idx.size > 0:
            #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            # 找到与pred_i最匹配的Gt
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                # if gt_match[j] > 0:
                if gt_match[j] == 1:
                    continue
            
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_threshold:
                    break

                # Do we have a match?
                # if pred_class_ids[i] == gt_class_ids[j]:
                #     match_count += 1
                #     gt_match[j] = i
                #     pred_match[i] = j
                #     break
                # else:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                
                break
        
        # 利用上面产生的gt_match, pred_match, overlaps计算
        # Compute precision and recall at each prediction box step
        # precisions = np.cumsum(pred_match > -1).astype(np.float32) / (np.arange(len(pred_match)) + 1)
        precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
        # recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
        recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

        # Pad with start and end values to simplify the math
        precisions = np.concatenate([[0.], precisions, [0.]])
        recalls = np.concatenate([[0.], recalls, [1.]])
        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])
        
        # Compute mean AP over recall range
        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                    precisions[indices])
        
        
        assert mAP <= 1.0
        return mAP, precisions, recalls, overlaps



    result_file = ""
    results = json.load(open(result_file, 'r'))

    score_threshold = 0.0
    iou_threshold = 0.5
    iou_thresholds = np.linspace(.5, 0.95, round((0.95 - .5) / .05) + 1, endpoint=True)

    AP_list = []
    pre_list = []
    recall_list = []
    
    APs = []

    for result in results:

        gt_boxes = result['gt_boxes']
        gt_masks = result['gt_makss']
        gt_classes = result['gt_class_ids']

        scores = result['scores']  # >0.5
        pred_boxes = result['pred_boxes']
        pred_classes = result['pred_class_ids']

        t_AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
        t_pre_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
        t_recall_list = np.zeros([len(iou_thresholds)], dtype=np.float32)

        ap = 0.0

        if len(scores) == 0:
            APs.append(0.0)
            AP_list.append(t_AP_list)
            pre_list.append(t_pre_list)
            recall_list.append(t_recall_list)

            print("no instance be predicted")

            print("累积AP[@0.5]: ", np.mean(APs))
            print("累积的mAP_list[@.5:0.95]: ", np.mean(AP_list, axis=0))
            print("累积mAP: ", np.mean(AP_list))

            print("累积的mPre_list[@.5:0.95]: ", np.mean(pre_list, axis=0))
            print("累积mPre: ", np.mean(pre_list))

            print("累积的mRecall_list[@.5:0.95]: ", np.mean(recall_list, axis=0))
            print("累积mRecall: ", np.mean(recall_list))
            
            continue

        ap, precisions, recalls, overlaps = compute_ap(gt_boxes, gt_classes, gt_masks, 
            pred_boxes, pred_classes, scores, pred_masks=None, 
            iou_threshold=iou_threshold, score_threshold=score_threshold, image_bin=None)
        

        APs.append(ap)

        for j in range(len(iou_thresholds)):
            iouThr = iou_thresholds[j]
            AP_single_iouThr, precisions, recalls, overlaps = compute_ap(gt_boxes, gt_classes, gt_masks, 
                                                                pred_boxes, pred_classes, scores, pred_masks=None, 
                                                                iou_threshold=iouThr, score_threshold=score_threshold, image_bin=None)
            
            t_AP_list[j] = AP_single_iouThr
            t_pre_list[j] = np.mean(precisions)
            # pre_list[j] = precisions[-1]
            t_recall_list[j] = np.mean(recalls)
        
        AP_list.append(t_AP_list)
        pre_list.append(t_pre_list)
        recall_list.append(t_recall_list)


        print("累积AP[@0.5]: ", np.mean(APs))
        print("累积的mAP_list[@.5:0.95]: ", np.mean(AP_list, axis=0))
        print("累积mAP: ", np.mean(AP_list))

        print("累积的mPre_list[@.5:0.95]: ", np.mean(pre_list, axis=0))
        print("累积mPre: ", np.mean(pre_list))

        print("累积的mRecall_list[@.5:0.95]: ", np.mean(recall_list, axis=0))
        print("累积mRecall: ", np.mean(recall_list))
    

def t16():
    import json
    import os
    from mrcnnref.phrasecut.refvg_loader import RefVGLoader
    from mrcnnref.datamapper_phrasecut import PhraseCutDatasetMapper
    from PIL import Image
    import numpy as np
    import cv2

    dataset_dir = '/home/lingpeng/project/PhraseCutDataset/data/VGPhraseCut_v0/'
    # img_fpath = dataset_dir.joinpath('images')
    img_fpath = os.path.join(dataset_dir, 'images', 'VG_100K')
    # name_att_rel_count_fpath = dataset_dir.joinpath('name_att_rel_count.json')
    name_att_rel_count_fpath = os.path.join(dataset_dir, 'name_att_rel_count.json')

    info_account = json.load(open(name_att_rel_count_fpath, 'r'))['cat']
    infos = {}
    cates = {}
    for item in info_account:
        c, v = item[0], item[1]

        infos[c] = v

        cates[c] = len(cates) + 1

    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
    
    refvg_loader = RefVGLoader(split='test')

    for img_id in refvg_loader.img_ids:

        if img_id != 3368:
            continue

        
        ref = refvg_loader.get_img_ref_data(img_id)
        for task_i, task_id in enumerate(ref['task_ids']):
            print(ref['phrases'][task_i])

            gt_image_file = "./" + str(img_id) + "_" + str(task_id) + ".png"
    
            bboxes = ref['gt_boxes'][task_i]
            oimg = Image.open(os.path.join(img_fpath, '%d.jpg' % img_id)).convert('RGB')
            # print(oimg.size)
            img = np.array(oimg)
            image_bbox = img.copy()
            for box in bboxes:
                x0, y0, x1, y1 = box[0], box[1], box[0] + box[2], box[1] + box[3]
                image_bbox = cv2.rectangle(image_bbox, (int(x0), int(y0)), (int(x1), int(y1)), [0, 0, 0], 2)

            Image.fromarray(image_bbox.astype(np.uint8)).save(gt_image_file)


def t17():
    from transformers import RobertaModel, RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    text_encoder = RobertaModel.from_pretrained("roberta-base")
    for p in text_encoder.parameters():
        p.requires_grad_(False)
                
    tokens = tokenizer.batch_encode_plus(["man", "the person"], padding="longest", return_tensors="pt")
    encoded_text = text_encoder(**tokens)
    
    print(tokens)  # 前后各加上一个pad，总长度为4，
    print(encoded_text) # 包括：last_hidden_state，hidden_state，以及pooler_output
    
    hn = encoded_text.last_hidden_state
    print(hn.shape)  # [N, max_len, 768]
    
    
def t18(split='train'):
    import json
    import os
    import numpy as np
    import scipy
    import scipy.io
    
    ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax
    # copy
    # ref_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data"
    
    semantic_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/CLASS_GT"
    instance_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/INSTANCE_GT"
    
    
    cref_name = "sentence_instance_" + split + ".json"

    cref_file = os.path.join(ref_root, cref_name)

    crefs = json.load(open(cref_file, 'r'))
    
    # np.random.shuffle(crefs)
    data_dict = []

    class_count = {}
    
    for cref in crefs:
        cimage_id = int(cref['key'])

        # if cimage_id not in [202]:
        #     continue    

        for k, v in cref['sen_instIdx_map'].items():

            ref = {}
            ref['image_id'] = cimage_id
            # ref['image_id'] = 999193

            sent = k

            ins_ids = v
            
            semantic_name = "sample_" + str(cimage_id) + "_class"
            instance_name = "sample_" + str(cimage_id) + "_instance"
            
            semantic_path = os.path.join(semantic_root, semantic_name)
            instance_path = os.path.join(instance_root, instance_name)
            
            INSTANCE_GT = scipy.io.loadmat(instance_path)['INSTANCE_GT']
            INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
            CLASS_GT = scipy.io.loadmat(semantic_path)['CLASS_GT']  # (750, 750)
            instance_count = np.bincount(INSTANCE_GT.flatten())  # 图中所有的实例 [inst_id: inst_pixel]
            
            instance_count = instance_count[1:]  # e.g. shape=(101,)
            selected_instance_ids_ = [item for item in ins_ids]
            real_instanceIdx = 0
            classes = []
            
            for i in range(instance_count.shape[0]):
                if instance_count[i] == 0:
                    continue
                
                instanceIdx = i + 1

                if real_instanceIdx in ins_ids:
                    selected_instance_ids_.remove(real_instanceIdx)
                    mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
                    mask[INSTANCE_GT == instanceIdx] = 1

                    assert np.sum(mask) != 0

                    class_gt_filtered = CLASS_GT * mask
                    class_gt_filtered = np.bincount(class_gt_filtered.flatten())
                    class_gt_filtered = class_gt_filtered[1:]
                    class_id = np.argmax(class_gt_filtered) + 1
                    # nclass_id = CATE[class_id]
                    classes.append(class_id)
                    
                real_instanceIdx += 1


            # assert len(set(classes)) == 1   # 在当前数据集中是这样的
            
            class_count[classes[-1]] = class_count.get(classes[-1], 0) + 1
            
            ref["raw_sent"] = sent
            ref['inst_ids'] = ins_ids
            ref['ref_id'] = len(data_dict)
            ref['class_ids'] = classes

            data_dict.append(ref)
            print(len(data_dict))

            # break
    
    print("before argumentation：", class_count)      
    # 根据类别做数据增强
    if split == 'train':
        ndata_dict = []
        for ref in data_dict:
            sent = ref['raw_sent']
            
            class_id = ref['class_ids'][-1]
            
            if class_id != 43:
                resample = np.random.randint(max(0, 5000 - class_count[class_id])) // 10
                
                if resample <= 10 and class_count[class_id] <= 20:
                    resample = class_count[class_id]
                
                
            # if "tree" not in sent.split(" ") or "trees" not in sent.split(" "):
                for _ in range(resample): 
                    ndata_dict.append(ref)
                    # class_count[ref['class_ids'][-1]] = class_count.get(ref['class_ids'][-1], 0) + 1
                
                class_count[class_id] = class_count.get(class_id, 0) + resample

            np.random.shuffle(ndata_dict)
            ndata_dict.append(ref)
        
        data_dict = ndata_dict
        del ndata_dict      
    
    
    print("after argumentation：", class_count)   
    
    with open("/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/" + split + "sentence_instance_" + split + "_argumentation.json", 'w') as f:
        json.dump({"train": data_dict}, f, indent=4)
        
          


# combine the bert and clip
def t19():
    import torch
    import clip
    import numpy as np
    
    from PIL import Image
    print("测试CLIP------------------------------")
    print(clip.available_models()) # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_model, preprocess = clip.load("RN101", device=device)

    image_file = [
                "/home/lingpeng/project/SketchyScene-pytorch/data/val/DRAWING_GT/L0_sample2.png",  # complete
                "/home/lingpeng/project/dataset/cat.png",
                "/home/lingpeng/project/dataset/tree.png",
                "/home/lingpeng/project/dataset/sun.png",
                "/home/lingpeng/project/dataset/grass.png",
                "/home/lingpeng/project/dataset/cloud.png"
                ][1]

    image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
    
    #
    import torch
    from transformers import BertTokenizer, BertModel
    from transformers import logging
    logging.set_verbosity_error()
    
    sentence = ["a dog", "a cat", "a cat on the right"]
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    
    tokens = ['[CLS]'] + tokenizer.tokenize(sentence[1]) + ['[SEP]']
    mask_attention = [1 for _ in range(len(tokens))]
    max_len = 15
    while len(tokens) < max_len:
        tokens += ["['PAD']"]
        mask_attention.append(0)

    mask_attention = torch.tensor(mask_attention).to(device).unsqueeze(0)
    
    sent_encode = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).to(device).unsqueeze(0)
    outputs1 = text_model(sent_encode, attention_mask=mask_attention)
    
    pooled1 = outputs1.pooler_output  # [N, 768]
    
    lc = torch.ones((768, 512), dtype=torch.float32).to(device)
    pooled1 = pooled1 @ lc
    print(pooled1.shape)
    
    
    
    text = clip.tokenize(["the cat"]).to(device)
    with torch.no_grad():
        image_features = image_model.encode_image(image).to(pooled1.dtype)  # [N, 512] , 这个是float16的
        print(image_features.shape)
        
        text_features = image_model.encode_text(text).to(pooled1.dtype)
        # print(text_features.shape) # [len(text), 512]
        # normalized features
        image_features_ = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        print(text_features @ image_features_.T)
    
    
    pooled1 /= pooled1.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # print(pooled1.dtype, image_features.dtype)
    
    simi = pooled1 @ image_features.T
    
    print(simi)

import torch
@torch.no_grad()
def t20():
    from transformers import BertModel, RobertaModel
    from transformers import RobertaTokenizerFast, BertTokenizer
    
    # from pytorch_pretrained_bert.tokenization import BertTokenizer as pyBertTokenizer
    

    device = "cpu"
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = pyBertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    # model = RobertaModel.from_pretrained("roberta-base").to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # sentence = ["a dog", "a cat", "a cat on the right"]
    sentence = ["the cat", "the dog", "the left house", "the house on the left", "the cat on the left"]
    
    # inputs = tokenizer("the two sheep on the left of the car", return_tensors="pt")
    tokens = ['[CLS]'] + tokenizer.tokenize(sentence[0]) + ['[SEP]']
    mask_attention = [1 for _ in range(len(tokens))]
    max_len = 15
    while len(tokens) < max_len:
        tokens += ["['PAD']"]
        mask_attention.append(0)

    mask_attention = torch.tensor(mask_attention).to(device).unsqueeze(0)
    
    sent_encode = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).to(device).unsqueeze(0)
    outputs1 = model(sent_encode, attention_mask=mask_attention)
    
    pooled1 = outputs1.pooler_output  # [N, 768]
    
    print(pooled1.shape)
    
    
    tokens = ['[CLS]'] + tokenizer.tokenize(sentence[-1]) + ['[SEP]']
    mask_attention = [1 for _ in range(len(tokens))]
    max_len = 15
    while len(tokens) < max_len:
        tokens += ["['PAD']"]
        mask_attention.append(0)

    mask_attention = torch.tensor(mask_attention).to(device).unsqueeze(0)
    
    sent_encode = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).to(device).unsqueeze(0)
    
    outputs2 = model(sent_encode, attention_mask=mask_attention)
    
    pooled2 = outputs2.pooler_output  # [N, 768]
    
    pooled1 /= pooled1.norm(dim=-1, keepdim=True)
    pooled2 /= pooled2.norm(dim=-1, keepdim=True)
    
    simi = pooled1 @ pooled2.T
    
    print(simi)
    
    
def t21():
    from transformers import BertModel, RobertaModel
    from transformers import RobertaTokenizerFast, BertTokenizer
    
    # from pytorch_pretrained_bert.tokenization import BertTokenizer as pyBertTokenizer
    

    device = "cpu"
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = pyBertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    # model = RobertaModel.from_pretrained("roberta-base").to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # sentence = ["a dog", "a cat", "a cat on the right"]
    sentence = ["the cat",
                "the dog",
                "the left house", 
                "the house on the left",
                "the cat on the left",
                "three thousand", 
                "3000"
                ]
    max_len = 15
    # tokens = []
    # mask_attention = []
    # for sent in sentence:
    #     token = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
    #     mask = [1 for _ in range(len(token))]
    #     while len(token) < max_len:
    #         token += ["['PAD']"]
    #         mask.append(0)
            
    #     tokens.append(token)
    #     mask_attention.append(mask)
        # print(len(mask), len(token))
        
    # mask_attention = torch.tensor(mask_attention).to(device)
    # sent_encode = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).to(device)
    tokenized = tokenizer.batch_encode_plus(sentence,
                    max_length=15,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True)
    
    tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
    outputs = model(tokenizer_input['input_ids'], attention_mask=tokenizer_input['attention_mask'])
    
    simis = [[0 for _ in range(len(sentence))] for _ in range(len(sentence))]
    for i in range(len(sentence)):
        for j in range(len(sentence)):
            pooled1 = outputs.pooler_output[i, :].unsqueeze(0)  # [N, 768]
    
            pooled2 = outputs.pooler_output[j, :].unsqueeze(0)  # [N, 768]
    
            pooled1 /= pooled1.norm(dim=-1, keepdim=True)
            pooled2 /= pooled2.norm(dim=-1, keepdim=True)
            
            simi = pooled1 @ pooled2.T
            simis[i][j] = simi
    
    print(simis)

import torch
@torch.no_grad()
def t22():
    from transformers import BertModel, RobertaModel
    from transformers import RobertaTokenizerFast, BertTokenizer
    
    from pytorch_pretrained_bert.tokenization import BertTokenizer as pyBertTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = pyBertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    # model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model = RobertaModel.from_pretrained("roberta-base").to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    sentence = ["a dog", "a cat", "a cat on the right"]
    
    # inputs = tokenizer("the two sheep on the left of the car", return_tensors="pt")
    tokens = ['[CLS]'] + tokenizer.tokenize(sentence[0]) + ['[SEP]']
    mask_attention = [1 for _ in range(len(tokens))]
    max_len = 15
    while len(tokens) < max_len:
        tokens += ["['PAD']"]
        mask_attention.append(0)

    mask_attention = torch.tensor(mask_attention).to(device).unsqueeze(0)
    
    sent_encode = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).to(device).unsqueeze(0)
    
    num_layers = 1
    outputs = model(sent_encode, attention_mask=mask_attention, output_hidden_states=True)
    
    # outputs has 13 layers, 1 input layer and 12 hidden layers
    encoded_layers = outputs.hidden_states[1:]
    features = None
    features = torch.stack(encoded_layers[-num_layers:], 1).mean(1)
    
    # language embedding has shape [len(phrase), seq_len, language_dim]
    features = features / num_layers

    embedded = features * mask_attention.unsqueeze(-1).float()
    aggregate = embedded.sum(1) / (mask_attention.sum(-1).unsqueeze(-1).float())
    
    
    print(embedded.shape)
    print(encoded_layers[-1].shape)  # 隐向量
    
    print(aggregate.shape) # fianl hidden
    '''
    torch.Size([1, 15, 768])
    torch.Size([1, 15, 768])
    torch.Size([1, 768])

    '''


def visualize_sem_seg_v2(im, predicts, sent, pred_box=None, save_path='', ap=None):
    # im_seg = im.copy()
    masked_image = im.astype(np.uint32).copy()
    
    def generate_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        # hsv = [(i / N, 1, brightness) for i in range(N)]
        hsv = [(0. , 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    def apply_mask(image, mask, color, alpha=1.):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
    
    # im_seg[:, :, 0] += predicts.astype('uint8') * 200
    # im_seg[:, :, 1] += predicts.astype('uint8') * 0
    # im_seg[:, :, 2] += predicts.astype('uint8') * 0
    color = generate_colors(1)[0]
    masked_image = apply_mask(masked_image, predicts, color)
    
    if pred_box:
        y0, x0, y1, x1 = pred_box
        dash_gap_ = 0
        masked_image = cv2.rectangle(masked_image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        
        
        
    plt.imshow(masked_image.astype('uint8'))
    if sent is not None:
        plt.title(sent)
        
    if ap:
        plt.xlabel(str(ap), fontdict={'family': 'Times New Romen', 'size': 12})
        
    if save_path != '':
        plt.savefig(save_path)
        # im_seg_png = Image.fromarray(im_seg, 'RGB')
        # im_seg_png.save(save_path)
    else:
        plt.show()
    
    plt.close()

    

def t23():
    import torch.nn as nn
    class FeatureResizer(nn.Module):
        """
        This class takes as input a set of embeddings of dimension C1 and outputs a set of
        embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
        """

        def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
            super().__init__()
            self.do_ln = do_ln
            # Object feature encoding
            self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
            self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
            self.dropout = nn.Dropout(dropout)

        def forward(self, encoder_features):
            x = self.fc(encoder_features)
            if self.do_ln:
                x = self.layer_norm(x)
            output = self.dropout(x)
            return output
    
    resizer = FeatureResizer(10, 20, 0.1)
    
    x = torch.randn((2, 3, 10))
    
    y = resizer(x)
    print(y.shape)



def t24():
    from transformers import RobertaTokenizerFast, BertTokenizer
    from transformers import BertModel, RobertaModel
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    text_encoder = RobertaModel.from_pretrained('roberta-base')
    
    captions = ['a cat on the left', 'a dog', 'pig']
    
    tokenized = tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt")
    print(tokenized)
    
    encoded_text = text_encoder(**tokenized)
    
    text_features = encoded_text.last_hidden_state
    text_sentence_features = encoded_text.pooler_output 
    
    print(text_features.shape)
    print(text_sentence_features.shape)
   

def t25(split='train'):
    import json
    import os
    import numpy as np
    import scipy.io
    
    ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax
    # copy
    # ref_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data"
    
    semantic_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/CLASS_GT"
    instance_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/INSTANCE_GT"
    
    
    cref_name = "sentence_instance_" + split + ".json"

    cref_file = os.path.join(ref_root, cref_name)

    crefs = json.load(open(cref_file, 'r'))
    
    np.random.shuffle(crefs)
    
    data_dict = []
    
    # 挑出一个小的子数据集
    # 每个类最多500个数据
    # 数据总量最大为500*24 = 12000
    # 其中，每个类数据中保持有位置推断的优先，但最多150个；不涉及位置的最少50个
    class_count = {}
    all_class_count = {}
    class_spatial_count = {}
    
    RELATIVE_DIRECTIONS = ["left front", "front", "right front", "right", "left",
                "left back", "back", "right back"]
    RELATIVE_DIRECTIONS_ = []
    for e in RELATIVE_DIRECTIONS:
        for l in e.split(" "):
            RELATIVE_DIRECTIONS_.append(l)

    DIRECTIONS = ["on the left front of", "in front of", "on the right front of", "on the right of", "on",
                "under", "on the left of", "on the left back of", "behind", "on the right back of"]
    DIRECTIONS_ =  []
    for e in DIRECTIONS:
        for l in e.split(' '):
            DIRECTIONS_.append(l)

    PSEUDO_DIRECTIONS_ = ["around", "among"]

    HORIZONAL_DIRECTIONS = ["leftmost", "left second", "middle", "right second", "rightmost"]
    HORIZONAL_DIRECTIONS_ = []
    for e in HORIZONAL_DIRECTIONS:
        for l in e.split(" "):
            HORIZONAL_DIRECTIONS_.append(l)
            
    VERTICAL_DIRECTIONS = ["topmost", "top second", "middle", "bottom second", "bottommost"]
    VERTICAL_DIRECTIONS_ = []
    for e in VERTICAL_DIRECTIONS:
        for l in e.split(" "):
            VERTICAL_DIRECTIONS_.append(l)
            
    RANK = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
        "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third", "twenty-fourth",
        "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth", "twenty-ninth", "thirtieth"]
    
    image_counts = {}
    image_count = set()
    
    all_images = set()
    
    bad_images = set()
    for cref in crefs:
        # break
        cimage_id = int(cref['key'])
        print(len(data_dict))
        # if cimage_id not in [202]:
        #     continue    

        for k, v in cref['sen_instIdx_map'].items():

            ref = {}
            ref['image_id'] = cimage_id
            # ref['image_id'] = 999193

            sent = k

            ins_ids = v
            
            semantic_name = "sample_" + str(cimage_id) + "_class"
            instance_name = "sample_" + str(cimage_id) + "_instance"
            
            semantic_path = os.path.join(semantic_root, semantic_name)
            instance_path = os.path.join(instance_root, instance_name)
            
            INSTANCE_GT = scipy.io.loadmat(instance_path)['INSTANCE_GT']
            INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
            CLASS_GT = scipy.io.loadmat(semantic_path)['CLASS_GT']  # (750, 750)
            instance_count = np.bincount(INSTANCE_GT.flatten())  # 图中所有的实例 [inst_id: inst_pixel]
            
            instance_count = instance_count[1:]  # e.g. shape=(101,)
            selected_instance_ids_ = [item for item in ins_ids]
            real_instanceIdx = 0
            classes = []
            
            for i in range(instance_count.shape[0]):
                if instance_count[i] == 0:
                    continue
                
                instanceIdx = i + 1

                if real_instanceIdx in ins_ids:
                    selected_instance_ids_.remove(real_instanceIdx)
                    mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
                    mask[INSTANCE_GT == instanceIdx] = 1

                    assert np.sum(mask) != 0

                    class_gt_filtered = CLASS_GT * mask
                    class_gt_filtered = np.bincount(class_gt_filtered.flatten())
                    class_gt_filtered = class_gt_filtered[1:]
                    class_id = np.argmax(class_gt_filtered) + 1
                    # nclass_id = CATE[class_id]
                    classes.append(class_id)
                    
                real_instanceIdx += 1

            class_now = classes[-1]
            all_images.add(cimage_id)
            
            if class_now == 16: # 去除chair类
                bad_images.add(cimage_id)
                continue
            
            all_class_count[classes[-1]] = all_class_count.get(classes[-1], 0) + 1
            
                
            if class_count.get(class_now, 0) > 1000:
                continue
            
            if image_counts.get(cimage_id, 0) > 25:
                continue
            
            first = False
            for e in sent.split(" "):
                if e in ['of', 'the', 'on', 'in', ]:
                    print("无实意词")
                    continue
                
                if e in RELATIVE_DIRECTIONS_ \
                or e in DIRECTIONS_ \
                or e in PSEUDO_DIRECTIONS_ \
                or e in HORIZONAL_DIRECTIONS_ \
                or e in VERTICAL_DIRECTIONS_ \
                or e in RANK:
                    first = True
                    # break
                
                if first:
                    break
            
            ref["raw_sent"] = sent
            ref['inst_ids'] = ins_ids
            ref['ref_id'] = len(data_dict)
            
            if first:
                if class_spatial_count.get(class_now, 0) <= 700:
                    print("-------涉及到位置---------")
                    class_spatial_count[class_now] = class_spatial_count.get(class_now, 0) + 1
                
                    class_count[class_now] = class_count.get(class_now, 0) + 1
                    data_dict.append(ref)
                    image_count.add(cimage_id)
                    image_counts[cimage_id] = image_counts.get(cimage_id, 0) + 1

            else:
                # assert class_count.get(class_now, 0) <= 200
                print("----------------普通数据-----------")
                class_count[class_now] = class_count.get(class_now, 0) + 1
                
                data_dict.append(ref)
                image_count.add(cimage_id)
                image_counts[cimage_id] = image_counts.get(cimage_id, 0) + 1


    
    # with open("/home/lingpeng/project/SketchySceneColorization/Instance_Matching/subdata/sentence_instance_"+split+".json", 'w') as f:
    #     json.dump(data_dict, f, indent=4)
    
    print("#"*20)
    print(len(all_images))
    print(len(data_dict))
    print(len(image_count))
    print(image_counts)
    print(class_count)
    print(class_spatial_count)
    # print(all_class_count)

    print("#"*20)
    print(bad_images)

def t26():
    import torch
    from torch.nn import functional as F
    
    x = torch.randn((2, 10)).unsqueeze(1)
    y = torch.randn((2, 3, 10))
    
    y1 = y.permute(0, 2, 1)
    
    gate = torch.bmm(x, y1).squeeze(1)
    print(gate.shape)
    
    gate = F.sigmoid(gate).unsqueeze(-1)
    print(y.shape)
    y = y + F.relu(torch.mul(gate, y))
    print(y.shape)


def t27():
    from copy import deepcopy
    import numpy as np
    import torch
    from torch import nn

    # from pytorch_pretrained_bert.modeling import BertModel
    from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


    class BertEncoder(nn.Module):
        def __init__(self):
            super(BertEncoder, self).__init__()
            # self.cfg = cfg
            self.bert_name = "bert-base-uncased"
            print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", False)

            if self.bert_name == "bert-base-uncased":
                config = BertConfig.from_pretrained(self.bert_name)
                config.gradient_checkpointing = False
                self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
                self.language_dim = 768
            elif self.bert_name == "roberta-base":
                config = RobertaConfig.from_pretrained(self.bert_name)
                config.gradient_checkpointing = False
                self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
                self.language_dim = 768
            else:
                raise NotImplementedError

            self.num_layers = 1

        def forward(self, x):
            input = x["input_ids"]
            mask = x["attention_mask"]

            if False:
                # with padding, always 256
                outputs = self.model(
                    input_ids=input,
                    attention_mask=mask,
                    output_hidden_states=True,
                )
                # outputs has 13 layers, 1 input layer and 12 hidden layers
                encoded_layers = outputs.hidden_states[1:]
                features = None
                features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

                # language embedding has shape [len(phrase), seq_len, language_dim]
                features = features / self.num_layers

                embedded = features * mask.unsqueeze(-1).float()
                aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

            else:
                # without padding, only consider positive_tokens
                max_len = (input != 0).sum(1).max().item()
                outputs = self.model(
                    input_ids=input[:, :max_len],
                    attention_mask=mask[:, :max_len],
                    output_hidden_states=True,
                )
                # outputs has 13 layers, 1 input layer and 12 hidden layers
                # print(outputs.hidden_states)
                # assert 1 == 0
                encoded_layers = outputs.hidden_states[1:]

                features = None
                features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)  # only last
                # features = outputs.last_hidden_state
                # features = torch.stack([encoded_layers[-1], encoded_layers[0]], 1).mean(1)  # first-last avg
                
                # language embedding has shape [len(phrase), seq_len, language_dim]
                features = features / self.num_layers

                embedded = features * mask[:, :max_len].unsqueeze(-1).float()
                aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

            ret = {
                "aggregate": aggregate,
                "embedded": embedded,
                "masks": mask,
                "hidden": encoded_layers[-1]
            }
            return ret
    
    
    bert = BertEncoder()
    for p in bert.parameters():
        p.requires_grad = False
        
    # print(bert)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_vocab = tokenizer.get_vocab()
    print("#"*20)
    # print(tokenizer_vocab)
    print("#"*20)
    tokenizer_vocab_ids = [item for key, item in tokenizer_vocab.items()]
    # print(tokenizer_vocab_ids)
    captions = ["the cat",
                "the dog",
                "the left house", 
                "the house on the left",
                "the cat on the left",
                "three thousand",
                "3000"
                ]
    tokenized = tokenizer.batch_encode_plus(captions,
                    max_length=256,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True)
    print("#"*20)
    # print(tokenized)
    
    tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
    
    with torch.no_grad():
        language_dict_features = bert(tokenizer_input)
    
    # ret = {
    #             "aggregate": aggregate, # [N, 768]
    #             "embedded": embedded,  # [N, max_len_word_in_this_batch, 768]
    #             "masks": mask,  # [N, 256]
    #             "hidden": encoded_layers[-1]  # [N, max_len_word_in_this_batch, 768]
    #         }
    for k, v in language_dict_features.items():
        print(k)
        print(v.shape)
    
    
    simis = [[0 for _ in range(len(captions))] for _ in range(len(captions))]
    for i in range(len(captions)):
        
        for j in range(len(captions)):
            pooled1 = language_dict_features['aggregate'][i, :].unsqueeze(0)
            pooled2 = language_dict_features['aggregate'][j, :].unsqueeze(0)
            pooled1 /= pooled1.norm(dim=-1, keepdim=True)
            pooled2 /= pooled2.norm(dim=-1, keepdim=True)
            
            simi = pooled1 @ pooled2.T
            simis[i][j] = simi
            
    
    pooled1 = language_dict_features['aggregate'][0, :].unsqueeze(0)
    pooled2 = language_dict_features['aggregate'][1, :].unsqueeze(0)
    
    pooled1 /= pooled1.norm(dim=-1, keepdim=True)
    pooled2 /= pooled2.norm(dim=-1, keepdim=True)
    
    simi = pooled1 @ pooled2.T
    
    print(simi)
    
    print(simis)


def t28():
    from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel
    from copy import deepcopy
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    
    class BertTextCNN(nn.Module):
        def __init__(self, hidden_size=256):
            super(BertTextCNN, self).__init__()
            # self.cfg = cfg
            self.bert_name = "bert-base-uncased"
            print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", False)

            if self.bert_name == "bert-base-uncased":
                config = BertConfig.from_pretrained(self.bert_name)
                config.gradient_checkpointing = False
                self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
                self.language_dim = 768
            elif self.bert_name == "roberta-base":
                config = RobertaConfig.from_pretrained(self.bert_name)
                config.gradient_checkpointing = False
                self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
                self.language_dim = 768
            else:
                raise NotImplementedError

            self.num_layers = 1

            self.dropout = nn.Dropout(0.1)
            self.conv1 = nn.Conv2d(1, hidden_size, (3, 768))
            self.conv2 = nn.Conv2d(1, hidden_size, (4, 768))
            self.conv3 = nn.Conv2d(1, hidden_size, (5, 768))
            
            # for key, value in self.named_parameters(recurse=True):
            #     print(key)
                # if "encoder.layer.11" in key:
                #     print("第12层")
            
        def forward(self, x):
            input = x["input_ids"]
            mask = x["attention_mask"]
            
            max_len = (input != 0).sum(1).max().item()
            sequence_output = self.model(
                input_ids=input[:, :max_len],
                attention_mask=mask[:, :max_len],
                # output_all_encoded_layers=False # 只要最后一层的隐向量表示，其余11层不输出
            )
            sequence_output = sequence_output[-1]
            print(sequence_output.shape) # [N, sen_len, 768]
            out = self.dropout(sequence_output).unsqueeze(1) # [N, 1, sen_len, 768]
            c1 = torch.relu(self.conv1(out).squeeze(3))
            
            print(c1.shape) # [N, 128, sen_len-3+1]
            p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)
            print(p1.shape)
            c2 = torch.relu(self.conv2(out).squeeze(3))
            p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
            c3 = torch.relu(self.conv3(out).squeeze(3))
            p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
            pool = self.dropout(torch.cat((p1, p2, p3), 1))
            
            return pool
            
            
    bert = BertTextCNN()
    for p in bert.parameters():
        p.requires_grad = False
        
    # print(bert)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_vocab = tokenizer.get_vocab()
    print("#"*20)
    # print(tokenizer_vocab)
    print("#"*20)
    # tokenizer_vocab_ids = [item for key, item in tokenizer_vocab.items()]
    # print(tokenizer_vocab_ids)
    captions = ["cat", "dog", "the left house", 
                "the house on the left", "the cat on the left",
                "three thousand", "3000"
                ]
    tokenized = tokenizer.batch_encode_plus(captions,
                    max_length=256,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True)
    print("#"*20)
    # print(tokenized)
    
    tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
    
    with torch.no_grad():
        sent_features = bert(tokenizer_input)        
    
    print(sent_features.shape) 
    
    simis = [[0 for _ in range(len(captions))] for _ in range(len(captions))]
    for i in range(len(captions)):
        
        for j in range(len(captions)):
            pooled1 = sent_features[i, :].unsqueeze(0)
            pooled2 = sent_features[j, :].unsqueeze(0)
            pooled1 /= pooled1.norm(dim=-1, keepdim=True)
            pooled2 /= pooled2.norm(dim=-1, keepdim=True)
            
            simi = pooled1 @ pooled2.T
            simis[i][j] = simi
            
    
    # pooled1 = sent_features[0, :].unsqueeze(0)
    # pooled2 = sent_features[1, :].unsqueeze(0)
    
    # pooled1 /= pooled1.norm(dim=-1, keepdim=True)
    # pooled2 /= pooled2.norm(dim=-1, keepdim=True)
    
    # simi = pooled1 @ pooled2.T
    
    # print(simi)
    
    print(simis)
  

def t29():
    print("#"*20)
    from transformers import CLIPTokenizer, CLIPTextModel
    import torch.nn as nn
    
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    # for key, value in model.named_parameters(recurse=True):
    #     print(key, value.requires_grad)  #n可以训
        
    
    # assert 1 == 0
    
    captions = ["the cat",
                "the dog",
                "the left house", 
                "the house on the left",
                "the cat on the left",
                "three thousand",
                "3000"
                ]
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # inputs = tokenizer(captions, padding=True, return_tensors="pt")

    # outputs = model(**inputs, output_hidden_states=True)
    
    # encoded_layers = outputs.hidden_states[1:]
    # print(len(encoded_layers))
    
    # last_encoded_layer = encoded_layers[-1]
    # print(last_encoded_layer.shape)
    
    
    class CLIPEncoder(nn.Module):
        def __init__(self):
            super(CLIPEncoder, self).__init__()
            # self.cfg = cfg
            self.model_name = "openai/clip-vit-base-patch32"
            print("build clip encoder: openai/clip-vit-base-patch32")

            self.model = CLIPTextModel.from_pretrained(self.model_name)
            

        def forward(self, x):
            input = x["input_ids"]
            mask = x["attention_mask"]
            
            max_len = (input != 0).sum(1).max().item()
            outputs = self.model(
                input_ids=input[:, :max_len],
                attention_mask=mask[:, :max_len],
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            # print(outputs.hidden_states)
            # assert 1 == 0
            encoded_layers = outputs.hidden_states[1:]

            features = None
            # features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)  # only last
            features = outputs.last_hidden_state  # CLIP和BERT差异：CLIP的last_hidden会过最后一个LN，而hidden_states都不会；BERT中的last_hidden则就是最后一个hidden_states
            # features = outputs.pooler_output
            # features = torch.stack([encoded_layers[-1], encoded_layers[0]], 1).mean(1)  # first-last avg
            # ret = {
            #     "aggregate": features,
            #     "embedded": None,
            #     "masks": mask,
            #     "hidden": encoded_layers[-1]
            # }
            # return ret
            # language embedding has shape [len(phrase), seq_len, language_dim]
            # features = features / self.num_layers

            embedded = features * mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

            ret = {
                "aggregate": aggregate,
                "embedded": embedded,
                "masks": mask,
                "hidden": encoded_layers[-1]
            }
            return ret
        
    
    model = CLIPEncoder()
    for p in model.parameters():
        p.requires_grad = False
        
    tokenized = tokenizer.batch_encode_plus(captions,
                    max_length=77,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True)
    
    # print(tokenized)
    # assert 1 == 0
    tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
    
    with torch.no_grad():
        language_dict_features = model(tokenizer_input)
    
    # for k, v in language_dict_features.items():
    #     print(k)
    #     print(v.shape)
    
    
    simis = [[0 for _ in range(len(captions))] for _ in range(len(captions))]
    for i in range(len(captions)):
        
        for j in range(len(captions)):
            pooled1 = language_dict_features['aggregate'][i, :].unsqueeze(0)
            pooled2 = language_dict_features['aggregate'][j, :].unsqueeze(0)
            pooled1 /= pooled1.norm(dim=-1, keepdim=True)
            pooled2 /= pooled2.norm(dim=-1, keepdim=True)
            
            simi = pooled1 @ pooled2.T
            simis[i][j] = simi
            
    
    pooled1 = language_dict_features['aggregate'][0, :].unsqueeze(0)
    pooled2 = language_dict_features['aggregate'][1, :].unsqueeze(0)
    
    pooled1 /= pooled1.norm(dim=-1, keepdim=True)
    pooled2 /= pooled2.norm(dim=-1, keepdim=True)
    
    simi = pooled1 @ pooled2.T
    
    print(simi)
    
    print(simis)



def t30():
    print("#"*20)
    from copy import deepcopy
    import numpy as np
    import torch
    from torch import nn

    # from pytorch_pretrained_bert.modeling import BertModel
    from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


    class RobertaEncoder(nn.Module):
        def __init__(self):
            super(RobertaEncoder, self).__init__()
            # self.cfg = cfg
            self.bert_name = "roberta-base"
            print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", False)

            if self.bert_name == "bert-base-uncased":
                config = BertConfig.from_pretrained(self.bert_name)
                config.gradient_checkpointing = False
                self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
                self.language_dim = 768
            elif self.bert_name == "roberta-base":
                config = RobertaConfig.from_pretrained(self.bert_name)
                config.gradient_checkpointing = False
                self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
                self.language_dim = 768
            else:
                raise NotImplementedError

            self.num_layers = 1

        def forward(self, x):
            input = x["input_ids"]
            mask = x["attention_mask"]

            if False:
                # with padding, always 256
                outputs = self.model(
                    input_ids=input,
                    attention_mask=mask,
                    output_hidden_states=True,
                )
                # outputs has 13 layers, 1 input layer and 12 hidden layers
                encoded_layers = outputs.hidden_states[1:]
                features = None
                features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

                # language embedding has shape [len(phrase), seq_len, language_dim]
                features = features / self.num_layers

                embedded = features * mask.unsqueeze(-1).float()
                aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

            else:
                # without padding, only consider positive_tokens
                max_len = (input != 0).sum(1).max().item()
                outputs = self.model(
                    input_ids=input[:, :max_len],
                    attention_mask=mask[:, :max_len],
                    output_hidden_states=True,
                )
                # outputs has 13 layers, 1 input layer and 12 hidden layers
                # print(outputs.hidden_states)
                # assert 1 == 0
                encoded_layers = outputs.hidden_states[1:]

                features = None
                features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)  # only last
                # features = outputs.last_hidden_state
                # features = torch.stack([encoded_layers[-1], encoded_layers[0]], 1).mean(1)  # first-last avg
                
                # language embedding has shape [len(phrase), seq_len, language_dim]
                features = features / self.num_layers

                embedded = features * mask[:, :max_len].unsqueeze(-1).float()
                aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

            ret = {
                "aggregate": aggregate,
                "embedded": embedded,
                "masks": mask,
                "hidden": encoded_layers[-1]
            }
            return ret
    
    
    bert = RobertaEncoder()
    for p in bert.parameters():
        p.requires_grad = False
        
    # print(bert)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # tokenizer_vocab = tokenizer.get_vocab()
    print("#"*20)
    # print(tokenizer_vocab)
    print("#"*20)
    # tokenizer_vocab_ids = [item for key, item in tokenizer_vocab.items()]
    # print(tokenizer_vocab_ids)
    captions = ["the cat",
                "the dog",
                "the left house", 
                "the house on the left",
                "the cat on the left",
                "three thousand",
                "3000"
                ]
    tokenized = tokenizer.batch_encode_plus(captions,
                    max_length=256,
                    padding='max_length',
                    return_special_tokens_mask=True,
                    return_tensors='pt',
                    truncation=True)
    print("#"*20)
    # print(tokenized)
    
    tokenizer_input = {"input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask}
    
    with torch.no_grad():
        language_dict_features = bert(tokenizer_input)
    
    # ret = {
    #             "aggregate": aggregate, # [N, 768]
    #             "embedded": embedded,  # [N, max_len_word_in_this_batch, 768]
    #             "masks": mask,  # [N, 256]
    #             "hidden": encoded_layers[-1]  # [N, max_len_word_in_this_batch, 768]
    #         }
    for k, v in language_dict_features.items():
        print(k)
        print(v.shape)
    
    
    simis = [[0 for _ in range(len(captions))] for _ in range(len(captions))]
    for i in range(len(captions)):
        
        for j in range(len(captions)):
            pooled1 = language_dict_features['aggregate'][i, :].unsqueeze(0)
            pooled2 = language_dict_features['aggregate'][j, :].unsqueeze(0)
            pooled1 /= pooled1.norm(dim=-1, keepdim=True)
            pooled2 /= pooled2.norm(dim=-1, keepdim=True)
            
            simi = pooled1 @ pooled2.T
            simis[i][j] = simi
            
    
    pooled1 = language_dict_features['aggregate'][0, :].unsqueeze(0)
    pooled2 = language_dict_features['aggregate'][1, :].unsqueeze(0)
    
    pooled1 /= pooled1.norm(dim=-1, keepdim=True)
    pooled2 /= pooled2.norm(dim=-1, keepdim=True)
    
    simi = pooled1 @ pooled2.T
    
    print(simi)
    
    print(simis)


    
if __name__ == "__main__":
    # t1()
    # t2()
    # t3()
    # t4()
    # t5()
    # t6()
    # t7()
    # t8()  
    # t9()     
    # t10() 
    # t11()
    # t12()
    # t13()
    # t14()
    # t15()
    # t16()
    # t17()
    # t18()
    # t19()
    # t20()
    # t21()
    # t22()
    # t23()
    # t24()
    # t25("test")
    # t26()
    # t27()
    # t28()
    t29()
    # t30()



