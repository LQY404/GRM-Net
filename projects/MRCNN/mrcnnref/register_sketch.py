import os.path as osp
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
import numpy as np
import scipy

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

def register_sketch_train():
    split = "train"
    
    ref_data = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/" + split + "sentence_instance_" + split + "_argumentation.json"
    
    crefs = json.load(open(ref_data, 'r'))['train']
    
    return crefs
    

def register_sketch(split='train'):
    # assert 1 == 0
    print("注册", split)
    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
    ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax
    # copy
    # ref_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data"
    
    semantic_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/CLASS_GT"
    instance_root = "/home/lingpeng/project/SketchyScene-pytorch/data/" + split + "/INSTANCE_GT"
    
    
    cref_name = "sentence_instance_" + split + ".json"

    cref_file = os.path.join(ref_root, cref_name)

    crefs = json.load(open(cref_file, 'r'))
    # print(len(crefs))
    # assert 1 == 0
    # return crefs
    # np.random.shuffle(crefs)
    data_dict = []

    class_count = {}
    # if split != 'train':
    #     crefs = crefs[: 500]
    
    # bad_images = [1218, 4518, 5607, 5609, 4715, 1259, 4157, 892, 142, 80, 882, 594, 220]
    for cref in crefs:
        break
        cimage_id = int(cref['key'])

        # if cimage_id in bad_images:
        #     continue
        
        # if cimage_id not in [202]:
        #     continue    

        for k, v in cref['sen_instIdx_map'].items():

            ref = {}
            ref['image_id'] = cimage_id
            # ref['image_id'] = 999193

            sent = k

            ins_ids = v
            
            if split == 'train':
                ref["raw_sent"] = sent
                ref['inst_ids'] = ins_ids
                ref['ref_id'] = len(data_dict)
                # ref['class_ids'] = classes

                data_dict.append(ref)
                # print(len(data_dict))
                
                continue
            
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
    if split == 'train' and 1==0:
        ndata_dict = []
        for ref in data_dict:
            # sent = ref['raw_sent']
            
            class_id = ref['class_ids'][-1]
            
            # if class_id != 43:
            resample = np.random.randint(max(1, 1000 - class_count[class_id])) // 20
                
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
    if len(data_dict) == 0:
        data_dict.append({
            'image_id': 142,
            'ref_id': 0,
            'raw_sent': ""
        })

    print("data for " + split + ": ", len(data_dict))
    # for d in data_dict:
    #    image_id = d['image_id']

    #    assert image_id != 
    # if split == 'train':
    #     np.random.shuffle(data_dict)
    
    
    return data_dict


def register_sketch_1000(split='train'):
    print("注册", split)
    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
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
            
            if split != 'train':
                ref["raw_sent"] = sent
                ref['inst_ids'] = ins_ids
                ref['ref_id'] = len(data_dict)
                # ref['class_ids'] = classes

                data_dict.append(ref)
                print(len(data_dict))
                
                continue
            
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
                
                if len(classes) > 0: # 只需要一个就可以了
                    break
                

            class_count[classes[-1]] = class_count.get(classes[-1], 0) + 1
            ref["raw_sent"] = sent
            ref['inst_ids'] = ins_ids
            ref['ref_id'] = len(data_dict)
            ref['class_ids'] = classes

            data_dict.append(ref)
            print(len(data_dict))

    print("before argumentation：", class_count)      
    # 根据类别做数据增强
    fdata_dict = []
    fclass_count = {}
    
    data_dict_category = dict()
    for k, v in SKETCH_CLASS_NAME.items():
        data_dict_category[k] = []
        
        
    if split != 'train':
        ndata_dict = []
        for ref in data_dict:
            sent = ref['raw_sent']
            class_id = ref['class_ids'][-1]
            
            data_dict_category[class_id].append(ref)
            # 重采样子集规则：1）原本就多于1000的类别数据中涉及到方位的优先，不足使用非方位数据补充
            # 2）其他类，先全部放入，后面再循环一遍采样直到满足1000个即可
            
            # 情况2）
            if class_id not in [9, 14, 18, 22, 27, 29, 32, 41, 43]:
                fdata_dict.append(ref)
                fclass_count[class_id] = fclass_count.get(class_id, 0) + 1
                continue
            # 情况1）
            if fclass_count.get(class_id, None) is not None and fclass_count[class_id] > 500:
                continue
            
            flag = False
            tokens = sent.split(' ')
            for t in ['left', 'right', 'bottom', 'top', 'middle']:
                if t in tokens:
                    flag = True
                    break
            
            if flag:
                fdata_dict.append(ref)
                fclass_count[class_id] = fclass_count.get(class_id, 0) + 1
        
        # 再采样，按照类别
        for k, v in fclass_count.copy().items():
            if v > 1000: # 已满足
                continue
            
            supply = 1000 - v + 1
            # 情况1）
            if class_id in [9, 14, 18, 22, 27, 29, 32, 41, 43]: # 随机选取
                dt = data_dict_category[k]
                np.random.shuffle(dt)
                
                for d in dt[: supply]:
                    fdata_dict.append(d)
                fclass_count[k] = fclass_count.get(k, 0) + supply
                
                
            else: # 采样
                dt = data_dict_category[k]
                np.random.shuffle(dt)
                
                for d in dt:
                    resample = np.random.randint(max(0, 1000 - supply)) // 5
                    
                    fclass_count[k] = fclass_count.get(k, 0) + resample
                    supply -= resample
                    
                    for _ in range(resample):
                        fdata_dict.append(d)
                        
                    if supply < 0:
                        break
    
    
    print("after argumentation：", fclass_count)      
                  
    print("data for " + split + ": ", len(fdata_dict))
    # assert 1 == 0
    # for d in data_dict:
    #    image_id = d['image_id']

    #    assert image_id != 
    # np.shu
    return fdata_dict           
                
        
    

def register_sketch_all():
    print("注册sketch数据集")

    # split = 'val'
    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"

    # cref_name = "sentence_instance_" + split + ".json"

    # cref_file = os.path.join(ref_root, cref_name)

    # DatasetCatalog.register("sketch_" + 'val', lambda: register_sketch(split=split))
    # MetadataCatalog.get('sketch_' + 'val').set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["ball"])

    
    split = 'train'
    split = 'test'
    # split = 'val'
    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
    # ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax
    # copy
    # ref_root = "/home/lingpeng/project/dataset/sketchyscene_copy1/data"


    # cref_name = "sentence_instance_" + split + ".json"

    # cref_file = os.path.join(ref_root, cref_name)

    # DatasetCatalog.register("sketch_" + "train", lambda: register_sketch(split=split))
    # DatasetCatalog.register("sketch_" + "train", lambda: register_sketch_train())
    # MetadataCatalog.get('sketch_' + "train").set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["sketch"])
    
    if split == 'train':
        ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax
        cref_name = "sentence_instance_" + split + ".json"

        cref_file = os.path.join(ref_root, cref_name)
        DatasetCatalog.register("sketch_" + "train", lambda: register_sketch())
        MetadataCatalog.get('sketch_' + "train").set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["sketch"])
    
    else:
    # split = 'test'
        ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
        ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax
        
        cref_name = "sentence_instance_" + split + ".json"

        cref_file = os.path.join(ref_root, cref_name)

        DatasetCatalog.register("sketch_" + 'test', lambda: register_sketch(split=split))
        MetadataCatalog.get('sketch_' + 'test').set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["sketch"])

    # for split in ['train']:
    #     ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"

    #     cref_name = "sentence_instance_" + split + ".json"

    #     cref_file = os.path.join(ref_root, cref_name)
    #     DatasetCatalog.register("sketch_" + split, lambda: register_sketch(split=split))
    #     MetadataCatalog.get('sketch_' + split).set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["sketch"])

   
register_sketch_all()