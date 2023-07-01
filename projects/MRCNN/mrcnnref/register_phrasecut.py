import os.path as osp
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
import numpy as np
import cv2

from .phrasecut.file_paths import name_att_rel_count_fpath
from .phrasecut.refvg_loader import RefVGLoader
from .phrasecut.data_transfer import polygons_to_mask

def register_phrasecut(split='train'):
    print("注册", split)

    info_account = json.load(open(name_att_rel_count_fpath, 'r'))['cat']
    infos = {}
    cates = {}
    for item in info_account:
        c, v = item[0], item[1]

        infos[c] = v

        cates[c] = len(cates) + 1

    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
    
    refvg_loader = RefVGLoader(split=split)
    
    # print(len(refvg_loader.img_ids))

    data_dict = []
    max_cate = 0
    instance_count = {}
    for img_id in refvg_loader.img_ids:
        if img_id in [2326771, 2402732, 2402732, ]:
            continue

        # if img_id != 3368:
        #     continue

        ref = refvg_loader.get_img_ref_data(img_id)

        # for ref in refs:
        # print(ref)
        nref = {}
        nref['image_id'] = img_id


        
        nref['height'] = ref['height']
        nref['width'] = ref['width']
        nref['split'] = ref['split']
        nref['bounds'] = ref['bounds']

        
        if 'gt_Polygons' in ref:
            # print(ref['img_ins_cats'])
            # assert 1 == 0
            for task_i, task_id in enumerate(ref['task_ids']):
                
                ps = ref['p_structures'][task_i]
                # assert ps['type'] == 'name'

                cate_name = ps['name']
                if "line" in cate_name or "handles" in cate_name:  # 去掉所有line类
                    continue
                # if cate_name not in infos or infos[cate_name] <= 20:  # 控制类别：1272
                if cate_name not in infos or infos[cate_name] < 731:
                    continue

                tnref = nref.copy()
                # print(ref)

                # assert 1 == 0

                tnref['task_id'] = task_id
                tnref['phrase']= ref['phrases'][task_i]
                print(tnref['phrase'])
                tnref['p_structure'] = ref['p_structures'][task_i]
                tnref['gt_boxes'] = ref['gt_boxes'][task_i]
                tnref['gt_Polygons'] = ref['gt_Polygons'][task_i]

                flag = True
                for p in ref['gt_Polygons'][task_i]:
                    break
                    mask = polygons_to_mask(p, ref['width'], ref['height']) + 0
                    # print(m)
                    # assert 1 == 0
                    # mask = m.astype(np.uint8)
                    # mask = cv2.resize(mask.astype(np.float32), (768, 768)).astype(mask.dtype)
            
                    # mask[mask > 1] = 0
                    if not (mask.sum() > 0):
                        flag = False
                        break 
                if not flag:
                    print("bed data:", str(img_id), str(task_id), cate_name)
                    continue

                tnref["class_ids"] = cates[cate_name]
                tnref['class_name'] = cate_name
                max_cate = max(max_cate, cates[cate_name])
                data_dict.append(tnref)

                instance_count[len(ref['gt_boxes'][task_i])] = instance_count.get(len(ref['gt_boxes'][task_i]), 0) + 1

        else:
            raise      

        # data_dict.append(ref)
    # assert 1 == 0
    print("max cate:", max_cate)
    print("data for " + split + ": ", len(data_dict))
    print("实例数目统计:", instance_count) 
    # {1: 256674, 2: 34226, 3: 9195, 4: 3762, 6: 634, 7: 281, 8: 179, 5: 1337, 9: 75, 10: 59, 
    # 21: 1, 12: 25, 11: 24, 13: 16, 15: 5, 26: 1, 19: 3, 22: 1, 16: 8, 14: 7, 37: 1, 17: 2, 18: 3, 20: 1, 24: 1, 25: 1, 23: 1, 29: 1, 28: 1}
    assert 1 == 0

    return data_dict



def register_phrasecut_all():
    print("注册phrasecut数据集")
    # assert 1 == 0
    # split = 'val'
    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"

    # cref_name = "sentence_instance_" + split + ".json"

    # cref_file = os.path.join(ref_root, cref_name)

    # DatasetCatalog.register("sketch_" + 'val', lambda: register_sketch(split=split))
    # MetadataCatalog.get('sketch_' + 'val').set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["ball"])

    
    split = 'train'
    # ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
    ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax


    cref_name = "sentence_instance_" + split + ".json"

    cref_file = os.path.join(ref_root, cref_name)

    DatasetCatalog.register("phrasecut_" + "train", lambda: register_phrasecut(split="train"))
    MetadataCatalog.get('phrasecut_' + "train").set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["phrasecut"])

    split = 'test'
    ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"  # share
    ref_root = "/home/lingpeng/project/SketchySceneColorization/Instance_Matching/data/"  # only cmax
    
    cref_name = "sentence_instance_" + split + ".json"

    cref_file = os.path.join(ref_root, cref_name)

    # DatasetCatalog.register("phrasecut_" + "test", lambda: register_phrasecut(split=split))
    # MetadataCatalog.get('phrasecut_' + "test").set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["phrasecut"])

    # for split in ['train']:
    #     ref_root = "/nfs/SketchySceneColorization/Instance_Matching/data/"

    #     cref_name = "sentence_instance_" + split + ".json"

    #     cref_file = os.path.join(ref_root, cref_name)
    #     DatasetCatalog.register("sketch_" + split, lambda: register_sketch(split=split))
    #     MetadataCatalog.get('sketch_' + split).set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["sketch"])

   
register_phrasecut_all()