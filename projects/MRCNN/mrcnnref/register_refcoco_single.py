import os.path as osp
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
import numpy as np


def register_data(split='train'):

    # pass
    ref_root = "/nfs/demo/data/"
    ref_type = ["refcoco", "refcoco+", "refcocog", "refclef"][0]
    ref_split = {
        "refcoco": ["unc", "google"],
        "refcoco+": ["unc"],
        "refcocog": ['umd', "google"],
        "refclef": ['unc', "berkeley"]
    }[ref_type][0]

    data_dict = []

    ref_path = os.path.join(ref_root, ref_type)
    ref_file = os.path.join(ref_path, "refs(" + ref_split + ").p")
    instances_file = os.path.join(ref_path, "instances.json")

    refs = pickle.load(open(ref_file, 'rb'))
    instances = json.load(open(instances_file, 'r'))

    data = {}
    data['refs'] = refs
    data['annotations'] = instances['annotations']
    data['images'] = instances['images']
    
    Anns = {}


    Imgs = {}
    for img in data['images']:
        Imgs[img['id']] = img

    for ann in data['annotations']:
        Anns[ann['id']] = ann

        if ann['iscrowd'] == 1:
            continue
        # dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
        # print(ann.keys())
        # print(ann)
        # break
        # assert 1 == 0
        # category_dict[ann['category_id']] = category_dict.get(ann['category_id'], 0) + 1

    Refs = {}

    category_dict = {}
    ann_ids = set()

    
    splits = set()
    for ref in data['refs']:
        splits.add(ref['split'])
        if ref['split'] != split:
            continue

        ref_id = ref['ref_id']
        ann_id = ref['ann_id']

        # if ann_id not in Anns.keys() or (ref_type != 'refclef' and Anns[ann_id]['iscrowd'] == 1):
        #     continue

        # print(ref.keys())
        # print(ref)
        # file_name 就是 image_name
        # dict_keys(['sent_ids', 'file_name', 'ann_id', 'ref_id', 'image_id', 'split', 'sentences', 'category_id'])
        # sentences是一个列表，每个元素是dict：tokens, raw, sent_id
        # 这些都是指代同一个实例
        # assert 1 == 0
        # if ref['ann_id'] in ann_ids:
        #     continue

        ann_ids.add(ref['ann_id'])
        category_dict[ref['category_id']] = category_dict.get(ref['category_id'], 0) + len(ref['sent_ids'])
        Refs[ref_id] = ref

    dataset_dict = []

    for ref in data['refs']:
        file_name = ref['file_name']
        ann_id = ref['ann_id']
        image_id = ref['image_id']
        category_id = ref['category_id']
        ref_id = ref['ref_id']


        if ref['split'] != split:
            continue

        for i, gref_id in enumerate(ref['sent_ids']):
            dataset_dict.append({
                'cref_id': len(dataset_dict),
                'ref_id': ref_id,
                'sent_id': gref_id,
                'ann': Anns[ann_id],
                "file_name": Imgs[image_id]['file_name'],
                'image_id': image_id,
                'category_id': category_id,
                "sentence": ref['sentences'][i],
            })
    
    print(len(dataset_dict)) # 120000
    print(splits)
    # assert 1 == 0
    return dataset_dict


def register_all_data():

    data_root = "/nfs/demo/data/"


    split = 'train'
    # register_data(split=split)
    DatasetCatalog.register("refcocos_" + "train", lambda: register_data(split=split))
    MetadataCatalog.get('refcocos_' + "train").set(json_file=data_root, evaluator_type="refcoco", thing_classes=["sketch"])
    


    # split = 'test'  # testA, testB, val


register_all_data()