import os.path as osp
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
import numpy as np


bad_images = {
    'train': [6406, 7583, 9866, 10229, 13720, 14128, 14138, 15485, 18467, 21451, 22014, 27769, 28766,
                44830, 46065, 54534, 54614, 54764, 57250, 58231, 60874, 65842, 67573, 68176, 69257, 75361, 77005, 78425, 78827,
                79822, 90328, 90985, 93581, 98304, 105620, 112040, 114142, 123336, 124893, 
                132874, 136962, 139617, 143501, 147342, 149078, 151729, 164286, 166594, 
                169495, 170980, 177915, 181475, 183445, 183785, 190646, 208513, 213282, 
                217779, 220037, 223664, 227512, 232254, 235651, 237617, 238502, 243645, 
                246779, 250234, 252373, 252751, 254938, 260715, 261235, 270186, 278467, 
                285890, 292226, 294837, 295864, 297637, 299051, 304047, 305215, 307322,
                309260, 315521, 316663, 317833, 318528, 318638, 320081, 322239, 323553,
                324012, 324381, 327205, 329018, 330652, 333075, 335955, 338971, 339115,
                339529, 339816, 341429, 349947, 353701, 356011, 359201, 361197, 364006,
                367934, 382102, 386603, 387264, 387919, 390474, 392703, 393297, 395684,
                395863, 402235, 407869, 408048, 409678, 412112, 412194, 422343, 424381,
                429865, 432053, 433240, 434201, 434699, 434884, 436295, 438101, 438331,
                443916, 450305, 451337, 452014, 455166, 458682, 460986, 461561, 464854,
                464937, 465825, 466097, 468615, 471905, 474256, 474963, 482195, 484103,
                484201, 484946, 490665, 498436, 506378, 506812, 507952, 511751, 518455,
                518509, 518954, 519607, 522365, 523564, 537188, 537532, 544449, 545844,
                547533, 548958, 552336, 554619, 557987, 559618, 561875, 562679, 566899,
                568117, 571656, 572949, 578734, 579255, 581136, 
                ]
}

bad_img = []
with open("/nfs/crefs/refcoco/bad_image_refcoco.txt", 'r') as f:
    for l in f.readlines():
        l = l[: -1]
        b = int(l)
        bad_img.append(b)

def register_refcoco_unc(split='val'):
    split = split
    # ref_root = "/home/lingpeng/project/demo/data/"
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
    
    Anns = {}
    for ann in data['annotations']:
        Anns[ann['id']] = ann

    Refs = {}
    for ref in data['refs']:
        ref_id = ref['ref_id']
        ann_id = ref['ann_id']

        if ref_type != 'refclef' and Anns[ann_id]['iscrowd'] == 1:
            continue

        Refs[ref_id] = ref

     # 再导入组合后的数据，注意的是，我们生成的数据为列表，列表的值为列表，长度为2/3或者4，都是原数据的ref_id
    # gref_name = "ref_res_" + ref_type + "_" + ref_split

    # grefs = json.load(open("/home/lingpeng/project/iep-ref-master/" + gref_name + '.json', 'r'))[ref_type]
    # gref_len = len(grefs)  # 所有的数据为9779，训练用7823，测试用9779 - 7823 = 1956

    # if split == 'train':
    #     grefs = grefs[: int(gref_len*0.8)]
    # else:
    #     grefs = grefs[int(gref_len*0.8): ]

    # cref_name = "picked_c_" + ref_type + '_' + ref_split + '_' + split + '_v2.json'
    cref_name = "picked_c_" + ref_type + '_' + ref_split + '_' + split + '.json'
    
    # cref_file = "/nfs/demo/data/" + ref_type + '/' + cref_name
    cref_file = "/nfs/crefs/" + ref_type + '/' + cref_name
    crefs = json.load(open(cref_file, 'r'))['refs']
    

    print("当前时" + str(split) + "模式，用到的数据量为：" + str(len(crefs)))
    # print("for " + str(split) + ": " + str(gref_len))
    # assert 1 == 0
    if split == 'train':
        np.random.shuffle(crefs)

    max_data_len = 50000

    for cref in crefs:
        cref_id = list(cref.values())[-1][-1]  #随便取一个都行，都是同一幅图像
        cimage_id = Refs[cref_id]['image_id']

        # assert cimage_id not in bad_images['train']
        # if cimage_id in bad_images['train']:
        if cimage_id in bad_img or cimage_id in bad_images['train']:
            continue

        # if int(cimage_id) != 458969:  #用于自己输入测试
        #     continue

        data_dict.append(cref) # key: values,e.g., cref_id: [ref_id1, ref_id2]

        # if len(data_dict) > max_data_len:
        #     break
    print("all data: ", len(data_dict))
    return data_dict


def register_refcoco_unc_all():
    print("注册refcoco_unc数据集")
    
    split = 'train'
    # ref_root = "/home/lingpeng/project/demo/data/"
    ref_root = "/nfs/demo/data/"
    ref_type = ["refcoco", "refcoco+", "refcocog", "refclef"][0]
    ref_split = {
        "refcoco": ["unc", "google"],
        "refcoco+": ["unc"],
        "refcocog": ['umd', "google"],
        "refclef": ['unc', "berkeley"]
    }[ref_type][0]

    # cref_name = "picked_c_" + ref_type + '_' + ref_split + '_' + split + '_v2.json'
    cref_name = "picked_c_" + ref_type + '_' + ref_split + '_' + split + '.json'
    
    # cref_file = "/nfs/demo/data/" + ref_type + "/" + cref_name
    cref_file = "/nfs/crefs/" + ref_type + '/' + cref_name


    DatasetCatalog.register("refcoco_unc_" + "train", lambda: register_refcoco_unc(split=split))
    MetadataCatalog.get('refcoco_unc_' + "train").set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["ball"])


    # split = 'val'
    # # cref_name = "picked_c_" + ref_type + '_' + ref_split + '_' + split + '_v2.json'
    # cref_name = "picked_c_" + ref_type + '_' + ref_split + '_' + split + '.json'

    # # cref_file = "/nfs/demo/data/" + ref_type + "/" + cref_name
    # cref_file = "/nfs/crefs/" + ref_type + '/' + cref_name

    # DatasetCatalog.register("refcoco_unc_" + 'val', lambda: register_refcoco_unc(split=split))
    # MetadataCatalog.get('refcoco_unc_' + 'val').set(json_file=cref_file, evaluator_type="refcoco", thing_classes=["ball"])
        
register_refcoco_unc_all()

