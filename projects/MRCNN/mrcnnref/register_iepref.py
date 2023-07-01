import os.path as osp
import json
from detectron2.data import DatasetCatalog, MetadataCatalog


img_val = json.load(open("/nfs/iep-ref-master/data/referring_rubber/refexps/img_val.txt", 'r'))['img_id']




def register_iep_refs(root="/nfs/iep-ref-master/data/", split='train'):


    scene_file = osp.join(root, 'clevr_ref+_1.0/scenes/clevr_ref+_' + split + '_scenes.json')
    refexp_file = osp.join(root, 'referring_rubber/refexps/ref_picked_' + split + '.json')

    
    scenes = json.load(open(scene_file))['scenes']
    exps = json.load(open(refexp_file))['refexps']
    if split != 'train':
        exps = exps[: 1000]
    

    dataset_dicts = []

    # 根据shape、size以及color三种属性组合生成不同的；类标签
    # shape = ["cube", "sphere", "cylinder"]
    # size = ["small", "large"]
    # color = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    category_ids = {  # 总共48个类，不包括背景
        'cube': {
            'small': {
                'gray': 1,
                'red': 2,
                'blue': 3,
                'green': 4,
                'brown': 5,
                'purple': 6,
                'cyan': 7,
                'yellow': 8

            },
            'large': {
                'gray': 9,
                'red': 10,
                'blue': 11,
                'green': 12,
                'brown': 13,
                'purple': 14,
                'cyan': 15,
                'yellow': 16
            }
        },
        'sphere': {
            'small': {
                'gray': 17,
                'red': 18,
                'blue': 19,
                'green': 20,
                'brown': 21,
                'purple': 22,
                'cyan': 23,
                'yellow': 24

            },
            'large': {
                'gray': 25,
                'red': 26,
                'blue': 27,
                'green': 28,
                'brown': 29,
                'purple': 30,
                'cyan': 31,
                'yellow': 32

            }
        },
        'cylinder': {
            'small': {
                'gray': 33,
                'red': 34,
                'blue': 35,
                'green': 36,
                'brown': 37,
                'purple': 38,
                'cyan': 39,
                'yellow': 40

            },
            'large': {
                'gray': 41,
                'red': 42,
                'blue': 43,
                'green': 44,
                'brown': 45,
                'purple': 46,
                'cyan': 47,
                'yellow': 48

            }
        }
    }

    bad_scene = [10863, 16798, 18413, 19808, 29842, 36319, 39246, 40540, 41063, 59913, 63899] if split == 'train' else [5257, 9156, 9535, 9994, 10132]
    
    rubber_scenes = []

    for qq, scene in enumerate(scenes):
        if scene['image_index'] in bad_scene:
            print("bad scene!!!!")
            continue
        
        obj_infos = scene['objects']
        flag = True
        for obj in obj_infos:
            material = obj["material"]

            if material == 'metal':
                flag = False
                break

        if flag:
            rubber_scenes.append(scene['image_index'])


    imgid2scene = {}
    for scene in scenes:
        if scene["image_index"] in bad_scene:
            continue
        
        if scene['image_index'] not in rubber_scenes:
            continue

        imgid2scene[scene["image_index"]] = scene

    imgval = 0
    for ref in exps:
        if ref["image_index"] in bad_scene:
            continue
        

        if ref['image_index'] not in rubber_scenes:
            continue

        if split=="train" and ref["image_index"] in img_val:
            print("训练数据可能在测试集中")
            imgval += 1
            continue


        # if ref['image_index'] not in [666]:  # 自己输入测试用
        #     continue

        scene = imgid2scene[ref["image_index"]]

        record = {}
        record["image_index"] = ref["image_index"]
        record["image_filename"] = ref["image_filename"]
        record["refexp"] = ref["refexp"]

        objlist = ref["program"][-1]["_output"]
        objlist = [e+1 for e in objlist]
        record["objlist"] = objlist.copy()

        # obj_bboxes = {}
        # obj_masks = {}
        record["obj_mask"] = {}
        record["obj_bbox"] = {}
        record['category_id'] = {}
        record['cref_id'] = ref["refexp_index"]

        all_boxs = scene["obj_bbox"]
        all_masks = scene["obj_mask"]

        for obj_id in objlist:
        # for obj in scene["objects"]:
        #     obj_id = obj['idx']
            # if obj_id not in objlist:
            #     continue
            
            obj = None
            for tobj in scene["objects"]:
                if tobj['idx'] == obj_id:
                    obj = tobj.copy()
                    break
            
            assert obj is not None

            record["obj_mask"][str(obj_id)] = all_masks[str(obj_id)]
            record["obj_bbox"][str(obj_id)] = all_boxs[str(obj_id)]

            # category
            color = obj["color"]
            shape = obj['shape']
            size = obj['size']
            record['category_id'][int(obj['idx'])] = category_ids[obj['shape']][obj['size']][obj['color']] - 1
            
        dataset_dicts.append(record.copy())

    print("训练集中可能包括测试数据量：", imgval)
    print("数据量为：", len(dataset_dicts))
    return dataset_dicts

def register_iep_ref_all():
    print("注册iep-ref数据集")
    # for d in ['val', "train"]:
    #     print("注册iep数据集" + str(d))
    simple = False

    split = 'train'
    # root="/nfs/iep-ref-master/data/"
    root = "/home/lingpeng/project/iep-ref-master/data3"   # 防止IO争用
    # scene_file = osp.join(root, 'clevr_ref+_1.0/scenes/clevr_ref+_' + split + '_scenes.json')
    # ref_root = '/nfs/iep-ref-master'
    ref_root = "/home/lingpeng/project/iep-ref-master"

    # refexp_file = osp.join(root, 'referring_simple_difficult/refexps/ref_picked_' + split + '.json')

    scene_file = osp.join(root, 'clevr_ref+_1.0/scenes/clevr_ref+_' + split + '_scenes.json')
    refexp_file = osp.join(root, 'referring_rubber/refexps/ref_picked_' + split + '.json')

    image_root = osp.join(root, 'clevr_ref+_1.0/images/' + (split if split == 'train' else 'val'))
        
    # DatasetCatalog.register("iep_ref_" + "train", lambda: register_iep_refs(root=root, split="train"))
    # MetadataCatalog.get('iep_ref_' + "train").set(json_file=refexp_file, image_root=image_root, evaluator_type="refcoco", thing_classes=["ball"])

    split = 'val'
    # root="/nfs/iep-ref-master/data/"
    root = "/home/lingpeng/project/iep-ref-master/data"

    # scene_file = osp.join(root, 'clevr_ref+_1.0/scenes/clevr_ref+_' + split + '_scenes.json')
    # ref_root = '/nfs/iep-ref-master/'
    ref_root = "/home/lingpeng/project/iep-ref-master"

    # refexp_file = osp.join(root, 'referring_simple_difficult/refexps/ref_picked_' + split + '.json')

    scene_file = osp.join(root, 'clevr_ref+_1.0/scenes/clevr_ref+_' + split + '_scenes.json')
    refexp_file = osp.join(root, 'referring_rubber/refexps/ref_picked_' + split + '.json')


    image_root = osp.join(root, 'clevr_ref+_1.0/images/' + (split if split == 'train' else 'val'))
     
    DatasetCatalog.register("iep_ref_" + 'val', lambda: register_iep_refs(root=root, split='val'))
    MetadataCatalog.get('iep_ref_' + 'val').set(json_file=refexp_file, image_root=image_root, evaluator_type="refcoco", thing_classes=["ball"])
        
register_iep_ref_all()





    
    


    



