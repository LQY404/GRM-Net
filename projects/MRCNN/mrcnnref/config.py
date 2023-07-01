# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_mrcnnref_config(cfg):
    """
    Add config for Mask R-CNN REF.
    """
    # cfg.SEQ_LEN = 24 + 2
    cfg.NUM_HIDEEN_LAYER = 1
    cfg.WORD_VEC_DIM = 512
    cfg.REF_RNN_DIM = 512

    cfg.MODEL.META_ARCHITECTURE = 'MRCNNRef'
    # cfg.MODEL.META_ARCHITECTURE = 'MGCN'
    # cfg.MODEL.META_ARCHITECTURE = "GCNRef"
    # cfg.MODEL.NUM_CLASSES = 77  
    # cfg.MODEL.NUM_CLASSES = 48
    cfg.MODEL.NUM_CLASSES = 46
    # cfg.MODEL.NUM_CLASSES = 90
    # cfg.MODEL.NUM_CLASSES = 1272
    
    cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsRef'
    # cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeads'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.MODEL.NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg.MODEL.NUM_CLASSES
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPNRef"
    # cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'RPN'
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHeadRef"
    # cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14 # 适当增大 for guild


    cfg.MODEL.USE_BERT = False
    cfg.MODEL.USE_ROBERTA = False
    cfg.MODEL.USE_CLIP = False
    
    cfg.bbfusion = False
    cfg.relpos = 5
    cfg.posnorm = False
    cfg.topk = 10
    
    if cfg.MODEL.META_ARCHITECTURE == "GCNRef":
        cfg.topk = 0
        
    cfg.HDC = True
    cfg.ori_abspos = True

    cfg.RPN = False
    cfg.ROI = False
    cfg.RPN_ROI = True
    
    

    if cfg.RPN or cfg.ROI:
        cfg.RPN_ROI = False
    
    if cfg.RPN_ROI:
        cfg.RPN = False
        cfg.ROI = False


    if cfg.relpos == 7:
        cfg.posnorm = True
    
    data_name = ""

    # 如果使用GCN-based模型，那么最好再添加两个卷积
    if cfg.MODEL.META_ARCHITECTURE == 'MGCN':
        cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeads'
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'RPN'
        cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
        
        cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 2
        cfg.MODEL.ROI_BOX_HEAD.NORM = "BN"
    
    if cfg.MODEL.META_ARCHITECTURE == 'GCNRef':
        cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsGCNRef'
        cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'RPNGCNRef'
        cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHeadGCNRef"
        
        # cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 2
        # cfg.MODEL.ROI_BOX_HEAD.NORM = "BN"
    
    if cfg.MODEL.NUM_CLASSES == 46:
        print("sketch模式不需要冻结resnet")
        cfg.MODEL.BACKBONE.NAME = "build_resnet_sketch_fpn_backbone"
        
        cfg.SOLVER.IMS_PER_BATCH = 8
        # cfg.MODEL.WEIGHTS = "/nfs/TEMP/pretrained_mrcnn/model_final.pth"  # dilation_v1, 512
        # cfg.MODEL.WEIGHTS = "/nfs/TEMP/pretrained_mrcnn_dilation_v1_768/model_final.pth"  #dilation_v1 768
        
        # cfg.MODEL.WEIGHTS = "/nfs/TEMP/pretrained_mrcnn_dilation_v2/model_final.pth"  # dilation_v2
        
        
        cfg.MODEL.BACKBONE.FREEZE_AT = 0 
        cfg.DATASETS.TRAIN = ("sketch_train", )
        cfg.DATASETS.TEST = ('sketch_test', )
        # cfg.SOLVER.MAX_ITER = 360000
        # cfg.SOLVER.STEPS = (270000, 330000)
        # cfg.SOLVER.MAX_ITER = 510000 
        # cfg.SOLVER.STEPS = (390000, 480000)
        # cfg.DATA_NUM = 30000
        # cfg.all_iter = 3000000  # 30epoch
        # cfg.all_iter = 3000000  # 50epoch
        # cfg.all_iter = 30000*100  # 100 epoch，没有做重采样
        # cfg.all_iter = 30000*100  # 100 epoch，没有做重采样
        
        cfg.all_iter = 30000 * 70 # 如果使用 oriconfig，根据对loss以及训练输出的分析，认为epoch可以降低到50-80，先降低到70试试
        # 此时时间可以降低到1.5天左右
        
        # cfg.all_iter = 15000*100
        
        # cfg.SOLVER.WARMUP_ITERS = int(cfg.all_iter * 0.2)
        # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        cfg.SOLVER.MAX_ITER = cfg.all_iter // min(8, cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.1)
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        # cfg.SOLVER.GAMMA = cfg.SOLVER.WARMUP_FACTOR
        cfg.SOLVER.STEPS = (int(cfg.all_iter * 0.7) // min(8, cfg.SOLVER.IMS_PER_BATCH), int(cfg.all_iter * 0.9) // min(8, cfg.SOLVER.IMS_PER_BATCH))

        data_name = "sketch"

    elif cfg.MODEL.NUM_CLASSES == 90:
        print("single instance模式")
        # print("sketch模式不需要冻结resnet")
        cfg.MODEL.BACKBONE.FREEZE_AT = -1
        # cfg.MODEL.WEIGHTS = "/home/lingpeng/project/R-101.pkl"
        cfg.MODEL.WEIGHTS = '/home/lingpeng/project/model_final_f96b26.pkl'
        cfg.DATASETS.TRAIN = ('refcocos_train', )
        cfg.DATASETS.TEST = ("refcocos_val", )
        
        
        cfg.all_iter = 120000 * 50 # 如果使用 oriconfig，根据对loss以及训练输出的分析，认为epoch可以降低到50-80，先降低到70试试
        # 此时时间可以降低到1.5天左右
        
        # cfg.all_iter = 15000*100
        
        # cfg.SOLVER.WARMUP_ITERS = int(cfg.all_iter * 0.2)
        # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        cfg.SOLVER.MAX_ITER = cfg.all_iter // min(16, cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.1)
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        # cfg.SOLVER.GAMMA = cfg.SOLVER.WARMUP_FACTOR
        cfg.SOLVER.STEPS = (int(cfg.all_iter * 0.7) // min(16, cfg.SOLVER.IMS_PER_BATCH), int(cfg.all_iter * 0.9) // min(16, cfg.SOLVER.IMS_PER_BATCH))


        data_name = "refcocos"

    elif cfg.MODEL.NUM_CLASSES == 77:
        # print("sketch模式不需要冻结resnet")
        # cfg.MODEL.BACKBONE.FREEZE_AT = 0 
        cfg.MODEL.WEIGHTS = "/home/lingpeng/project/SparseR-CNN-main/SparseR-CNN-main/model_final_a3ec72.pkl"
        cfg.DATASETS.TRAIN = ('refcoco_unc_train',)
        cfg.DATASETS.TEST = ("refcoco_unc_val",)

        data_name = "refcoco"

    elif cfg.MODEL.NUM_CLASSES == 48:
        cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
        cfg.DATASETS.TRAIN = ('iepref_train',)
        cfg.DATASETS.TEST = ("iepref_val",)

        data_name = "iepref"

    elif cfg.MODEL.NUM_CLASSES == 1272:
        # cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
        cfg.DATASETS.TRAIN = ('phrasecut_train',)
        cfg.DATASETS.TEST = ("phrasecut_test",)
        # cfg.INPUT.MASK_FORMAT = "polygon"
        # cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
        cfg.MODEL.BACKBONE.NAME = "build_resnet_sketch_fpn_backbone"
        
        cfg.all_iter = 120000 * 50 # 如果使用 oriconfig，根据对loss以及训练输出的分析，认为epoch可以降低到50-80，先降低到70试试
        # 此时时间可以降低到1.5天左右
        
        # cfg.all_iter = 15000*100
        
        # cfg.SOLVER.WARMUP_ITERS = int(cfg.all_iter * 0.2)
        # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        cfg.SOLVER.MAX_ITER = cfg.all_iter // min(8, cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.1)
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        # cfg.SOLVER.GAMMA = cfg.SOLVER.WARMUP_FACTOR
        cfg.SOLVER.STEPS = (int(cfg.all_iter * 0.7) // min(8, cfg.SOLVER.IMS_PER_BATCH), int(cfg.all_iter * 0.9) // min(8, cfg.SOLVER.IMS_PER_BATCH))

        data_name = "phrasecut"

    cfg.VIS_PERIOD = 1000  # 设置太小会导致模型占据CPU内存过多

    if cfg.RPN_ROI:
        cfg.RPN_SAVE = data_name + "_reltop" + str(cfg.topk) + "_relpos" + str(cfg.relpos) + "_" + \
            ("noposnorm" if not cfg.posnorm else "posnorm") + "_dilation" + ('_BERT' if cfg.MODEL.USE_BERT else "") + ("_RPNoripos" if cfg.ori_abspos else "_RPNnewpos")
    elif cfg.RPN:
        cfg.RPN_SAVE = data_name + "onlyRPN"
    elif cfg.ROI:
        cfg.RPN_SAVE = data_name + "onlyROI"

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    # cfg.SOLVER.BACKBONE_MULTIPLIER = 0.01
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    if cfg.MODEL.USE_BERT or cfg.MODEL.USE_CLIP:
        cfg.SOLVER.TEXTENCODER = 0.0001
    else:
        cfg.SOLVER.TEXTENCODER = 1.0
    # cfg.SOLVER.BASE_LR = 0.0002
    # cfg.SOLVER.BASE_LR = 0.00025
    # cfg.SOLVER.BASE_LR = (0.00025/8) * cfg.SOLVER.IMS_PER_BATCH # 保持bs/lr为定值
    cfg.SOLVER.BASE_LR = 0.00025 * max(1.0, (cfg.SOLVER.IMS_PER_BATCH / 8))

    # matching
    cfg.MATCHER = CN()
    cfg.MATCHER.set_cost_class = 2.0
    cfg.MATCHER.set_cost_bbox = 5.0
    cfg.MATCHER.set_cost_giou = 0.0

    # focal
    cfg.FOCAL = CN()
    cfg.FOCAL.LOSS_ALPHA = 0.25
    cfg.FOCAL.LOSS_GAMMA = 2.0