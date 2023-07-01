import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from numpy.core.fromnumeric import nonzero
import torch
from torch import nn
import os 
import torch.nn.functional as F
import time
import cv2

from detectron2.structures import BitMasks
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.layers import cat
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.proposal_generator import build_proposal_generator

from .langencoder.rnn import CustomerBert
from .langencoder.build_lang import init_ref
from .resnet_sketch import build_resnet_sketch_fpn_backbone
from .rpn_ref import build_proposal_generator_ref
from .roi_heads_ref import GATNet, select_foreground_proposals
from .utils.sketch_visualizer import visualize_sem_inst_mask_withGT
from .utils.fuse_helper import _make_mlp, _make_coord, _make_conv

from .data_mapper_sketch import ICATE_SKETCH, SKETCH_CLASS_NAME



@META_ARCH_REGISTRY.register()
class MGCN(nn.Module):
    
    @configurable
    def __init__(
            self,
            *,
            cfg,
            pixel_mean, 
            pixel_std
        ):
        super().__init__()
        
        self.rpn_hidden_dim = 256
        self.visual_dim = 256
        self.roi_dim = 1024
        
        #############
        # text encoder
        #############
        self.use_bert = True
        if self.use_bert:
            self.bert = CustomerBert()
            
        else:
            self.textencoder = init_ref(cfg, is_training=self.training)
        
        # self.textDim = 512 if not self.use_bert else 768
        self.textDim = 512
        
        
        ############
        # backbone
        ############
        assert self.textDim == 512
        self.backbone = build_resnet_sketch_fpn_backbone(cfg)
        
        
        ############ 
        # RPN
        ############
        self.visualencoder = 'RPN'
        
        if self.visualencoder == 'RPN':
            # self.proposal_generator = build_proposal_generator_ref(cfg, self.backbone.output_shape())
            self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        else:
            raise
        self.rpn_hn_mapping = nn.Sequential(
            nn.Linear(self.textDim, self.rpn_hidden_dim),
            nn.ReLU(),
        )
        
        self.use_film = False
        # 后面再补充
        if self.use_film:
            self.joint_embedding_size = self.visual_dim + 8
            self.joint_embedding_dropout = 0.1
            self.joint_inp_dim = self.joint_embedding_size
            self.joint_out_dim = self.visual_dim
            
            self.mapping_lang = _make_mlp(self.textDim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            # self.gamma = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(len(self.in_features)))
            # self.beta = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(len(self.in_features)))
            # self.joint_fusion = nn.ModuleList([_make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
            #                                    for _ in range(len(self.in_features))])
            self.gamma = nn.ModuleDict()
            self.beta = nn.ModuleDict()
            self.joint_fusion = nn.ModuleDict()
            for feat_name in cfg.MODEL.RPN.IN_FEATURES:
                self.gamma[feat_name] = nn.Linear(self.joint_embedding_size, self.joint_inp_dim)
                self.beta[feat_name] = nn.Linear(self.joint_embedding_size, self.joint_inp_dim)
                self.joint_fusion[feat_name] = _make_conv(self.joint_inp_dim, self.joint_out_dim, 1)
        
        self.test2 = True
        # assert (self.test2 and not self.use_film) or (self.use_film and not self.test2)
        if self.test2:
            self.joint_embedding_size = self.visual_dim
            self.joint_inp_dim = self.visual_dim + 8 + self.joint_embedding_size
            self.joint_embedding_dropout = 0.1
            self.joint_out_dim = self.visual_dim
            
            self.mapping_lang = _make_mlp(self.textDim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.joint_fusion = nn.ModuleDict()
            for feat_name in cfg.MODEL.RPN.IN_FEATURES:
                self.joint_fusion[feat_name] = _make_conv(self.joint_inp_dim, self.joint_out_dim, 1)
        
        ############
        # ROI
        ############
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        input_shape = self.backbone.output_shape()
        in_channels = [input_shape[f].channels for f in self.in_features]
        
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        
        self.roi_hidden_dim = 512
        assert self.roi_hidden_dim == self.textDim
        self.globalVisual = nn.Sequential(
            nn.Conv2d(self.visual_dim, self.roi_hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.roi_hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # average pool
        )
        self.roi_hn_mapping = nn.Sequential(
            nn.Linear(self.textDim, self.roi_hidden_dim),
            nn.ReLU(),
        )
        
        self.roi_mapping = nn.Sequential(
            nn.Linear(self.roi_dim, self.roi_hidden_dim),
            nn.ReLU(),
        )
        self.combined_node_dim = self.textDim + 5 + self.roi_hidden_dim + self.roi_hidden_dim
        
        self.node_mapping = nn.Sequential(
            nn.Linear(self.combined_node_dim, self.roi_hidden_dim),
            nn.ReLU()
        )
        
        
        ############
        # GAT
        ############
        self.use_gcn = False
        self.use_one_gcn = True
        if not self.use_gcn:
            self.use_one_gcn = False
            
        if self.use_gcn:
            self.Rstep = 17 # 与max_len一致
            self.graph_head = 8
            self.graph_in_channels = self.roi_hidden_dim
            self.graph_out_channels = 256
            self.graphNet = GATNet(self.graph_in_channels, self.graph_out_channels, nheads=self.graph_head)
        
            # self.gcn_init_bn_relu = nn.Sequential(
            #     nn.BatchNorm1d(self.graph_in_channels),
            #     nn.ReLU()
            # )
            if not self.use_one_gcn:
                self.gcn_bn_relu = nn.ModuleList(nn.Sequential(nn.BatchNorm1d(self.graph_in_channels), nn.ReLU()) for _ in range(self.Rstep))
        
        self.node_mapping_back = nn.Sequential(
            nn.Linear(self.roi_hidden_dim, self.roi_dim),
            nn.ReLU()
        )
        ############
        # others
        ############
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        
        self.train_on_pred_boxes = False
        self.mask_on = True
        
        self.sketch = True
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.vis_period = 1000
        self.time = 0
        self.save_path = "/home/lingpeng/project/Adet_2022/effe_RPN_pred_in_train/" + \
                                cfg.RPN_SAVE + "_gcn" + ("2" if self.test2 else "") + \
                                ("_use_film" if self.use_film else "") + \
                                ("_wogcn" if not self.use_gcn else "") + \
                                ("onenewgraph" if self.use_one_gcn else "") + \
                                ("_woaligmentloss")
                                
        os.makedirs(self.save_path, exist_ok=True)
        
        self.save_path_test = "/home/lingpeng/project/Adet_2022/effe_RPN_pred_in_test/" + \
                                cfg.RPN_SAVE + "_gcn" + ("2" if self.test2 else "") + \
                                ("_use_film" if self.use_film else "") + \
                                ("_wogcn" if not self.use_gcn else "") + \
                                ("onenewgraph" if self.use_one_gcn else "")
                                
        os.makedirs(self.save_path_test, exist_ok=True)
        self.input_format = 'RGB'
             
    @classmethod
    def from_config(cls, cfg):
        
        return {
            "cfg": cfg,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    @torch.no_grad()
    def save_train_pred(self, batched_inputs, proposals, stage='RPN'):
        # storage = get_event_storage()
        max_vis_prop = 20

        visual_num_batch = 6
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            image_id = input['image_id']
            cref_id = input['cref_id']


            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            gt_boxes = input["instances"].gt_boxes.tensor.cpu().detach().numpy()
            gt_labels = input['instances'].gt_classes.cpu().detach().numpy()

            box_size = min(len(prop.proposal_boxes), max_vis_prop)

            pred_boxes = prop.proposal_boxes[0:box_size].tensor.cpu().detach().numpy()
            pred_labels = prop.objectness_logits[0:box_size].cpu().detach().sigmoid().numpy()

            fileter = pred_labels >= 0.9
            pred_labels = pred_labels[fileter]
            pred_boxes = pred_boxes[fileter]

            # assert self.sketch
            save_file = os.path.join(self.save_path, "RPN_PRED_"+str(self.time)+"iter_"+str(image_id)+"_"+str(cref_id)+'.png')
            visualize_sem_inst_mask_withGT(img, gt_boxes, pred_boxes, gt_labels, scores=pred_labels, save_path=save_file)
            # self.time += 1
            if visual_num_batch <= 0:
                break
            
            visual_num_batch -= 1

    def save_test_pred(self, batched_inputs, proposals, stage='RPN'):
        # storage = get_event_storage()
        print("save RPN results to " + self.save_path)
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]

            image_id = input['image_id']
            cref_id = input['cref_id']

            file_root_per_image = os.path.join(self.save_path_test, str(image_id))
            os.makedirs(file_root_per_image, exist_ok=True)

            rpn_pred_inst_file = os.path.join(file_root_per_image, str(image_id) + '_' + str(cref_id) + "_rpn_pred_image_inst.png")
            # rpn_pred_inst_list_file = os.path.join(file_root_per_image, str(image_id) + '_' + str(cref_id) + "_rpn_pred_image_inst.txt")

            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            out_height, out_width = img.shape[:2]
            img = cv2.resize(img.astype(np.float32), (768, 768)).astype(img.dtype)

            gt_boxes = input["instances"].gt_boxes.tensor.cpu().numpy()
            gt_labels = input['instances'].gt_classes.cpu().numpy()

            box_size = min(len(prop.proposal_boxes), max_vis_prop)

            pred_boxes = prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            pred_labels = prop.objectness_logits[0:box_size].cpu().sigmoid().numpy()
            fileter = pred_labels >= 0.5
            pred_labels = pred_labels[fileter]
            pred_boxes = pred_boxes[fileter]
            
            scale_factor_x, scale_factor_y = 768*1.0 / out_width, 768*1.0 / out_height

            tpred_boxes = []

            for i in range(pred_boxes.shape[0]):
                x0, y0, x1, y1 = pred_boxes[i]
                x0 *= scale_factor_x
                x1 *= scale_factor_x
                y0 *= scale_factor_y
                y1 *= scale_factor_y

                tpred_boxes.append([x0, y0, x1, y1])

            pred_boxes = np.array(tpred_boxes)
            # 类id映射回初始
            if self.num_classes == 46:
                gt_labels = np.array([ICATE_SKETCH[x] for x in gt_labels])
            else:
                raise


            # fileter = pred_labels >= 0.001
            # pred_labels = pred_labels[fileter]
            # pred_boxes = pred_boxes[fileter]

            # assert self.sketch
            # save_file = os.path.join(self.save_path, "RPN_PRED_"+str(self.time)+"iter_"+str(image_id)+"_"+str(cref_id)+'.png')
            # visualize_sem_inst_mask_withGT(img, gt_boxes, pred_boxes, gt_labels, class_names=list(SKETCH_CLASS_NAME.values()), scores=pred_labels, save_path=rpn_pred_inst_file, list_file_name=rpn_pred_inst_list_file)
            visualize_sem_inst_mask_withGT(img, gt_boxes, pred_boxes, gt_labels, class_names=list(SKETCH_CLASS_NAME.values()), scores=pred_labels, save_path=rpn_pred_inst_file)
            # visualize_sem_inst_mask_withGT(img, gt_boxes, pred_boxes, gt_labels, scores=None, save_path=rpn_pred_inst_file)
            # self.time += 1
    
    
    @torch.no_grad() 
    def inference(self, batched_inputs, do_postprocess: bool = True):
        sent_encodes = cat([x['sent_encode'].unsqueeze(0).to(self.device) for x in batched_inputs])
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.use_bert:
            mask_attention = cat([x["mask_attention"].unsqueeze(0).to(self.device) for x in batched_inputs])
            word_embeddings, hn = self.bert(sent_encodes, attention_mask=mask_attention)

            sent_dict = {
                "hs": None,
                "hn": hn,
                "embedding": word_embeddings,
                'words': sent_encodes
            }

        else:
            seq_len = (sent_encodes != 0).sum(1).max().item()
            sent_encodes = sent_encodes[:, :seq_len] 
            # hs, hn, embedding = self.textencoder(sent_encodes)
            lang_dict = self.textencoder(sent_encodes)
        
            sent_dict = {
                "hs": lang_dict["output"],
                "hn": lang_dict['final_output'],
                "embedding": lang_dict['embedded'],
                'words': sent_encodes
            }
        
        # handle image
        images = self.preprocess_image(batched_inputs)
        if self.sketch:
            imgs = [x["image"].to(self.device).unsqueeze(0) for x in batched_inputs]
            imgs = cat(imgs, dim=0)
            imgs = imgs[:, 0, :, :]
            # print(imgs.shape) # [N, H, W]
            # print(imgs)
            # assert 1 == 0
            image_bin = torch.ones_like(imgs)
            image_bin[imgs == 255] = 0
        else:
            image_bin = None
            
        ###############
        #  backbone
        ###############
        features = self.backbone(images.tensor)
        assert 'p5' in features.keys()
        # print("backbone' time: " + str(time.time()-t1))
        # get global feature
        feature5 = features['p5']
        # encoder globa feature information
        globalFeature = self.globalVisual(feature5).squeeze(-1).squeeze(-1) # [N, roi_hidden_dim]
        
        ###############
        #  RPN
        ###############
        if not self.use_film:
            if self.test2:
                language_feature = sent_dict['hn']
                language_feature = self.mapping_lang(language_feature)
                visu_feat = {}
                for feat_name, feat in features.items():
                    coord_feat = _make_coord(feat.shape[0], feat.shape[-2], feat.shape[-1]).to(feat.device)
                    lang_ = language_feature.view(feat.shape[0], -1, 1, 1).expand_as(feat)
                    feat = torch.cat([feat, coord_feat, lang_], dim=1)
                    visu_feat[feat_name] = self.joint_fusion[feat_name](feat)
                    
                proposals, _ = self.proposal_generator(images, visu_feat) # [N, B, D]
        
            else:
                proposals, _ = self.proposal_generator(images, features) # [N, B, D]
        else:
            # pass
            language_feature = sent_dict['hn']
            language_feature = self.mapping_lang(language_feature)
            # gamma = [F.tanh(gamma(language_feature)) for gamma in self.gamma]
            # beta = [F.tanh(beta(language_feature)) for beta in self.beta]
            visu_feat = {}
            for feat_name, feat in features.items():
                
                coord_feat = _make_coord(feat.shape[0], feat.shape[-2], feat.shape[-1]).to(feat.device)
                feat = torch.cat([feat, coord_feat], dim=1)
                
                gamma = F.tanh(self.gamma[feat_name](language_feature))
                beta = F.tanh(self.beta[feat_name](language_feature))
                
                b = beta.view(feat.shape[0], -1, 1, 1).expand_as(feat)
                g = gamma.view(feat.shape[0], -1, 1, 1).expand_as(feat)
                
                feat = F.relu(g*feat + b)
                visu_feat[feat_name] = self.joint_fusion[feat_name](feat)
            # features = visu_feat # 保持RPN与ROI在这个角度上看是独立的
            proposals, _ = self.proposal_generator(images, visu_feat) # [N, B, D]
        
        
        ###############
        #  ROI stage1
        ###############
        # just sample for the roi heads, e.g., box and mask
        # we custome the operation of box head and mask head
        proposals = self.roi_heads(images, features, proposals)
        
        # save the RPN results
        # if self.vis_period > 0 and self.time > 0 and self.time % self.vis_period == 0:
        self.save_test_pred(batched_inputs, proposals, stage='RPN')
            
        assert len(proposals) == feature5.shape[0]
        N = len(proposals)
        
        # use roi_head.box_pooler and roi_head.box_head to extract the reoi features (N*B, 1024)
        roi_feature = self.roi_box_feature_extractor(features, proposals)  # [N*B, D]
        roi_feature = self.roi_mapping(roi_feature) # [N*B, roi_hidden_dim]
        assert roi_feature.shape[0] % N == 0
        roi_num = roi_feature.shape[0] // N
        
        # extract the absolute position and aligment info
        abs_pos_per_img, _ = self.compute_abs_pos_concurrent(proposals)
        abs_pos_per_img = abs_pos_per_img.to(feature5.device)
        # contrastive_gt = contrastive_gt.to(feature5.device)
        abs_pos_per_img = abs_pos_per_img.reshape(-1, 5) # [N*B, 5]
        
        # combine each visual feature, position information, global feature, and final state
        final_state = self.roi_hn_mapping(sent_dict['hn']) # [N, roi_hidden_dim]
        final_state = final_state.repeat_interleave(repeats=roi_num, dim=0) # [N*B, roi_hidden_dim]
        globalFeature = globalFeature.repeat_interleave(repeats=roi_num, dim=0)  # [N*B, roi_hidden_dim]
        print(roi_feature.shape, abs_pos_per_img.shape, globalFeature.shape, final_state.shape)
        # [N*B, roi_hidden_dim], [N*B, 5], [N*B, roi_hidden_dim], [N*B, roi_hidden_dim]
        
        # mapping to the space same as the text embedding
        node_features = torch.cat((roi_feature, abs_pos_per_img, globalFeature, final_state), dim=1)  # [N*B, 3*roi_hidden_dim+5]
        node_features = self.node_mapping(node_features) # [N*B, roi_hidden_dim]
        node_features = F.normalize(node_features, p=2, dim=1)
        
        if not self.use_gcn:
            node_features = node_features.reshape(-1, self.graph_in_channels if self.use_gcn else self.roi_hidden_dim)
            node_features = self.node_mapping_back(node_features)
            node_features = F.normalize(node_features, p=2, dim=1)
            results, _ = self.roi_excute(node_features, features, proposals)
        
            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                return MGCN._postprocess(results, batched_inputs, images.image_sizes)
            else:
                return results
        
        assert self.use_gcn
        ###############
        #  GAT
        ###############
        # construct the graph
        
        # compute the grpah weight
        graph_weights_logits, graph_weights = self.compute_graph_edge_weight_batch(node_features.reshape(N, -1, self.roi_hidden_dim), final_state.reshape(N, -1, self.roi_hidden_dim), roi_num)
        graph_weights_logits = graph_weights_logits.squeeze(1)
        graph_weights = graph_weights.squeeze(1)
        
        t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
        t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
        # use my own principle
        adj = torch.min(t1, t2) / torch.max(t1, t2)
        
        node_features = node_features.reshape(N, -1, self.graph_in_channels) # [N, roi_num, roi_hidden_dim]
        node_features = node_features + self.graphNet(node_features, adj=adj)
        # node_features = self.gcn_init_bn_relu(node_features.reshape(-1, self.graph_in_channels)).reshape(N, roi_num, self.graph_in_channels)
        
        # reasoning step by step
        steps = sent_dict['embedding'].shape[1]
        for step in range(steps):
            if self.use_one_gcn:
                break
            
            # reconstruct the graph weight and adj matrix by using word embedding
            word_embedding = sent_dict['embedding'][:, step, :].repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
            _, graph_weights = self.compute_graph_edge_weight_batch(node_features.reshape(N, -1, self.roi_hidden_dim), word_embedding.reshape(N, -1, self.roi_hidden_dim), roi_num)
            graph_weights = graph_weights.squeeze(1)
            
            t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
            t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
            
            adj = torch.min(t1, t2) / torch.max(t1, t2)
            
            node_features = node_features + self.graphNet(node_features, adj=adj)
            node_features = self.gcn_bn_relu[step](node_features.reshape(-1, self.graph_in_channels)).reshape(N, roi_num, self.graph_in_channels)
        
        node_features = node_features.reshape(-1, self.graph_in_channels)
        node_features = self.node_mapping_back(node_features)
        node_features = F.normalize(node_features, p=2, dim=1)
        
        ###############
        #  ROI stage2
        ###############
        results, _ = self.roi_excute(node_features, features, proposals)
        
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return MGCN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        
        self.time = self.time + 1
        # handle text
        sent_encodes = cat([x['sent_encode'].unsqueeze(0).to(self.device) for x in batched_inputs])
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.use_bert:
            mask_attention = cat([x["mask_attention"].unsqueeze(0).to(self.device) for x in batched_inputs])
            word_embeddings, hn = self.bert(sent_encodes, attention_mask=mask_attention)

            sent_dict = {
                "hs": None,
                "hn": hn,
                "embedding": word_embeddings,
                'words': sent_encodes
            }

        else:
            seq_len = (sent_encodes != 0).sum(1).max().item()
            sent_encodes = sent_encodes[:, :seq_len] 
            # hs, hn, embedding = self.textencoder(sent_encodes)
            lang_dict = self.textencoder(sent_encodes)
        
            sent_dict = {
                "hs": lang_dict["output"],
                "hn": lang_dict['final_output'],
                "embedding": lang_dict['embedded'],
                'words': sent_encodes
            }
        
        # handle image
        images = self.preprocess_image(batched_inputs)
        if self.sketch:
            imgs = [x["image"].to(self.device).unsqueeze(0) for x in batched_inputs]
            imgs = cat(imgs, dim=0)
            imgs = imgs[:, 0, :, :]
            # print(imgs.shape) # [N, H, W]
            # print(imgs)
            # assert 1 == 0
            image_bin = torch.ones_like(imgs)
            image_bin[imgs == 255] = 0
        else:
            image_bin = None
            
        # t1 = time.time()
        ###############
        #  backbone
        ###############
        features = self.backbone(images.tensor)
        assert 'p5' in features.keys()
        # print("backbone' time: " + str(time.time()-t1))
        # get global feature
        feature5 = features['p5']
        # encoder globa feature information
        globalFeature = self.globalVisual(feature5).squeeze(-1).squeeze(-1) # [N, roi_hidden_dim]
        
        ###############
        #  RPN
        ###############
        if not self.use_film:
            if self.test2:
                language_feature = sent_dict['hn']
                language_feature = self.mapping_lang(language_feature)
                visu_feat = {}
                for feat_name, feat in features.items():
                    coord_feat = _make_coord(feat.shape[0], feat.shape[-2], feat.shape[-1]).to(feat.device)
                    lang_ = language_feature.view(feat.shape[0], -1, 1, 1).expand_as(feat)
                    feat = torch.cat([feat, coord_feat, lang_], dim=1)
                    visu_feat[feat_name] = self.joint_fusion[feat_name](feat)
                    
                proposals, vlosses = self.proposal_generator(images, visu_feat, gt_instances) # [N, B, D]
        
            else:
                proposals, vlosses = self.proposal_generator(images, features, gt_instances) # [N, B, D]
        else:
            # pass
            language_feature = sent_dict['hn']
            language_feature = self.mapping_lang(language_feature)
            # gamma = [F.tanh(gamma(language_feature)) for gamma in self.gamma]
            # beta = [F.tanh(beta(language_feature)) for beta in self.beta]
            visu_feat = {}
            for feat_name, feat in features.items():
                
                coord_feat = _make_coord(feat.shape[0], feat.shape[-2], feat.shape[-1]).to(feat.device)
                feat = torch.cat([feat, coord_feat], dim=1)
                
                gamma = F.tanh(self.gamma[feat_name](language_feature))
                beta = F.tanh(self.beta[feat_name](language_feature))
                
                b = beta.view(feat.shape[0], -1, 1, 1).expand_as(feat)
                g = gamma.view(feat.shape[0], -1, 1, 1).expand_as(feat)
                
                feat = F.relu(g*feat + b)
                visu_feat[feat_name] = self.joint_fusion[feat_name](feat)
            # features = visu_feat # 保持RPN与ROI在这个角度上看是独立的
            proposals, vlosses = self.proposal_generator(images, visu_feat, gt_instances) # [N, B, D]
        
        
        ###############
        #  ROI stage1
        ###############
        # just sample for the roi heads, e.g., box and mask
        # we custome the operation of box head and mask head
        proposals = self.roi_heads(images, features, proposals, gt_instances)
        
        # save the RPN results
        if self.vis_period > 0 and self.time > 0 and self.time % self.vis_period == 0:
            self.save_train_pred(batched_inputs, proposals, stage='RPN')
            
        assert len(proposals) == feature5.shape[0]
        N = len(proposals)
        
        # use roi_head.box_pooler and roi_head.box_head to extract the reoi features (N*B, 1024)
        roi_feature = self.roi_box_feature_extractor(features, proposals)  # [N*B, D]
        
        # 改一下这个地方：此时的roi feature已经是vector，空间信息完全丢失
        # 改造成：[N*B, D] -> [N*B, 256, vh, vw]
        
         
        
        
        roi_feature = self.roi_mapping(roi_feature) # [N*B, roi_hidden_dim]
        assert roi_feature.shape[0] % N == 0
        roi_num = roi_feature.shape[0] // N
        
        # extract the absolute position and aligment info
        abs_pos_per_img, contrastive_gt = self.compute_abs_pos_concurrent(proposals)
        abs_pos_per_img = abs_pos_per_img.to(feature5.device)
        contrastive_gt = contrastive_gt.to(feature5.device)
        abs_pos_per_img = abs_pos_per_img.reshape(-1, 5) # [N*B, 5]
        
        # combine each visual feature, position information, global feature, and final state
        final_state = self.roi_hn_mapping(sent_dict['hn']) # [N, roi_hidden_dim]
        final_state = final_state.repeat_interleave(repeats=roi_num, dim=0) # [N*B, roi_hidden_dim]
        globalFeature = globalFeature.repeat_interleave(repeats=roi_num, dim=0)  # [N*B, roi_hidden_dim]
        print(roi_feature.shape, abs_pos_per_img.shape, globalFeature.shape, final_state.shape)
        # [N*B, roi_hidden_dim], [N*B, 5], [N*B, roi_hidden_dim], [N*B, roi_hidden_dim]
        
        # mapping to the space same as the text embedding
        node_features = torch.cat((roi_feature, abs_pos_per_img, globalFeature, final_state), dim=1)  # [N*B, 3*roi_hidden_dim+5]
        node_features = self.node_mapping(node_features) # [N*B, roi_hidden_dim]
        node_features = F.normalize(node_features, p=2, dim=1)
        
        if not self.use_gcn:
            node_features = node_features.reshape(-1, self.graph_in_channels if self.use_gcn else self.roi_hidden_dim)
            node_features = self.node_mapping_back(node_features)
            node_features = F.normalize(node_features, p=2, dim=1)
            _, roi_loss = self.roi_excute(node_features, features, proposals, image_bin=image_bin)
            # print("roi's time: " + str(time.time()-t1))
        
            losses = {}
            losses.update(vlosses)
            losses.update(roi_loss)
        
            # assert 1 == 0
            # print(losses)
            return losses

        assert self.use_gcn
        ###############
        #  GAT
        ###############
        # construct the graph
        
        # compute the grpah weight
        graph_weights_logits, graph_weights = self.compute_graph_edge_weight_batch(node_features.reshape(N, -1, self.roi_hidden_dim), final_state.reshape(N, -1, self.roi_hidden_dim), roi_num)
        graph_weights_logits = graph_weights_logits.squeeze(1)
        graph_weights = graph_weights.squeeze(1)

        # aligment loss
        # contrastive_loss = F.binary_cross_entropy_with_logits(graph_weights_logits, contrastive_gt, reduction="mean")    
        
        # 构造邻接矩阵：[N, roi_num, roi_num]
        # to do
        # need weight: [N, roi_num, roi_num]
        
        t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
        t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
        # use my own principle
        adj = torch.min(t1, t2) / torch.max(t1, t2)
        
        node_features = node_features.reshape(N, -1, self.graph_in_channels) # [N, roi_num, roi_hidden_dim]
        node_features = node_features + self.graphNet(node_features, adj=adj)
        # node_features = self.gcn_init_bn_relu(node_features.reshape(-1, self.graph_in_channels)).reshape(N, roi_num, self.graph_in_channels)
        
        
        # reasoning step by step
        steps = sent_dict['embedding'].shape[1]
        for step in range(steps):
            if self.use_one_gcn:
                break
            # reconstruct the graph weight and adj matrix by using word embedding
            word_embedding = sent_dict['embedding'][:, step, :].repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
            _, graph_weights = self.compute_graph_edge_weight_batch(node_features.reshape(N, -1, self.roi_hidden_dim), word_embedding.reshape(N, -1, self.roi_hidden_dim), roi_num)
            graph_weights = graph_weights.squeeze(1)
            
            t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
            t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
            
            adj = torch.min(t1, t2) / torch.max(t1, t2)
            
            node_features = node_features + self.graphNet(node_features, adj=adj)
            node_features = self.gcn_bn_relu[step](node_features.reshape(-1, self.graph_in_channels)).reshape(N, roi_num, self.graph_in_channels)
        
        node_features = node_features.reshape(-1, self.graph_in_channels)
        node_features = self.node_mapping_back(node_features)
        node_features = F.normalize(node_features, p=2, dim=1)
        ###############
        #  ROI stage2
        ###############
        
        # use roi_heads.box_predictorm and mask_head
        _, roi_loss = self.roi_excute(node_features, features, proposals, image_bin=image_bin)
        losses = {}
        losses.update(vlosses)
        losses.update(roi_loss)
        # losses.update({
        #     "contrastive_alignment_loss": contrastive_loss
        # })
        return losses
    
    # box_pooler and box_head
    def roi_box_feature_extractor(self, features, proposals):
        # pass
        features = [features[f] for f in self.in_features]
        box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        # print(box_features.shape)
        box_features = self.roi_heads.box_head(box_features)
        
        return box_features

    # box_predictor and mask head
    def roi_excute(self, box_features, features, proposals, image_bin=None):
        
        # predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        if self.training:
            assert not torch.jit.is_scripting()
            losses = self._forward_box(box_features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if image_bin is not None:
                nproposals = []
                for index, p in enumerate(proposals):
                    np = p
                    
                    image_bin_i = image_bin[index, :, :].unsqueeze(0).repeat(len(np), 1, 1)
                    # print(image_bin_i.shape)
                    np.image_bins = BitMasks(image_bin_i)  # 会导致值为bool

                    nproposals.append(np)
            # proposals.image_bins = image_bin
                proposals = nproposals
                
            losses.update(self._forward_mask(features, proposals))
            # losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        
        else:
            pred_instances = self._forward_box(box_features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, box_features, proposals):
        #
        # box_features = self.box_head(box_features)
        # print(self.box_predictor)
        predictions = self.roi_heads.box_predictor(box_features)
        del box_features

        if self.training:
            assert not torch.jit.is_scripting()
            losses = self.roi_heads.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.roi_heads.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.roi_heads.box_predictor.inference(predictions, proposals)
            return pred_instances
    
    # mask_pooler and mask_head
    def _forward_mask(self, features, instances):
        
        if not self.mask_on:
            # https://github.com/pytorch/pytorch/issues/43942
            if self.training:
                assert not torch.jit.is_scripting()
                return {}
            else:
                return instances

        # https://github.com/pytorch/pytorch/issues/46703
        assert hasattr(self.roi_heads, "mask_head")

        if self.training:
            assert not torch.jit.is_scripting()
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.roi_heads.mask_pooler is not None:
            features = [features[f] for f in self.in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.roi_heads.mask_pooler(features, boxes)
        else:
            # https://github.com/pytorch/pytorch/issues/41448
            features = dict([(f, features[f]) for f in self.in_features])
            
        return self.roi_heads.mask_head(features, instances)
    
    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        # instances = self._forward_keypoint(features, instances)
        return instances

    def compute_graph_edge_weight_batch(self, roi_features, final_state, roi_num):
        assert roi_features.shape[0] == final_state.shape[0] and roi_features.shape[1] == final_state.shape[1]
        # print(roi_features.shape, final_state.shape)
        # [N, roi_num, D] * 2
        # assert 1 == 0
        similarity_per_roi = torch.bmm(final_state, roi_features.permute(0, 2, 1))  # [N, roi_num, roi_num]
        
        return similarity_per_roi[:, ::roi_num, :], similarity_per_roi.softmax(-1)[:, ::roi_num, :]
           

    def compute_abs_pos_concurrent(self, proposals):
        abs_pos_per_img = []
        contrastive_gt = []
        
        # rel_pos_per_img = []
        # gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        # bg_class_ind = self.num_classes
        
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.cpu().detach()
            if self.training:
              gt_classes = proposals_one_img.gt_classes.data.cpu().detach()
              bg_class_ind = self.num_classes
              # bg_idxs = (gt_classes == bg_class_ind).reshape(-1, 1).repeat(1, 5)
            # print(bg_idxs)
            ih, iw = proposals_one_img.image_size
            # im_array = np.array([iw, ih, iw, ih, iw*ih])
            im_array = torch.tensor([iw, ih, iw, ih, iw*ih], dtype=torch.float32)
            
            # abs_pos_one_img = np.zeros((len(boxes), self.rel_pos), dtype=np.float32)
            abs_pos_one_img = torch.zeros((len(boxes), 5), dtype=torch.float32)
            if self.training:
              contrastive_gt.append(torch.as_tensor((gt_classes != bg_class_ind), dtype=torch.float32))
            
            abs_pos_one_img[:, 0] = boxes[:, 0]
            abs_pos_one_img[:, 1] = boxes[:, 1]
            abs_pos_one_img[:, 2] = boxes[:, 2] - 1
            abs_pos_one_img[:, 3] = boxes[:, 3] - 1
            abs_pos_one_img[:, 4] = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
            
            abs_info = abs_pos_one_img / im_array
            # print(abs_info.shape, bg_idxs.shape)
            # abs_info = abs_info * bg_idxs
            # 移除背景区域的位置
            # abs_info[bg_idxs, :] = torch.zeros((len(boxes), 5), dtype=torch.float32)
            # abs_info = torch.where(bg_idxs, abs_info, torch.zeros((1,5), dtype=torch.float32))
            # print(abs_info)
            abs_pos_per_img.append(abs_info)
            
        abs_pos_per_img = cat([abs_pos_one_img for abs_pos_one_img in abs_pos_per_img])
        if self.training:
          contrastive_gt = cat([e.unsqueeze(0) for e in contrastive_gt])
        # print(abs_pos_per_img.shape, contrastive_gt.shape) # [N*B, 5], [N, B]
        # assert 1 == 0
        return abs_pos_per_img, contrastive_gt
     
    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        # 先输出看看，应该是要添加NMS和matching_Score筛选的
        
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
      
    