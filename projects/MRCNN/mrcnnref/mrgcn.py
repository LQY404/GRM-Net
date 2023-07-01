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

from .langencoder.rnn import CustomerBert
from .langencoder.build_lang import init_ref
from .resnet_sketch import build_resnet_sketch_fpn_backbone
from .rpn_ref import build_proposal_generator_ref
from .roi_heads_ref import GATNet, select_foreground_proposals
from .utils.sketch_visualizer import visualize_sem_inst_mask_withGT
from .utils.fuse_helper import _make_mlp, _make_coord, _make_conv

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
        
        self.mdim = 256
        self.visualDim = 256
        self.roiDim = 1024
        
        
        self.use_bert = True
        if self.use_bert:
            self.bert = CustomerBert()
            
        else:
            self.textencoder = init_ref(cfg, is_training=self.training)
        
        # self.textDim = 512 if not self.use_bert else 768
        self.textDim = 256
        
        self.backbone = build_resnet_sketch_fpn_backbone(cfg)
        
        self.visualencoder = 'RPN'
        
        if self.visualencoder == 'RPN':
            self.proposal_generator = build_proposal_generator_ref(cfg, self.backbone.output_shape())
        else:
            raise
        
        #
        # build roi pooler
        self.in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        input_shape = self.backbone.output_shape()
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        
        # flatten roi feature
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)

        # build for mask
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )
        self.mask_head = build_mask_head(cfg, shape)
        
        # self.visualencoder = buildVEncoder(cfg)  # Mask R-CNN or DETR or RPN.
        # at this time, we choose the RPN
        
        self.globalVisual = nn.Sequential(
            nn.Conv2d(self.visualDim, self.mdim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.mdim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # average pool
        )
        
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        
        self.sketch = True
        
        self.combined_node_dim = self.textDim + 5 + self.mdim + self.mdim
        
        self.Rstep = 17 # 与max_len一致
        self.graph_head = 2
        self.graph_in_channels = self.combined_node_dim
        self.graph_out_channels = 256
        self.graphNet = GATNet(self.graph_in_channels, self.graph_out_channels, nheads=self.graph_head)
        
        self.node_mapping = nn.Sequential(
            nn.Linear(self.graph_in_channels, self.roiDim),
            nn.ReLU()
        )
        
        self.hn_mapping = nn.Sequential(
            nn.Linear(self.textDim, self.mdim),
            nn.ReLU(),
        )
        
        self.word_mapping = nn.Sequential(
            nn.Linear(self.textDim, self.mdim),
            nn.ReLU(),
        )
        
        self.roi_mapping = nn.Sequential(
            nn.Linear(self.roiDim, self.textDim),
            nn.ReLU(),
        )
        
        self.proposal_append_gt = True
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=True,
            # allow_low_quality_matches=False
        )
        
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        
        self.train_on_pred_boxes = False
        self.mask_on = True
        
        
        self.use_film = False
        # 后面再补充
        if self.use_film:
            self.joint_embedding_size = self.visualDim + 8
            self.joint_embedding_dropout = 0.1
            self.joint_inp_dim = self.joint_embedding_size
            self.joint_out_dim = self.visualDim
            
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
        
        self.test2 = False
        # assert (self.test2 and not self.use_film) or (self.use_film and not self.test2)
        if self.test2:
            self.joint_embedding_size = self.visualDim
            self.joint_inp_dim = self.visualDim + 8 + self.joint_embedding_size
            self.joint_embedding_dropout = 0.1
            self.joint_out_dim = self.visualDim
            
            self.mapping_lang = _make_mlp(self.textDim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.joint_fusion = nn.ModuleDict()
            for feat_name in cfg.MODEL.RPN.IN_FEATURES:
                self.joint_fusion[feat_name] = _make_conv(self.joint_inp_dim, self.joint_out_dim, 1)
        
        self.vis_period = 1000
        self.time = 0
        self.save_path = "/home/lingpeng/project/Adet_2022/effe_RPN_pred_in_"  + ("test/" if not self.training else "train/") + cfg.RPN_SAVE + "_gcn" + ("2" if self.test2 else "") + ("_use_film" if self.use_film else "")
        os.makedirs(self.save_path, exist_ok=True)
        self.input_format = 'RGB'
        # self.logit_scale = torch.nn.Parameter(torch.ones([])*np.log(1 / 0.07))
        
        '''
        现在跑的三个实验：test, test2, 以及test_film中
        test是RPN部分完全没有文本以及其他影响，只有图像特征自身
        test2与test的差异仅在于test2会在RPN中添加文本以及位置信息，添加信息的方式是简单的concat
        test_film则与test2类似，但是添加信息的方式是film
        
        参数分别对应：
        test——self.film=False, self.test2=False
        test2——self.film=False, self.test2=True
        test_film——self.film=True, self.test2=False
        
        需要注意的是，这三个实验都没有使用重采样的策略，但是用到了类别损失加权
        '''
        
        
        
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
    
    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        # sampled_idxs = sampled_fg_idxs
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals_more2one(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)  # 把GT也当做proposal

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # print(match_quality_matrix.shape)
            # print(match_quality_matrix)
            # assert 1 == 0
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # proposals_per_image.gt_classes[proposals_per_image.gt_classes == self.num_classes] = 0
            

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
    
    def save_train_pred(self, batched_inputs, proposals, stage='RPN'):
        # storage = get_event_storage()
        max_vis_prop = 20

        visual_num_batch = 6
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            image_id = input['image_id']
            cref_id = input['cref_id']


            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)

            gt_boxes = input["instances"].gt_boxes.tensor.cpu().numpy()
            gt_labels = input['instances'].gt_classes.cpu().numpy()

            box_size = min(len(prop.proposal_boxes), max_vis_prop)

            pred_boxes = prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            pred_labels = prop.objectness_logits[0:box_size].cpu().sigmoid().numpy()

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

    
    @torch.no_grad() 
    def inference(self, batched_inputs, do_postprocess: bool = True):
        # pass
        sent_encodes = cat([x['sent_encode'].unsqueeze(0).to(self.device) for x in batched_inputs])
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
        # t1 = time.time()
        features = self.backbone(images.tensor)
        assert 'p5' in features.keys()
        # print("backbone' time: " + str(time.time()-t1))
        # get global feature
        feature5 = features['p5']
        # encoder globa feature information
        globalFeature = self.globalVisual(feature5).squeeze(-1).squeeze(-1) # [N, D]
        
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
        
        # proposals = self.label_and_sample_proposals_more2one(proposals, gt_instances)
        assert len(proposals) == feature5.shape[0]
        
        N = len(proposals)
        # print(self.box_pooler)
        # print(self.box_head)
        # print()
        # print(proposals)
        
        roi_feature = self.roi_box_feature_extractor(features, proposals)  # [N*B, D]
        roi_feature = self.roi_mapping(roi_feature) # [N*B, mdim]
        assert roi_feature.shape[0] % N == 0
        roi_num = roi_feature.shape[0] // N
        
        # get pos info of each proposal
        # t1 = time.time()
        abs_pos_per_img, _ = self.compute_abs_pos_concurrent(proposals)
        abs_pos_per_img = abs_pos_per_img.to(feature5.device)
        # contrastive_gt = contrastive_gt.to(feature5.device)
        abs_pos_per_img = abs_pos_per_img.reshape(-1, 5) # [N*B, 5]
        # print("time of extraction other info: " + str(time.time()-t1))
        # combine each visual feature, position information, global feature, and final state
        # to do
        
        final_state = self.hn_mapping(sent_dict['hn']) # [N, D]
        final_state = final_state.repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
        
        globalFeature = globalFeature.repeat_interleave(repeats=roi_num, dim=0)  # [N*B, D]
        # print(roi_feature.shape, abs_pos_per_img.shape, globalFeature.shape, final_state.shape)
        # [N*B, text_dim], [N*B, 5], [N*B, 256], [N*B, 256]
        
        # print(abs_pos_per_img)
        # assert 1 == 0
        node_features = torch.cat((roi_feature, abs_pos_per_img, globalFeature, final_state), dim=1)  # [N*B, D+5+D+D]
        # print(node_features.shape)
        # constructing the graph edge's weight 
        # method 1
        # to do
        hn = sent_dict['hn'].repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
        # graph_weights = self.compute_graph_edge_weight(roi_feature, hn)[::roi_num, :] # 注意这里用的是直接通过text encoder出来的final state
        # print(graph_weights.shape) # [len(text)(N), N*B]
        # print(graph_weights[::roi_num, :])
        
        
        # t1 = time.time()
        graph_weights_logits, graph_weights = self.compute_graph_edge_weight_batch(roi_feature.reshape(N, -1, self.textDim), hn.reshape(N, -1, self.textDim), roi_num)
        graph_weights_logits = graph_weights_logits.squeeze(1)
        graph_weights = graph_weights.squeeze(1)
        
        # 构造邻接矩阵：[N, roi_num, roi_num]
        # to do
        # need weight: [N, roi_num, roi_num]
        
        t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
        t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
        
        adj = torch.min(t1, t2) / torch.max(t1, t2)
        # print(adj)
        # assert 1 == 0
        # print(adj.shape) # [N, roi_num, roi_num]
                                                                                                                                                                                                                                                                                                                                                                                           
        
        node_features = node_features.reshape(N, -1, self.graph_in_channels) # [N, roi_num, D]
        # t1 = time.time()
        node_features = self.graphNet(node_features, adj=adj)
        # 这个地方后面要改，上下统一（是否使用残差）
        # assert 1 == 0
        
        # reasoning step by step
        # 考虑要不要加残差
        # 这部分需要参考其他工作的实现
        
        steps = sent_dict['embedding'].shape[1]
        # start = time.time()
    
        for step in range(steps):
            # reconstruct the graph weight and adj matrix by using word embedding
            word_embedding = sent_dict['embedding'][:, step, :].repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
            _, graph_weights = self.compute_graph_edge_weight_batch(roi_feature.reshape(N, -1, self.textDim), word_embedding.reshape(N, -1, self.textDim), roi_num)
            graph_weights = graph_weights.squeeze(1)
            
            t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
            t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
            
            adj = torch.min(t1, t2) / torch.max(t1, t2)
            
            node_features = node_features + self.graphNet(node_features, adj=adj)
            
        # print("time of through the graph 17 times: " + str(time.time()-start))
        # print(node_features.shape) # [N, roi_num, 256*3+5 (773)]
        # assert 1 == 0
        node_features = node_features.reshape(-1, self.graph_in_channels)
        node_features = self.node_mapping(node_features)
        node_features = F.normalize(node_features, p=2, dim=1)
        # final, execute classification, segmentation and regression
        # to do
        # t1 = time.time()
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
        features = self.backbone(images.tensor)
        assert 'p5' in features.keys()
        # print("backbone' time: " + str(time.time()-t1))
        # get global feature
        feature5 = features['p5']
        # encoder globa feature information
        globalFeature = self.globalVisual(feature5).squeeze(-1).squeeze(-1) # [N, D]
        
        # print(globalFeature.shape) # [N, D]
        # assert 1 == 0
        # get proposal feature from RPN
        # t1 = time.time()
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
        
        # print("RPN' time: " + str(time.time()-t1))
        
        proposals = self.label_and_sample_proposals_more2one(proposals, gt_instances)
        
        if self.vis_period > 0 and self.time > 0 and self.time % self.vis_period == 0:
        # if self.vis_period > 0:
            # storage = get_event_storage()
            # if storage.iter > 0 and storage.iter % self.vis_period == 0:
            #     self.visualize_training(batched_inputs, proposals, stage='RPN')

            self.save_train_pred(batched_inputs, proposals, stage='RPN')
            
        # print(proposals.shape)
        # print(feature5.shape) 
        # print(type(proposals), len(proposals))
        # print(proposals)
        # assert 1 == 0
        # contrastive_gt = []
        # for i in proposals:
        assert len(proposals) == feature5.shape[0]
        
        N = len(proposals)
        # print(self.box_pooler)
        # print(self.box_head)
        # print()
        # print(proposals)
        
        roi_feature = self.roi_box_feature_extractor(features, proposals)  # [N*B, D]
        roi_feature = self.roi_mapping(roi_feature) # [N*B, mdim]
        # print(N, roi_feature.shape) # batch size, 
        # assert 1 == 0

        # assert 1 == 0
        
        assert roi_feature.shape[0] % N == 0
        roi_num = roi_feature.shape[0] // N
        
        # get pos info of each proposal
        # t1 = time.time()
        abs_pos_per_img, contrastive_gt = self.compute_abs_pos_concurrent(proposals)
        abs_pos_per_img = abs_pos_per_img.to(feature5.device)
        contrastive_gt = contrastive_gt.to(feature5.device)
        abs_pos_per_img = abs_pos_per_img.reshape(-1, 5) # [N*B, 5]
        # print("time of extraction other info: " + str(time.time()-t1))
        # combine each visual feature, position information, global feature, and final state
        # to do
        # ---------------------------------------
        # 明天试试这里用hn之前可以再加一个线性层
        # ---------------------------------------
        final_state = self.hn_mapping(sent_dict['hn']) # [N, D]
        final_state = final_state.repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
        
        globalFeature = globalFeature.repeat_interleave(repeats=roi_num, dim=0)  # [N*B, D]
        print(roi_feature.shape, abs_pos_per_img.shape, globalFeature.shape, final_state.shape)
        # [N*B, text_dim], [N*B, 5], [N*B, 256], [N*B, 256]
        
        # print(abs_pos_per_img)
        # assert 1 == 0
        node_features = torch.cat((roi_feature, abs_pos_per_img, globalFeature, final_state), dim=1)  # [N*B, D+5+D+D]
        # print(node_features.shape)
        # constructing the graph edge's weight 
        # method 1
        # to do
        # ---------------------------------------
        # 明天试试同样这里用hn之前可以再加一个线性层
        # ---------------------------------------
        hn = sent_dict['hn'].repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
        # graph_weights = self.compute_graph_edge_weight(roi_feature, hn)[::roi_num, :] # 注意这里用的是直接通过text encoder出来的final state
        # print(graph_weights.shape) # [len(text)(N), N*B]
        # print(graph_weights[::roi_num, :])
        
        
        # t1 = time.time()
        # ---------------------------------------
        # 计算图边权重时到底是使用原生的roi feature还是添加了位置等信息的node feature？？？
        # 这个问题需要再考虑，先看一下本次测试结果
        # 但是从理论上来说应该是使用添加了位置等信息的node feature效果更好
        # ---------------------------------------
        graph_weights_logits, graph_weights = self.compute_graph_edge_weight_batch(roi_feature.reshape(N, -1, self.textDim), hn.reshape(N, -1, self.textDim), roi_num)
        graph_weights_logits = graph_weights_logits.squeeze(1)
        graph_weights = graph_weights.squeeze(1)
        # print("struct graph weight'time : " + str(time.time()-t1))
        
        # print(graph_weights.shape, graph_weights_logits.shape)
        # [N, roi_num] * 2
        # print(graph_weights.shape) # [N, roi_num]
        # print(graph_weights[0].topk(10))
        # assert 1 == 0
        # 添加对比损失:one-hot形式即可
        
        contrastive_loss = F.binary_cross_entropy_with_logits(graph_weights_logits, contrastive_gt, reduction="mean")    
        
        
        # 构造邻接矩阵：[N, roi_num, roi_num]
        # to do
        # need weight: [N, roi_num, roi_num]
        
        t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
        t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
        
        adj = torch.min(t1, t2) / torch.max(t1, t2)
        # print(adj)
        # assert 1 == 0
        # print(adj.shape) # [N, roi_num, roi_num]
                                                                                                                                                                                                                                                                                                                                                                                           
        
        node_features = node_features.reshape(N, -1, self.graph_in_channels) # [N, roi_num, D]
        # t1 = time.time()
        # ---------------------------------------
        # 明天试试加残差
        # ---------------------------------------
        node_features = self.graphNet(node_features, adj=adj)  # 同样要改
        # print("through graph 1 time: " + str(time.time()-t1))
        # print(node_features.shape)
        
        # everything is ok albow
        # assert 1 == 0
    
        # to do
        # reasoning step by step
        # 考虑要不要加残差
        # 这部分需要参考其他工作的实现
        
        steps = sent_dict['embedding'].shape[1]
        # start = time.time()
    
        for step in range(steps):
            # reconstruct the graph weight and adj matrix by using word embedding
            word_embedding = sent_dict['embedding'][:, step, :].repeat_interleave(repeats=roi_num, dim=0) # [N*B, D]
            _, graph_weights = self.compute_graph_edge_weight_batch(roi_feature.reshape(N, -1, self.textDim), word_embedding.reshape(N, -1, self.textDim), roi_num)
            graph_weights = graph_weights.squeeze(1)
            
            t1 = graph_weights.repeat((1, 1, roi_num)).reshape(N, roi_num, roi_num) # [N, roi_num, roi_num]
            t2 = graph_weights.unsqueeze(-1) # [N, roi_num, 1]
            
            adj = torch.min(t1, t2) / torch.max(t1, t2)
            
            node_features = node_features + self.graphNet(node_features, adj=adj)
            
            # pass
        
        # print("time of through the graph 17 times: " + str(time.time()-start))
        # print(node_features.shape) # [N, roi_num, 256*3+5 (773)]
        # assert 1 == 0
        node_features = node_features.reshape(-1, self.graph_in_channels)
        node_features = self.node_mapping(node_features)
        node_features = F.normalize(node_features, p=2, dim=1)
        # final, execute classification, segmentation and regression
        # to do
        # t1 = time.time()
        _, roi_loss = self.roi_excute(node_features, features, proposals, image_bin=image_bin)
        # print("roi's time: " + str(time.time()-t1))
        
        losses = {}
        losses.update(vlosses)
        losses.update(roi_loss)
        losses.update({
            "contrastive_alignment_loss": contrastive_loss
        })
        # 还需要添加对比损失
        # to do
        
        # assert 1 == 0
        # print(losses)
        return losses
        
         
    
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
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            assert not torch.jit.is_scripting()
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

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
    
    def _forward_mask(self, features, instances):
        
        if not self.mask_on:
            # https://github.com/pytorch/pytorch/issues/43942
            if self.training:
                assert not torch.jit.is_scripting()
                return {}
            else:
                return instances

        # https://github.com/pytorch/pytorch/issues/46703
        assert hasattr(self, "mask_head")

        if self.training:
            assert not torch.jit.is_scripting()
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            # https://github.com/pytorch/pytorch/issues/41448
            features = dict([(f, features[f]) for f in self.in_features])
            
        return self.mask_head(features, instances)
    
        
    # box_pooler and box_head
    def roi_box_feature_extractor(self, features, proposals):
        # pass
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # print(box_features.shape)
        box_features = self.box_head(box_features)
        
        return box_features
    
    def roi_mask_feature_extractor(self, features, proposals):
        # pass
        features = [features[f] for f in self.in_features]
        mask_features = self.mask_pooler(features, [x.proposal_boxes for x in proposals])
        # mask_features = self.mask_head(mask_features)
        
        return mask_features
    
    
    # bug
    def compute_graph_edge_weight_batch_bug(self, roi_features, final_state, roi_num):
        
        assert roi_features.shape[0] == final_state.shape[0] and roi_features.shape[1] == final_state.shape[1]
        
        # print(roi_features.shape, final_state.shape)
        
        roi_features /= roi_features.norm(dim=-1, keepdim=True) # [N, B, D]
        final_state /= final_state.norm(dim=-1, keepdim=True)  # [N, B, D]
        
        similarity_per_roi = torch.bmm(final_state, roi_features.permute(0, 2, 1))
        
        # print(similarity_per_roi.shape) # [N, B, B]
        # print(similarity_per_roi)
        # print(similarity_per_roi.softmax(-1)[:, ::roi_num, :]) # [N, 1, B]
        # print(similarity_per_roi.softmax(-1)[:, ::roi_num, :][0, 0, :].sum()) 
        # assert 1 == 0
        # return similarity_per_roi.softmax(-1)[:, ::roi_num, :]
        # return similarity_per_roi[:, ::roi_num, :]  # 不取softmax
        return similarity_per_roi[:, ::roi_num, :], similarity_per_roi.softmax(-1)[:, ::roi_num, :]
    
    def compute_graph_edge_weight_batch(self, roi_features, final_state, roi_num):
        assert roi_features.shape[0] == final_state.shape[0] and roi_features.shape[1] == final_state.shape[1]
        # print(roi_features.shape, final_state.shape)
        # [N, roi_num, D] * 2
        # assert 1 == 0
        similarity_per_roi = torch.bmm(final_state, roi_features.permute(0, 2, 1))  # [N, roi_num, roi_num]
        
        return similarity_per_roi[:, ::roi_num, :], similarity_per_roi.softmax(-1)[:, ::roi_num, :]
        
        
        
        
        
    def compute_graph_edge_weight(self, roi_features, final_state):
        # print(roi_features, final_state.shape)
        # roi_features: [N*B, D]
        # final_state: [N*B, D]
        assert roi_features.shape[1] == final_state.shape[1] and roi_features.shape[0] == final_state.shape[0]
        
        # roi_features = roi_features.unsqueeze(1) # [N*B, 1, D]
        # final_state = final_state.unsqueeze(-1)  # [N*B, D, 1]
        
        # weights = torch.bmm(roi_features, final_state).squeeze(-1).squeeze(-1)  # [N*B, ]
        
        # return weights
        
        # follow the similarity of CLIP
        roi_features /= roi_features.norm(dim=-1, keepdim=True)
        final_state /= final_state.norm(dim=-1, keepdim=True)
        
        similarity_per_roi = self.logit_scale * roi_features @ final_state.T
        
        similarity_per_text = similarity_per_roi.t()
        
        
        return similarity_per_text.softmax(-1)
   
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
                bg_idxs = (gt_classes == bg_class_ind).reshape(-1, 1).repeat(1, 5)
            # print(bg_idxs)
            ih, iw = proposals_one_img.image_size
            # im_array = np.array([iw, ih, iw, ih, iw*ih])
            im_array = torch.tensor([iw, ih, iw, ih, iw*ih], dtype=torch.float32)
            
            # abs_pos_one_img = np.zeros((len(boxes), self.rel_pos), dtype=np.float32)
            abs_pos_one_img = torch.zeros((len(boxes), 5), dtype=torch.float32)
            if self.training:
                contrastive_gt.append(torch.as_tensor((gt_classes == bg_class_ind), dtype=torch.float32))
            
            abs_pos_one_img[:, 0] = boxes[:, 0]
            abs_pos_one_img[:, 1] = boxes[:, 1]
            abs_pos_one_img[:, 2] = boxes[:, 2] - 1
            abs_pos_one_img[:, 3] = boxes[:, 3] - 1
            abs_pos_one_img[:, 4] = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
            
            abs_info = abs_pos_one_img / im_array
            # print(abs_info.shape, bg_idxs.shape)
            if self.training:
                abs_info = abs_info * bg_idxs
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