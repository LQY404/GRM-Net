# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import math

from detectron2.layers import cat
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.layers import batched_nms
from detectron2.structures import BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .matcher import build_matcher, HungarianMatcher
from .langencoder.rnn import RNNEncoder, CustomerBert
from .utils.coordconv import CoordConv

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
ROI_HEADS_REGISTRY = Registry("ROI_HEADS_REF")
ROI_HEADS_REGISTRY.__doc__ = """

Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


from .langencoder.phraseAttn import PhraseAttention


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True,
        phraseAttn: PhraseAttention = None,
        proposal_matcher_one2one: HungarianMatcher = None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of classes. Used to label background proposals.
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        # assert proposal_append_gt == True
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=True,
                # allow_low_quality_matches=False
            ),
            "proposal_matcher_one2one": build_matcher(cfg)  # 使用one2one匹配
        }

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
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:

        return self.label_and_sample_proposals_one2one(proposals, targets)

    @torch.no_grad()
    def label_and_sample_proposals_one2one(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        # 匹配的方法使用的是匈牙利算法，匹配的结果是：使得每一个GT有且仅有一个pred与之对应
        # 匹配的标准是：使得class_Loss以及Bbox loss最小时的分配方式
        # 内部使用scipy中的包实现

        indices = self.proposal_matcher_one2one(proposals, targets)

        return indices

        # pos_map = np.zeros((len(proposals['pred_boxes']), len(targets["boxes"])), dtype=np.int32)

        # pos_map[pro_inds, tar_inds] = 1
        # print(pos_map)

        # assert 1 == 0



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

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        sent_dict: Optional[Dict] = None, 
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()

from .langencoder.phraseAttn import RMI, RATTN
from .langencoder.build_lang import init_ref
from .langencoder.rnn import RNNEncoder, CustomerBert
from functools import cmp_to_key

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsGCNRef(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        phraseAttn: PhraseAttention = None,
        proposal_matcher_one2one: HungarianMatcher = None, 
        textencoder = None, 
        cfg = None,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes


        # print("ROI部分需要训练的参数")
        # for name, m in self.named_parameters():
        #     if m.requires_grad:
        #         print(name)

        # self.phraseAttn = phraseAttn

        # concat back
        self.textencoder = textencoder
        # self.textencoder = None

        self.use_bert = cfg.MODEL.USE_BERT or cfg.MODEL.USE_ROBERTA or cfg.MODEL.USE_CLIP
        self.use_clip = cfg.MODEL.USE_CLIP
        
        if self.use_bert:
            self.textencoder = None
            # self.bert = CustomerBert()
        # else:
        #     self.bert = None

        self.rnn_dim = cfg.REF_RNN_DIM
        bidirectional = 2
        self.hn_dim = self.rnn_dim * bidirectional if not self.use_bert else 768
        if self.use_clip:
            self.hn_dim = 512
        # self.hn_dim = 768  # use bert

        self.hs_dim = self.rnn_dim * bidirectional
        self.embedding_dim = cfg.WORD_VEC_DIM if not self.use_bert else 768
        if self.use_clip:
            self.embedding_dim = 512
            
        self.roi_dim = 256

        self.fusion_choice = 1

        # self.topk = cfg.topk
        self.topk = 0
        self.rel_pos = cfg.relpos
        self.pos_norm = cfg.posnorm
        '''
        rel_pos = 3, pos_norm = False must  使用中心面积表示
        rel_pos = 5, pos_norm = False时使用左上角右下角面积表示，pos_norm = True时使用中心面积标准化表示
        rel_pos = 7, pos_norm = True must 使用左上角右下角面积表示
        '''

        if self.rel_pos == 7:
            assert self.pos_norm
        if self.rel_pos == 3:
            assert not self.pos_norm

        self.add_abs = True

        if self.fusion_choice == 1:
            # self.concat_dim = 2 * self.roi_dim + 5 + self.topk*5
            self.concat_dim = 2 * self.roi_dim + self.rel_pos + self.topk*self.rel_pos + self.roi_dim
            # self.concat_dim = 2 * self.roi_dim + 5 + self.topk*(self.roi_dim+5)
            # self.guild_dim = self.roi_dim
            # self.concat_dim = 3 * self.roi_dim + 5

            if not self.add_abs:
                self.concat_dim -= self.rel_pos
            
            self.lfc = nn.Sequential(
                nn.Linear(self.hn_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.roi_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(),
            )
            # del self.lfc
            # self.lfc = nn.Sequential(
            #     nn.Linear(self.hn_dim, self.roi_dim),
            #     # nn.LayerNorm(self.roi_dim),
            #     # nn.Dropout(),
                # # nn.Dropout(0.1 if self.training else 0.0),
                # nn.ReLU(),
                # nn.Dropout(0.1),
                # nn.Linear(self.roi_dim, self.roi_dim),
            #     # nn.LayerNorm(self.roi_dim),
            #     # nn.Dropout(0.5 if self.training else 0.0),
            #     nn.ReLU(),
            # )

            self.mback = nn.Sequential(
                # nn.Conv2d(self.concat_dim, self.roi_dim, 1),
                nn.Conv2d(self.concat_dim, self.roi_dim, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.roi_dim),
                # nn.SyncBatchNorm(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(),
                  
            )
            self.globalVisual = nn.Sequential(
                # nn.Conv2d(self.concat_dim, self.roi_dim, 1),
                nn.Conv2d(self.roi_dim, self.roi_dim, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.roi_dim),
                # nn.SyncBatchNorm(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(), 
                nn.AdaptiveAvgPool2d((1,1))  # average pool
            )
            
            
            self.graphNet = GATNet(1024, 1024, nheads=8)
            self.gcn_init_relu = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU()
            )
            # del self.mback
            # self.mback = nn.Sequential(
            #     nn.Conv2d(self.concat_dim, self.roi_dim, 3, bias=False, padding=1),
            #     nn.BatchNorm2d(self.roi_dim),
            #     nn.ReLU(),
            # )
            
            # self.guild_coord = nn.Sequential(
            #     CoordConv(self.guild_dim, self.guild_dim, kernel_size=3, padding=1, bias=False),
            #     nn.Conv2d(self.guild_dim, self.guild_dim, 3, bias=False, padding=1),
            #     nn.BatchNorm2d(self.guild_dim),
            #     nn.ReLU(),
            # )

        elif self.fusion_choice == -1:
            print("只在RPN阶段融合文本特征")
            self.concat_dim = self.roi_dim + self.rel_pos + self.topk*self.rel_pos
            self.mback = nn.Sequential(
                nn.Conv2d(self.concat_dim, self.roi_dim, 3, bias=False, padding=1),
                nn.BatchNorm2d(self.roi_dim),
                nn.ReLU(),
            )

        elif self.fusion_choice == 0:  # 在roi变成向量的时候再fusion
            self.roi_dim = 1024
            self.concat_dim = self.roi_dim * 2 + 5
            # self.guild_dim = self.roi_dim
            # self.concat_dim = 3 * self.roi_dim + 5
            
            self.lfc = nn.Sequential(
                nn.Linear(self.hn_dim, self.roi_dim),
                # nn.LayerNorm(self.roi_dim),
                # nn.Dropout(),
                # nn.Dropout(0.1 if self.training else 0.0),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.roi_dim, self.roi_dim),
                # nn.LayerNorm(self.roi_dim),
                # nn.Dropout(0.5 if self.training else 0.0),
                nn.ReLU(),
            )

            self.mback = nn.Sequential(
                nn.Linear(self.concat_dim, self.roi_dim),
                nn.BatchNorm1d(self.roi_dim),
                nn.ReLU(),
                nn.Linear(self.roi_dim, self.roi_dim),
                nn.BatchNorm1d(self.roi_dim),
                nn.ReLU(),
            )
            # self.guild_coord = nn.Sequential(
            #     CoordConv(self.guild_dim, self.guild_dim, kernel_size=3, padding=1, bias=False),
            #     nn.Conv2d(self.guild_dim, self.guild_dim, 3, bias=False, padding=1),
            #     nn.BatchNorm2d(self.guild_dim),
            #     nn.ReLU(),
            # )

        elif self.fusion_choice == 2:  # 使guild feature,并使用小型网络进行更新
            assert self.rel_pos == 5
            # assert not self.pos_norm
            
            self.guildNet = GuildNet()
            self.concat_dim = 3 * self.roi_dim + self.rel_pos

            self.lfc = nn.Sequential(
                nn.Linear(self.hn_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.roi_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                nn.ReLU(),
            )

            self.mback = nn.Sequential(
                nn.Conv2d(self.concat_dim, self.roi_dim, 3, bias=False, stride=1, padding=1),
                nn.BatchNorm2d(self.roi_dim),
                nn.ReLU(),
            )

        elif self.fusion_choice == 4:  # 使用GCN
            self.concat_dim = 2 * self.roi_dim + 5
            self.lfc = nn.Sequential(
                nn.Linear(self.hn_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.roi_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                nn.ReLU(),
            )
            
            self.mback = nn.Sequential(
                nn.Conv2d(self.concat_dim, self.roi_dim, 1),
                # nn.Conv2d(self.concat_dim, self.roi_dim, 3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(),
                # nn.Conv2d(self.roi_dim, self.roi_dim, 3, bias=False, stride=1, padding=1),
                # nn.BatchNorm2d(self.roi_dim),
                # nn.ReLU(),
   
            )
            
            # self.gcnnet = GCNNet(1024)
            # self.gatnet = GATNet(1024, 1024)
            self.gatnet = GCN(1024, 2048)
        
        else:
            raise 
        self.proposal_matcher_one2one = proposal_matcher_one2one
        print("roi head for ref")

        # for name, m in self.named_parameters():
        #     if m.requires_grad:
        #         print(name)



    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))

        # ret.update(cls._init_phraseAttn(cfg))
        # ret["textencoder"] = init_ref(cfg)
        ret['cfg'] = cfg

        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    @classmethod
    def _init_phraseAttn(cls, cfg):

        # bidirectional = 2
        # hs_dim = cfg.REF_RNN_DIM * bidirectional
        # phraseAttn = PhraseAttention(hs_dim)

        # return {
        #     "phraseAttn": phraseAttn
        # }
        return {"proposal_matcher_one2one": build_matcher(cfg)}  # 使用one2one匹配


    def matching_label_proposals(self, proposals, targets):
        # 根据proposal与target之间的IOU来打标签
        # 如果某个proposal与target之间IOU大于0.7，则设其GT为1
        # pass

        match_threshold = 0.5
        matching_gt = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):

            match = np.zeros(len(proposals_per_image.proposal_boxes), dtype=np.float32)
            match_quality_matrix = pairwise_iou(
                proposals_per_image.proposal_boxes, targets_per_image.gt_boxes,

            )  
            match_quality_matrix = np.array(match_quality_matrix.cpu())
            # print(match_quality_matrix.shape) # [len(proposals_per_image.proposal_boxes), len(targets_per_image.gt_boxes)]
            max_iou = np.max(match_quality_matrix, axis=1)
            # match[max_iou > match_threshold] = 1
            # match[max_iou > match_threshold] = 1.0
            match = np.where(max_iou > match_threshold, max_iou, match)  # gt为max_iou，更为soft

            matching_gt.append(match)

            # assert 1 == 0

        # matching_gt = cat([torch.tensor(x) for x in matching_gt], dim=0)

        # print(matching_gt.shape)

        # assert 1 == 0

        return matching_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        sent_dict: Optional[Dict] = None, 
        sent_encodes = None,
        image_bin=None,
        bert_hn=None,

    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert not torch.jit.is_scripting()
            assert targets, "'targets' argument is required during training"
            # 需要考虑，是否保持随机采样，还是按照得分情况采样，现在得分只有前背景得分
            # print(len(proposals)) # batch
            # print(proposals[-1])  # RPN.POST_NMS_TOPK_TRAIN，现在设为1000, POSITIVE_FRACTION = 0.5
            # 这一部分可以去除很多的没有意义的proposal，简化计算
            proposals = self.label_and_sample_proposals_more2one(proposals, targets)  # 现在只有ROI_HEADS.BATCH_SIZE_PER_IMAGE个

            # print(proposals[-1])
            # assert 1 == 0
            # 
            # matching_gt = self.matching_label_proposals(proposals, targets)  #这是自己添加的部分，可以不要
            # proposals.set("matching_gt", matching_gt)

            # nproposals = []
            # for index, proposal in enumerate(proposals):
            #     proposal.set("matching_gt", torch.tensor(matching_gt[index], dtype=torch.float32, device=proposal.proposal_boxes.device))
            #     nproposals.append(proposal)
            

            # proposals = nproposals
            # print(proposals[-1])

            # print("#" * 20)
            # print(targets[-1])

            # assert 1 == 0
            
        del targets  # 暂时不能删除

        # print(images.tensor)

        # assert 1 == 0

        if self.training:
            assert not torch.jit.is_scripting()


            # losses, proposals = self._forward_box(features, proposals, sent_dict=sent_dict, sent_encodes=sent_encodes, bert_hn=bert_hn)
            losses, proposals = self._forward_box_ori(features, proposals, sent_dict=sent_dict, sent_encodes=sent_encodes, bert_hn=bert_hn)
            # losses, proposals = self._forward_box_ori(features, proposals, sent_dict=None, sent_encodes=None, bert_hn=None)

            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            # print(len(proposals))

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

            # assert 1 == 0


            losses.update(self._forward_mask(features, proposals, sent_dict=None))


            # losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box_ori(features, proposals, sent_encodes=sent_encodes, sent_dict=sent_dict, bert_hn=bert_hn)
            # pred_instances = self._forward_box_ori(features, proposals, sent_encodes=None, sent_dict=None, bert_hn=None)

            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances, sent_dict)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances], sent_dict: Optional[Dict] = None
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
        instances = self._forward_keypoint(features, instances)
        return instances

    
    @torch.no_grad()
    def _build_edge(self, proposals):
        
        adj_per_img = []
        
        top_k = self.topk
        
        
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach().cpu()  # [B, 4]
            
            # adj_one_img = torch.zeros((boxes.shape[0], boxes.shape[0]))  # [N, N]
            adj_one_img = np.identity(boxes.shape[0])
            # 因为已经排好序了所以可以直接采用下面非常简单的写法
            adj_one_img[: top_k, :] = 1
            adj_one_img[:, : top_k] = 1
            
            adj_per_img.append(torch.tensor(adj_one_img.tolist()))
        
        
        # adj_per_img = torch.tensor(adj_per_img)
        adj_per_img = cat([x.unsqueeze(0) for x in adj_per_img], dim=0)  # [B, N, N]
        
        assert adj_per_img.shape[0] == len(proposals)
        
        # print(adj_per_img.shape) # [B, N, N]
        # assert 1 == 0
        
        return adj_per_img

    @torch.no_grad()
    def _build_adj(self, node_features, edge):
        # pass
        normalized_node_features = node_features.norm(dim=-1, p=2, keepdim=True)
        
        adj = torch.bmm(normalized_node_features, normalized_node_features.T)  # [B, N, N]
        
        # adj = torch.where(edge > 0, adj, 0)
        # 或者这样写： 
        adj = adj * edge
        
        # return adj
        return F.relu(adj)
        

    def _fusion4(self, features, proposals, hn):
        print("GAT")
        N, C, vh, vw = features[-1].shape
        
        abs_pos_per_img = self.compute_abs_pos_concurrent(proposals)
        abs_pos_per_img = abs_pos_per_img.to(features[-1].device)
        
        # 构造伪邻接矩阵
        edge = self._build_edge(proposals).to(features[-1].device)
        
        
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [N*ROI_HEADS.BATCH_SIZE_PER_IMAGE, 256, 7, 7] , ROI_HEADS.BATCH_SIZE_PER_IMAGE根据config而定，我们设为256
        roi_shape = box_features.shape[-2:]
        
        assert box_features.shape[0] % N == 0  # 0000000
        roi_num = box_features.shape[0] // N
        # nhn = []
        hn = self.lfc(hn)
        hn = F.normalize(hn, p=2, dim=-1)

        hn = hn.repeat_interleave(repeats=roi_num, dim=0)  # [N*roi_num, D]
        hn = hn.reshape(box_features.shape[0], -1, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])  # [N*roi_num, D, roih, roiw]
        
        # for index in range(N):
        #     hn_i = hn[index, :].reshape(1, -1, 1, 1).repeat(roi_num, 1, roi_shape[0], roi_shape[1])
        #     nhn.append(hn_i)

        # hn = cat([x for x in nhn], dim=0)
        
        abs_pos_per_img = abs_pos_per_img.reshape(-1, 5, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])
        print(box_features.shape, abs_pos_per_img.shape, hn.shape)
        
        box_features = torch.cat((box_features, abs_pos_per_img, hn), dim=1) # [N*256, 5+hn_dim+5*topk, 7, 7]
        
        box_features = self.mback(box_features)
        box_features = F.normalize(box_features, p=2, dim=1)
        
        box_features = self.box_head(box_features)
        
        # box_features = self.gcnnet(box_features)
        # 如果使用GAT，需要将box_features变成[B, N, 1024]的形式
        n_box_features = []
        for i in range(N):
            n_box_features.append(box_features[i*roi_num: (i+1)*roi_num, :].unsqueeze(0))
            
        n_box_features = cat(n_box_features, dim=0) # [B, N, D]
        
        adj = self._build_adj(n_box_features.data.detach(), edge).to(features[-1].device)
        
        # print(n_box_features.shape)  # [B, N, 1024]
        # assert 1 == 0
        
        # box_features = self.gatnet(n_box_features) # [B, N, 1024]
        Nbox_features = self.gatnet(n_box_features, adj=adj) # [B, N, 1024]
        
        # 是否要加上残差
        
        Nbox_features = cat([Nbox_features[i] for i in range(N)], dim=0)
        
        box_features = box_features + Nbox_features
        # print(box_features.shape)
        # assert 1 == 0
        
        # 这个地方可能需要添加一个归一化促进模型收敛
        box_features = F.normalize(box_features, p=2, dim=1)
        
        return box_features
        
    def _fusion_abs(self, features, proposals):
        N, C, vh, vw = features[-1].shape

        # 为每个ROI获取位置编码
        abs_pos_per_img = self.compute_abs_pos(proposals)
        rel_pos_per_img = self.fetch_obj_proposals(proposals)  # [N*B, TOPK*5]
        

        # abs_pos_per_img.to(features[-1].device)
        # print(abs_pos_per_img.shape) # [N*256, 5]
        abs_pos_per_img = abs_pos_per_img.to(features[-1].device)
        rel_pos_per_img = rel_pos_per_img.to(features[-1].device)


        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [N*ROI_HEADS.BATCH_SIZE_PER_IMAGE, 256, 7, 7] , ROI_HEADS.BATCH_SIZE_PER_IMAGE根据config而定，我们设为256

        roi_shape = box_features.shape[-2:]

        abs_pos_per_img = abs_pos_per_img.reshape(-1, 5, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])
        rel_pos_per_img = rel_pos_per_img.reshape(-1, self.topk * 5, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])


        box_features = torch.cat((box_features, abs_pos_per_img, rel_pos_per_img), dim=1) # [N*256, 5+hn_dim+5*topk, 7, 7]

        box_features = self.mback(box_features)
        box_features = F.normalize(box_features, p=2, dim=1)
        
        box_features = self.box_head(box_features)

        return box_features

    def _fusion1(self, features, proposals, hn, images=None):
        N, C, vh, vw = features[-1].shape

        # 为每个ROI获取位置编码
        if self.add_abs:
            abs_pos_per_img = self.compute_abs_pos_concurrent(proposals)  # [N * B, 5]

        if self.topk > 0:
            raise
            rel_pos_per_img = self.fetch_obj_proposals_concurrent(proposals)  # [N*B, TOPK*5]
            # rel_pos_per_img = self.fetch_obj_proposals_concurrent_mattnet(proposals)

        # rel_pos_per_img, rel_pos_ids_per_img = retry_if_cuda_oom(self.fetch_obj_proposals)(proposals)  # [N*B, TOPK*5]
        # rel_pos_per_img, rel_pos_ids_per_img = retry_if_cuda_oom(self.fetch_obj_proposals_v2)(proposals)  # [N*B, TOPK*5]

        # abs_pos_per_img, rel_pos_per_img = retry_if_cuda_oom(self.fetch_abs_rel_pos)(proposals)
        # print(rel_pos_per_img.shape)  # [N*B, topk*5]
        # print(rel_pos_ids_per_img.shape)  # [N*B, topk]
        # assert 1 == 0

        # print(abs_pos_per_img.shape) # [N * B, 5]
        # print(rel_pos_per_img.shape) # [N*B, TOPK*5]

        # assert 1 == 0
        
        # abs_pos_per_img.to(features[-1].device)
        # print(abs_pos_per_img.shape) # [N*256, 5]
        if self.add_abs:
            abs_pos_per_img = abs_pos_per_img.to(features[-1].device)

        if self.topk > 0:
            rel_pos_per_img = rel_pos_per_img.to(features[-1].device)

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [N*ROI_HEADS.BATCH_SIZE_PER_IMAGE, 256, 7, 7] , ROI_HEADS.BATCH_SIZE_PER_IMAGE根据config而定，我们设为256
        # print(box_features.shape)  # [N*B, C, vh, vw]

        #############
        # 添加全局信息
        globalFeature = self.globalVisual(features[-2]).squeeze(-1).squeeze(-1) # [N, roi_hidden_dim]
        globalFeature = F.normalize(globalFeature, p=2, dim=1) # 添加一个normalization
        
        ############
        
        # selected_rel_features = box_features[rel_pos_ids_per_img, :, :].data.detach()
        # print(selected_rel_features.shape)  # [N*B, topk, C, vh, vw]

        # 将每个rel的特征和其与sub的相对位置信息concat起来

        roi_shape = box_features.shape[-2:]

        # rel_infos = []
        # topk = selected_rel_features.shape[1]
        # for ik in range(topk):
        #     irel_feature = selected_rel_features[:, ik, :, :, :]  # [N*B, C, vh, vw]
        #     irel_pos = rel_pos_per_img[:, ik*5: (ik+1)*5]
        #     irel_pos = irel_pos.reshape(-1, 5, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])
        #     # print(irel_feature.shape)  # [N*B, C, vh, vw]
        #     # print(irel_pos.shape)  # [N*B, 5]
        #     # assert 1 == 0

        #     rel_info = torch.cat((irel_feature, irel_pos), dim=1)  # [N*B, C+5, vh, vw]

        #     rel_infos.append(rel_info)

        # rel_infos = torch.cat([x for x in rel_infos], dim=1)  # [N*B, (C+5)*topk, vh, vw]
        # print(rel_infos.shape)
        # assert 1 == 0
        if self.add_abs:
            abs_pos_per_img = abs_pos_per_img.reshape(-1, self.rel_pos, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])

        if self.topk > 0:
            rel_pos_per_img = rel_pos_per_img.reshape(-1, self.topk * self.rel_pos, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])

        
        # assert N == 1
        # roi_num = box_features.shape[0] // N
        assert box_features.shape[0] % N == 0  # 0000000
        roi_num = box_features.shape[0] // N

        # nhn = []
        # uimages = []
        
        

        # assert 1 == 0
        hn2 = self.lfc(hn)
        # hn = F.normalize(hn, p=2, dim=-1)

        hn2 = hn2.repeat_interleave(repeats=roi_num, dim=0)  # [N*roi_num, D]
        hn2 = hn2.reshape(box_features.shape[0], -1, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])  # [N*roi_num, D, roih, roiw]
        
        globalFeature = globalFeature.repeat_interleave(repeats=roi_num, dim=0)  # [N*B, roi_hidden_dim]
        globalFeature = globalFeature.reshape(box_features.shape[0], -1, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])
        
        # for index in range(N):
        #     hn_i = hn[index, :].reshape(1, -1, 1, 1).repeat(roi_num, 1, roi_shape[0], roi_shape[1])
        #     nhn.append(hn_i)
        # hn = cat([x for x in nhn], dim=0)
        
        # uimages = cat([x for x in uimages], dim=0)
        if self.add_abs and self.topk > 0:    
            print(box_features.shape, hn2.shape, abs_pos_per_img.shape, rel_pos_per_img.shape, globalFeature.shape)
        elif not self.add_abs and self.topk > 0:
            print(box_features.shape, hn2.shape, rel_pos_per_img.shape, globalFeature.shape)
        elif self.add_abs and self.topk == 0:
            print(box_features.shape, hn2.shape, abs_pos_per_img.shape, globalFeature.shape)
        else:
            print(box_features.shape, hn2.shape, globalFeature.shape)
        
        # box_features = F.normalize(box_features, p=2, dim=1)
        # box_features = torch.cat((box_features, hn, abs_pos_per_img), dim=1) # [N*256, 5+hn_dim+256, 7, 7]
        if self.add_abs and self.topk > 0:
            box_features = torch.cat((box_features, hn2, abs_pos_per_img, rel_pos_per_img, globalFeature), dim=1) # [N*256, 5+hn_dim+256, 7, 7]
        elif not self.add_abs and self.topk > 0:
            box_features = torch.cat((box_features, hn2, rel_pos_per_img, globalFeature), dim=1)
        elif self.add_abs and self.topk == 0:
            box_features = torch.cat((box_features, hn2, abs_pos_per_img, globalFeature), dim=1)
        else:
            box_features = torch.cat((box_features, hn2, globalFeature), dim=1)
        
        # box_features = torch.cat((box_features, hn, abs_pos_per_img, rel_infos), dim=1) # [N*256, 5+hn_dim+256, 7, 7]

        # add guild image
        # print(uimages.shape)


        # uimages = self.guild_coord(uimages)
        
        # assert uimages.shape[-2: ] == roi_shape
        # guild = torch.cat((box_features, uimages, hn, abs_pos_per_img), dim=1)  # [M, 5+hn_dim+256+, 7, 7]

        #################

        # print(box_features.shape)
        box_features = self.mback(box_features)
        # box_features = self.mback(guild)
        # box_features = F.normalize(box_features, p=2, dim=1)

        box_features = self.box_head(box_features)
        
        box_features = box_features.reshape(N, roi_num, 1024)
        # print(box_features.shape)
        box_features = box_features + self.graphNet(box_features)
        
        box_features = box_features.reshape(-1, 1024)
        box_features = self.gcn_init_relu(box_features)
        
        box_features = F.normalize(box_features, p=2, dim=1)

        return box_features
    
    def _fusion3(self, features, proposals, hn, hs, embedding, words, images=None):
        N, C, vh, vw = features[-1].shape

        # 为每个ROI获取位置编码
        abs_pos_per_img = retry_if_cuda_oom(self.compute_abs_pos)(proposals)  # [N * B, 5]
        abs_pos_per_img = abs_pos_per_img.to(features[-1].device)

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [N*ROI_HEADS.BATCH_SIZE_PER_IMAGE, 256, 7, 7] , ROI_HEADS.BATCH_SIZE_PER_IMAGE根据config而定，我们设为256
        box_features = self.box_head(box_features) 
        # print(box_features.shape)  # [N*B, 1024]

        # assert 1 == 0
        assert box_features.shape[0] % N == 0  # 0000000
        roi_num = box_features.shape[0] // N


        nhn = []
        hn = self.lfc(hn)
        for index in range(N):
            hn_i = hn[index, :].reshape(1, -1).repeat(roi_num, 1)
            nhn.append(hn_i)
        hn = cat([x for x in nhn], dim=0)  # [N*B, dim]

        box_features = torch.cat((box_features, hn, abs_pos_per_img), dim=-1)

        # print(box_features.shape)

        # assert 1 == 0
        box_features = self.mback(box_features)


        return box_features

    def _fusion_guild(self, features, proposals, hn, images=None):
        N, C, _, _ = features[-1].shape

        abs_pos_per_img = retry_if_cuda_oom(self.compute_abs_pos)(proposals)  # [N * B, 5]
        abs_pos_per_img = abs_pos_per_img.to(features[-1].device)

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        roi_shape = box_features.shape[-2:]

        abs_pos_per_img = abs_pos_per_img.reshape(-1, self.rel_pos, 1, 1).repeat(1, 1, roi_shape[0], roi_shape[1])
        
        assert box_features.shape[0] % N == 0  # 0000000
        roi_num = box_features.shape[0] // N

        nhn = []
        hn = self.lfc(hn)
        # hn = F.normalize(hn, p=2, dim=-1)

        # choice guild feature, we use P3
        guild_feature = features[1].data.detach()
        guild_feature = torch.nn.functional.interpolate(guild_feature, size=roi_shape, mode='bilinear', align_corners=True)
        guild_feature = self.guildNet(guild_feature)  # 使用小型网络进行更新

        nguild_feature = []

        for index in range(N):
            hn_i = hn[index, :].reshape(1, -1, 1, 1).repeat(roi_num, 1, roi_shape[0], roi_shape[1])
            nhn.append(hn_i)

            guild_featurei = guild_feature[index, :, :, :].reshape(1, -1, roi_shape[0], roi_shape[1]).repeat(roi_num, 1, 1, 1)
            nguild_feature.append(guild_featurei)
        
        hn = cat([x for x in nhn], dim=0)
        guild_feature = cat([x for x in nguild_feature], dim=0)


        print(box_features.shape, hn.shape, abs_pos_per_img.shape, guild_feature.shape)
        box_features = torch.cat((box_features, hn, abs_pos_per_img, guild_feature), dim=1) # [N*256, 5+hn_dim+256, 7, 7]

        box_features = self.mback(box_features)
        box_features = self.box_head(box_features)

        return box_features


    def _fusion2(self, features, proposals, hn):
        N, C, vh, vw = features[-1].shape

        # 为每个ROI获取位置编码
        abs_pos_per_img = self.compute_abs_pos(proposals)
        # abs_pos_per_img.to(features[-1].device)
        # print(abs_pos_per_img.shape) # [N*256, 5]
        abs_pos_per_img = abs_pos_per_img.to(features[-1].device)

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [N*ROI_HEADS.BATCH_SIZE_PER_IMAGE, 256, 7, 7] , ROI_HEADS.BATCH_SIZE_PER_IMAGE根据config而定，我们设为256
        abs_pos_per_img = abs_pos_per_img.reshape(-1, 5, 1, 1).repeat(1, 1, box_features.shape[2], box_features.shape[-1])

        assert N == 1
        roi_num = box_features.shape[0] // N

        nhn = []
        for index in range(N):
            hn_i = hn[index, :].reshape(1, -1).repeat(roi_num, 1)
            nhn.append(hn_i)

        hn = cat([x for x in nhn], dim=0)  # [N*256, C]

        print(box_features.shape, hn.shape, abs_pos_per_img.shape)
        
        box_features = torch.cat((box_features, abs_pos_per_img), dim=1) # [N*256, C+5, 7, 7]
        box_features = F.normalize(box_features, p=2, dim=1)
        box_features = self.bacti(self.bconv(box_features))

        box_features = self.box_head(box_features)  # [N*256, 1024]

        assert self.fusion_choice == 2

        mbox_features = self.mapacti(self.map(box_features) ) # [N*256, C]

        feature_matching = torch.bmm(hn.reshape(hn.shape[0], 1, -1), mbox_features.reshape(mbox_features.shape[0], -1, 1)) # [N*256, 1, 1]

        feature_matching_score = feature_matching.squeeze(-1).squeeze(-1)

        return box_features, feature_matching_score

    # def _fusion3(self, features, hs, embedding, context, words):

    def _forward_box_ori(self, features: Dict[str, torch.Tensor], proposals: List[Instances], sent_dict: Optional[Dict] = None, targets: Optional[List[Instances]] = None, sent_encodes=None, image_bin=None, bert_hn=None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        
        # box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # box_features = self.box_head(box_features)

        if sent_dict is not None:
            # print("共用一个rnn")
            assert self.textencoder is None and sent_encodes is None
            hs, hn, embedding, words = sent_dict["hs"], sent_dict["hn"], sent_dict["embedding"], sent_dict["words"]

        elif sent_encodes is not None:
            assert self.textencoder is not None and sent_encodes is not None
            lang_dict = self.textencoder(sent_encodes)
            hs, hn, embedding, words = lang_dict["output"], lang_dict['final_output'], lang_dict['embedded'], sent_encodes

        elif bert_hn is not None:
            # print("不在ROI融合文本特征")
            # pass
            assert self.use_bert

            hn = bert_hn
            hs, embedding, words = None, None, None

        # use bert
        # hn = self.textencoder(sent_encodes)


        # 直接在通道维度拼接
        if sent_dict is not None or sent_encodes is not None or bert_hn is not None:
            if not self.training:
                print("fusion text on roi")
            assert self.fusion_choice == 1
            assert self.topk == 0 # 使用GCN的时候不用GRM
            
            if self.fusion_choice == 1:
                box_features = self._fusion1(features, proposals, hn, images=None)  # [N*ROI_HEADS.BATCH_SIZE_PER_IMAGE, 1024]
            # box_features = self._fusion3(features, proposals, hn, hs, embedding, words)
            # box_features = self._fusion_guild(features, proposals, hn, images=None)
            elif self.fusion_choice == 4:
                box_features = self._fusion4(features, proposals, hn)
        
        else: # original，只添加位置信息
            # raise
            box_features = self._fusion_abs(features, proposals)

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
            return losses, proposals
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    @torch.no_grad()
    def _add_delta_box(self, deltas, proposals, weights = (10.0, 10.0, 5.0, 5.0), scale_clamp = _DEFAULT_SCALE_CLAMP):
        

        deltas = deltas.float()  # ensure fp32 for decoding precision
        boxes = proposals.to(deltas.dtype)
        # print(boxes)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=scale_clamp)
        dh = torch.clamp(dh, max=scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances], sent_dict: Optional[Dict] = None):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
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
            # instances, _ = select_foreground_proposals(instances, 0)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            # https://github.com/pytorch/pytorch/issues/41448
            features = dict([(f, features[f]) for f in self.mask_in_features])
            
        return self.mask_head(features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            if self.training:
                assert not torch.jit.is_scripting()
                return {}
            else:
                return instances

        assert hasattr(self, "keypoint_head")

        if self.training:
            assert not torch.jit.is_scripting()
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = dict([(f, features[f]) for f in self.keypoint_in_features])
        return self.keypoint_head(features, instances)     
        

    def fetch_obj_proposals_old(self, proposals):  # 不管同类还是异类
        rel_pos_per_img = []
        rel_pos_ids_per_img = []

        top_k = self.topk
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach().cpu()  # [B, 4]
            # print(proposals_one_img)
            # assert 1 == 0
            # objectness_logits = proposals_one_img.objectness_logits.data.detach().cpu()  # 此时已经按照从大到小排列了
            # print(objectness_logits[: 20])
            # print(boxes.shape)
            # assert 1 == 0
            # rel_pos_one_img = np.empty((len(boxes), top_k, 5), dtype=np.float32)
            rel_pos_one_img = []
            # rel_pos_ids_one_img = []
            # print(boxes.shape)
            assert  self.rel_pos == 5
            for ox, obox in enumerate(boxes):  
                ox1, oy1, ox2, oy2 = obox
                rx, ry, rw, rh = ox1, oy1, ox2-ox1, oy2-oy1
                arx, ary = rx+rw*0.5, ry+rh*0.5

                box_ids = list(range(len(boxes)))
                # print("RPN输出proposal数目：", len(box_ids))
                rel_pos = []
                re_index = 0
                for cx in box_ids:
                    if cx == ox:
                        continue
                    if re_index >= top_k:
                        break

                    ibox = boxes[cx]
                    ix, iy, iw, ih = ibox[0], ibox[1], ibox[2]-ibox[0], ibox[3]-ibox[1]
                    aix = ix + 0.5*iw
                    aiy = iy + 0.5*ih
                    rel_pos.append([(ix-arx)/rw, (iy-ary)/rh, (ix+iw-arx)/rw, (iy+ih-ary)/rh, iw*ih/(rw*rh)])
                    
                    re_index += 1

                assert re_index == top_k
                # assert len(rel_pos) == top_k
                rel_pos = cat([torch.tensor(x) for x in rel_pos])
                # print(rel_pos.shape)
                # assert 1 == 0
                rel_pos_one_img.append(rel_pos)

            rel_pos_one_img = cat([x.unsqueeze(0) for x in rel_pos_one_img])
            rel_pos_per_img.append(rel_pos_one_img)


        rel_pos_per_img = cat([rel_pos_one_img for rel_pos_one_img in rel_pos_per_img])
        return rel_pos_per_img


    def fetch_obj_proposals(self, proposals):  # 不管同类还是异类
        rel_pos_per_img = []
        rel_pos_ids_per_img = []

        top_k = self.topk
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach().cpu()  # [B, 4]
            # print(proposals_one_img)
            # assert 1 == 0
            # objectness_logits = proposals_one_img.objectness_logits.data.detach().cpu()  # 此时已经按照从大到小排列了
            # print(objectness_logits[: 20])
            # print(boxes.shape)
            # assert 1 == 0
            # rel_pos_one_img = np.empty((len(boxes), top_k, 5), dtype=np.float32)
            rel_pos_one_img = []
            # rel_pos_ids_one_img = []
            # print(boxes.shape)
            # assert  self.rel_pos == 5
            for ox, obox in enumerate(boxes):  
                ox1, oy1, ox2, oy2 = obox
                rx, ry, rw, rh = ox1, oy1, ox2-ox1, oy2-oy1
                arx, ary = rx+rw*0.5, ry+rh*0.5

                box_ids = list(range(len(boxes)))
                # box_ids = sorted(box_ids, key=cmp_to_key(compare))
                # box_ids = sorted(box_ids, lambda id1, id2: compare(id1, id2))


                # rel_pos = np.empty((top_k, 5), dtype=np.float32)
                # rel_pos = []
                # rel_pos_ids = []

                rel_pos = torch.zeros((self.topk, self.rel_pos))
                # rel_pos_ids = torch.ones((self.topk)) + ox
                re_index = 0
                for cx in box_ids:
                    if cx == ox:
                        continue
                    if re_index >= top_k:
                        break

                    ibox = boxes[cx]
                    ix, iy, iw, ih = ibox[0], ibox[1], ibox[2]-ibox[0], ibox[3]-ibox[1]
                    # if ih*iw <= 16:  # 太小的不要
                    #     continue

                    # rel_pos_ids.append(cx)
                    # rel_pos_ids[re_index] = cx
                    aix, aiy = ix + iw*0.5, iy + ih*0.5
                    # rel_pos[re_index, :] = np.array([(ix-ox)/rw, (iy-ry)/rh, (ix+iw-ox)/rw, (iy+ih-ry)/rh, iw*ih/(rw*rh)])
                    # rel_pos.append([(ix-arx)/rw, (iy-ary)/rh, (ix+iw-arx)/rw, (iy+ih-ary)/rh, iw*ih/(rw*rh)])
                    if self.rel_pos == 7:
                        assert self.pos_norm
                        rel_pos[re_index, 0] = (ix-arx)/rw
                        rel_pos[re_index, 1] = (iy-ary)/rh
                        rel_pos[re_index, 2] = (ix+iw-arx)/rw
                        rel_pos[re_index, 3] = (iy+ih-ary)/rh
                        rel_pos[re_index, 4] = (iw*ih)/(rw*rh)
                        rel_pos[re_index, 5] = 1. / rw
                        rel_pos[re_index, 6] = 1. / rh

                    elif self.rel_pos == 5:
                        if not self.pos_norm:
                            rel_pos[re_index, 0] = (ix-arx)/rw
                            rel_pos[re_index, 1] = (iy-ary)/rh
                            rel_pos[re_index, 2] = (ix+iw-arx)/rw
                            rel_pos[re_index, 3] = (iy+ih-ary)/rh
                            rel_pos[re_index, 4] = (iw*ih)/(rw*rh)
                        else:
                            rel_pos[re_index, 0] = (aix - arx) / rw
                            rel_pos[re_index, 1] = (aiy - ary) / rh
                            rel_pos[re_index, 2] = (iw*ih)/(rw*rh)
                            rel_pos[re_index, 3] = 1. / rw
                            rel_pos[re_index, 4] = 1. / rh

                    elif self.rel_pos == 3:
                        assert not self.pos_norm

                        rel_pos[re_index, 0] = (aix - arx) / rw
                        rel_pos[re_index, 1] = (aiy - ary) / rh
                        rel_pos[re_index, 2] = (iw*ih)/(rw*rh)

                    else:
                        raise

                    re_index += 1
                
                assert re_index == top_k
                # assert len(rel_pos) == top_k

                # for x in rel_pos:
                #     print(x.shape)
                # print(rel_pos)
                # rel_pos = cat([torch.tensor(x) for x in rel_pos])  # [25, ]
                rel_pos = cat([x for x in rel_pos])
                # rel_pos_ids = cat([torch.tensor([x]) for x in rel_pos_ids])
                # rel_pos_ids = cat([x for x in rel_pos_ids])

                # print(rel_pos.shape)

                rel_pos_one_img.append(rel_pos)
                # rel_pos_ids_one_img.append(rel_pos_ids)
            # rel_pos_one_img = np.array(rel_pos_one_img)
            # print(rel_pos_one_img.shape)
            rel_pos_one_img = cat([x.unsqueeze(0) for x in rel_pos_one_img])
            # rel_pos_ids_one_img = cat([x.unsqueeze(0) for x in rel_pos_ids_one_img])

            # print(rel_pos_one_img.shape)
            rel_pos_per_img.append(rel_pos_one_img)
            # rel_pos_ids_per_img.append(rel_pos_ids_one_img)


        # rel_pos_per_img = cat([torch.tensor(rel_pos_one_img) for rel_pos_one_img in rel_pos_per_img])
        rel_pos_per_img = cat([rel_pos_one_img for rel_pos_one_img in rel_pos_per_img])

        # rel_pos_ids_per_img = cat([x for x in rel_pos_ids_per_img])
        # print(rel_pos_ids_per_img.shape)
        
        # print(rel_pos_per_img)  # [N * B, top_k * 5]


        # assert 1 == 0

        return rel_pos_per_img 
        # return rel_pos_per_img, rel_pos_ids_per_img

    def fetch_obj_proposals_v2(self, proposals):  # 不管同类还是异类
        rel_pos_per_img = []
        rel_pos_ids_per_img = []

        top_k = self.topk
        assert top_k > 0
        for index, proposals_one_img in enumerate(proposals):

            boxes = proposals_one_img.proposal_boxes.tensor.data.detach().cpu()  # [B, 4]
            # print(proposals_one_img)

            # assert 1 == 0
            # objectness_logits = proposals_one_img.objectness_logits.data.detach().cpu()  # 此时已经按照从大到小排列了
            # print(objectness_logits[: 20])
            # print(boxes.shape)
            # assert 1 == 0

            # rel_pos_one_img = np.empty((len(boxes), top_k, 5), dtype=np.float32)
            rel_pos_one_img = []

            rel_pos_ids_one_img = []

            # print(boxes.shape)

            for ox, obox in enumerate(boxes):  
                ox1, oy1, ox2, oy2 = obox
                rx, ry, rw, rh = ox1, oy1, ox2-ox1, oy2-oy1
                arx, ary = rx+rw/2, ry+rh/2

                box_ids = list(range(len(boxes)))
                # box_ids = sorted(box_ids, key=cmp_to_key(compare))
                # box_ids = sorted(box_ids, lambda id1, id2: compare(id1, id2))


                # rel_pos = np.empty((top_k, 5), dtype=np.float32)
                # rel_pos = []
                # rel_pos_ids = []

                # rel_pos = torch.zeros((self.topk, 5))
                rel_pos = torch.zeros((self.topk, self.rel_pos))
                rel_pos_ids = torch.ones((self.topk)) + ox
                re_index = 0
                for cx in box_ids:
                    if cx == ox:
                        continue
                    if re_index >= top_k:
                        break

                    ibox = boxes[cx]
                    ix, iy, iw, ih = ibox[0], ibox[1], ibox[2]-ibox[0], ibox[3]-ibox[1]
                    # if ih*iw <= 16:  # 太小的不要
                    #     continue

                    # rel_pos_ids.append(cx)
                    rel_pos_ids[re_index] = cx
                    aix, aiy = ix + iw/2, iy + ih/2
                    # rel_pos[re_index, :] = np.array([(ix-ox)/rw, (iy-ry)/rh, (ix+iw-ox)/rw, (iy+ih-ry)/rh, iw*ih/(rw*rh)])
                    # rel_pos.append([(ix-arx)/rw, (iy-ary)/rh, (ix+iw-arx)/rw, (iy+ih-ary)/rh, iw*ih/(rw*rh)])
                    if self.rel_pos == 7:
                        assert self.pos_norm
                        rel_pos[re_index, 0] = (arx - ix) / rw
                        rel_pos[re_index, 1] = (ary - iy) / rh
                        rel_pos[re_index, 2] = (arx - ix - iw) / rw
                        rel_pos[re_index, 3] = (ary - iy - ih) / rh
                        rel_pos[re_index, 4] = (iw*ih) / (rw*rh)
                        rel_pos[re_index, 5] = 1. / rw
                        rel_pos[re_index, 6] = 1. / rh

                    elif self.rel_pos == 5:
                        if not self.pos_norm:
                            rel_pos[re_index, 0] = (arx - ix) / rw
                            rel_pos[re_index, 1] = (ary - iy) / rh
                            rel_pos[re_index, 2] = (arx - ix - iw) / rw
                            rel_pos[re_index, 3] = (ary - iy - ih) / rh
                            rel_pos[re_index, 4] = (iw*ih) / (rw*rh)
                        else:
                            rel_pos[re_index, 0] = (arx - aix) / rw
                            rel_pos[re_index, 1] = (ary - aiy) / rh
                            rel_pos[re_index, 2] = (iw*ih) / (rw*rh)
                            rel_pos[re_index, 3] = 1. / rw
                            rel_pos[re_index, 4] = 1. / rh

                    elif self.rel_pos == 3:
                        assert not self.pos_norm

                        rel_pos[re_index, 0] = (arx - aix) / rw
                        rel_pos[re_index, 1] = (ary - aiy) / rh
                        rel_pos[re_index, 2] = (iw*ih) / (rw*rh)

                    else:
                        raise

                    re_index += 1
                
                assert re_index == top_k
                # assert len(rel_pos) == top_k

                # for x in rel_pos:
                #     print(x.shape)

                # rel_pos = cat([torch.tensor(x) for x in rel_pos])  # [25, ]
                rel_pos = cat([x for x in rel_pos])
                rel_pos_ids = cat([torch.tensor([x]) for x in rel_pos_ids])
                # rel_pos_ids = cat([x for x in rel_pos_ids])

                # print(rel_pos.shape)

                rel_pos_one_img.append(rel_pos)
                rel_pos_ids_one_img.append(rel_pos_ids)
            
            # rel_pos_one_img = np.array(rel_pos_one_img)
            # print(rel_pos_one_img.shape)
            rel_pos_one_img = cat([x.unsqueeze(0) for x in rel_pos_one_img])
            rel_pos_ids_one_img = cat([x.unsqueeze(0) for x in rel_pos_ids_one_img])

            # print(rel_pos_one_img.shape)
            rel_pos_per_img.append(rel_pos_one_img)
            rel_pos_ids_per_img.append(rel_pos_ids_one_img)


        # rel_pos_per_img = cat([torch.tensor(rel_pos_one_img) for rel_pos_one_img in rel_pos_per_img])
        rel_pos_per_img = cat([rel_pos_one_img for rel_pos_one_img in rel_pos_per_img])

        rel_pos_ids_per_img = cat([x for x in rel_pos_ids_per_img])
        # print(rel_pos_ids_per_img.shape)
        
        # print(rel_pos_per_img)  # [N * B, top_k * 5]


        # assert 1 == 0

        # return rel_pos_per_img 
        return rel_pos_per_img, rel_pos_ids_per_img


    def fetch_abs_rel_pos(self, proposals):
        abs_pos_per_img = []

        rel_pos_per_img = []

        top_k = self.topk
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach().cpu()

            ih, iw = proposals_one_img.image_size
            abs_pos_one_img = np.empty((len(boxes), self.rel_pos), dtype=np.float32)

            rel_pos_one_img = []

            for ox, obox in enumerate(boxes):
                ox1, oy1, ox2, oy2 = obox
                rx, ry, rw, rh = ox1, oy1, ox2-ox1, oy2-oy1
                arx, ary = rx+rw/2, ry+rh/2

                box_ids = list(range(len(boxes)))

                if self.rel_pos == 7:
                    assert self.pos_norm
                    abs_pos = np.array([arx/iw, ary/ih, (ox1+rw-1)/iw, (oy1+rh-1)/ih, rw*rh/(iw*ih), 1./iw, 1./ih], np.float32)

                elif self.rel_pos == 5:
                    if not self.pos_norm :
                        abs_pos = np.array([arx/iw, ary/ih, (rx+rw-1)/iw, (ry+rh-1)/ih, rw*rh/(iw*ih)], np.float32)
                    else:    
                        abs_pos = np.array([arx/iw, ary/ih, rw*rh/(iw*ih), 1.0/iw, 1.0/ih], np.float32)  #

                # abs_pos = np.array([cx/iw, cy/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)], np.float32)
                elif self.rel_pos == 3:
                    assert not self.pos_norm
                    abs_pos = np.array([arx/iw, ary/ih, rw*rh/(iw*ih)], np.float32)  

                else:
                    raise

                abs_pos_one_img[ox] = abs_pos

                # 相对信息
                rel_pos = torch.zeros((self.topk, self.rel_pos))
                re_index = 0
                for cx in box_ids:
                    if cx == ox:
                        continue
                    if re_index >= top_k:
                        break

                    ibox = boxes[cx]
                    ix, iy, iw, ih = ibox[0], ibox[1], ibox[2]-ibox[0], ibox[3]-ibox[1]
                    # if ih*iw <= 16:  # 太小的不要
                    #     continue

                    # rel_pos_ids.append(cx)
                    # rel_pos_ids[re_index] = cx
                    aix, aiy = ix + iw/2, iy + ih/2
                    # rel_pos[re_index, :] = np.array([(ix-ox)/rw, (iy-ry)/rh, (ix+iw-ox)/rw, (iy+ih-ry)/rh, iw*ih/(rw*rh)])
                    # rel_pos.append([(ix-arx)/rw, (iy-ary)/rh, (ix+iw-arx)/rw, (iy+ih-ary)/rh, iw*ih/(rw*rh)])
                    if self.rel_pos == 7:
                        assert self.pos_norm
                        rel_pos[re_index, 0] = (ix-arx)/rw
                        rel_pos[re_index, 1] = (iy-ary)/rh
                        rel_pos[re_index, 2] = (ix+iw-arx)/rw
                        rel_pos[re_index, 3] = (iy+ih-ary)/rh
                        rel_pos[re_index, 4] = (iw*ih)/(rw*rh)
                        rel_pos[re_index, 5] = 1. / rw
                        rel_pos[re_index, 6] = 1. / rh

                    elif self.rel_pos == 5:
                        if not self.pos_norm:
                            rel_pos[re_index, 0] = (ix-arx)/rw
                            rel_pos[re_index, 1] = (iy-ary)/rh
                            rel_pos[re_index, 2] = (ix+iw-arx)/rw
                            rel_pos[re_index, 3] = (iy+ih-ary)/rh
                            rel_pos[re_index, 4] = (iw*ih)/(rw*rh)
                        else:
                            rel_pos[re_index, 0] = (ix - arx) / rw
                            rel_pos[re_index, 1] = (iy - ary) / rh
                            rel_pos[re_index, 2] = (iw*ih)/(rw*rh)
                            rel_pos[re_index, 3] = 1. / rw
                            rel_pos[re_index, 4] = 1. / rh

                    elif self.rel_pos == 3:
                        assert not self.pos_norm
                        rel_pos[re_index, 0] = (aix - arx) / rw
                        rel_pos[re_index, 1] = (aiy - ary) / rh
                        rel_pos[re_index, 2] = (iw*ih)/(rw*rh)

                    else:
                        raise

                    re_index += 1

                assert re_index == top_k
            
                # rel_pos = cat([torch.tensor(x) for x in rel_pos])  # [25, ]
                rel_pos = cat([x for x in rel_pos])

                rel_pos_one_img.append(rel_pos)


            abs_pos_per_img.append(abs_pos_one_img)
            
            
            rel_pos_one_img = cat([x.unsqueeze(0) for x in rel_pos_one_img])
            rel_pos_per_img.append(rel_pos_one_img)
            
        abs_pos_per_img = cat([torch.tensor(abs_pos_one_img) for abs_pos_one_img in abs_pos_per_img])
        rel_pos_per_img = cat([rel_pos_one_img for rel_pos_one_img in rel_pos_per_img])


        return abs_pos_per_img, rel_pos_per_img


    def fetch_obj_proposals_concurrent_mattnet(self, proposals):
        rel_pos_per_img = []
        # rel_pos_ids_per_img = []

        top_k = self.topk
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach()  # [B, 4]
            all_boxes = boxes

            box_ids = list(range(boxes.shape[0]))
    
            cboxes = torch.zeros((len(box_ids), 2))
            cboxes[:, 0] = (boxes[:, 0]+boxes[:, 2])/2
            cboxes[:, 1] = (boxes[:, 1]+boxes[:, 3])/2
            cboxes1 = cboxes.repeat_interleave(repeats=boxes.shape[0], dim=0)
            cboxes2 = cboxes.repeat(boxes.shape[0], 1)
            
            cboxes3 = cboxes1 - cboxes2
            cboxes3 = cboxes3.reshape(boxes.shape[0], boxes.shape[0], -1)

            cboxes_ids = torch.zeros((boxes.shape[0], boxes.shape[0]), dtype=torch.int)
            for i in range(boxes.shape[0]):
                cboxes_ids[i] = torch.tensor(list(range(boxes.shape[0])))

            ncboxes_ids = []
        #     print(cboxes_ids)
            for i in range(boxes.shape[0]):
                cboxes_id = cboxes_ids[i].tolist()
                
                def compare(box_id0, box_id1):

                #         print(box_id0, box_id1)
                        delta1 = cboxes3[i][box_id0].tolist()
                        delta2 = cboxes3[i][box_id1].tolist()

        
                        if (delta1[0]**2 + delta1[1]**2) <= (delta2[0]**2 + delta2[1]**2):
                #             print("sec1")
                            return -1
                        else:
                #             print("sec2")
                            return 1
    
                ncboxes_id = sorted(cboxes_id, key=cmp_to_key(compare))
                ncboxes_ids.append(ncboxes_id)

            ncboxes_ids = torch.tensor(ncboxes_ids) # [N, N]
            # print("排序后id————————")
            # print(ncboxes_ids)  

            rel_infos = []
            
            
            all_boxes = boxes
            # print("所有boxes：")
            # print(all_boxes.shape)
            nall_boxes = all_boxes.repeat_interleave(repeats=all_boxes.shape[0], dim=0)
            all_centerX = 0.5*(nall_boxes[:, 2] + nall_boxes[:, 0])
            all_centerY = 0.5*(nall_boxes[:, 3] + nall_boxes[:, 1])

            all_boxes2 = boxes
            nsub_boxes = all_boxes2.repeat(all_boxes.shape[0], 1)

            assert nall_boxes.shape == nsub_boxes.shape

            temp = torch.zeros((nall_boxes.shape[0], 5), dtype=torch.float32)
            temp[:, 0] = (nsub_boxes[:, 0] - all_centerX) / (nall_boxes[:, 2] - nall_boxes[:, 0])
            temp[:, 1] = (nsub_boxes[:, 1] - all_centerY) / (nall_boxes[:, 3] - nall_boxes[:, 1])
            temp[:, 2] = (nsub_boxes[:, 2] - all_centerX) / (nall_boxes[:, 2] - nall_boxes[:, 0])
            temp[:, 3] = (nsub_boxes[:, 3] - all_centerY) / (nall_boxes[:, 3] - nall_boxes[:, 1])
            temp[:, 4] = ((nsub_boxes[:, 2] - nsub_boxes[:, 0]) * (nsub_boxes[:, 3] - nsub_boxes[:, 1])) / ((nall_boxes[:,2] - nall_boxes[:, 0]) * (nall_boxes[:, 3] - nall_boxes[:, 1]))
    
            temp = temp.reshape(boxes.shape[0], boxes.shape[0], -1)  #[N, N, 5]
            # print(temp.shape)
            
            # 根据ncboxes_ids选
            # rel_pos_per_img = []
            for jk in range(ncboxes_ids.shape[0]):
                cboxes_id = ncboxes_ids[jk]
                
                # 遍历可能能作为参照信息的box
                count = 0
                rel_info = torch.zeros((top_k, 5), dtype=torch.float)
                for jd in range(cboxes_id.shape[0]):
                    cid = cboxes_id[jd]
                    
                    if cid == jk: # 自身
                        continue
                        
        #             print(count, jk, cid)
                    rel_info[count] = temp[jk, cid]  # 
                    count += 1
                    
                    if count >= top_k:
                        break
            
                # print(rel_info.shape) # [top_k, 5]
                # assert 1 == 0
                rel_info = rel_info.reshape(1, -1)  # 展平
                rel_pos_per_img.append(rel_info)

        rel_pos_per_img = cat([rel_pos_one_img for rel_pos_one_img in rel_pos_per_img])

        # print(rel_pos_per_img.shape)
        # assert 1 == 0
        return rel_pos_per_img

    def fetch_obj_proposals_concurrent(self, proposals):
        rel_pos_per_img = []
        rel_pos_ids_per_img = []

        top_k = self.topk
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach()  # [B, 4]
            all_boxes = boxes
            # print("所有boxes：")
            # print(all_boxes.shape)
            nall_boxes = all_boxes.repeat_interleave(repeats=all_boxes.shape[0], dim=0)
            all_centerX = 0.5*(nall_boxes[:, 2] + nall_boxes[:, 0])
            all_centerY = 0.5*(nall_boxes[:, 3] + nall_boxes[:, 1])
            
            all_boxes2 = boxes
            nsub_boxes = all_boxes2.repeat(all_boxes.shape[0], 1)
            
            assert nall_boxes.shape == nsub_boxes.shape
            
            temp = torch.zeros((nall_boxes.shape[0], 5), dtype=torch.float32)
            temp[:, 0] = (nsub_boxes[:, 0] - all_centerX) / (nall_boxes[:, 2] - nall_boxes[:, 0])
            temp[:, 1] = (nsub_boxes[:, 1] - all_centerY) / (nall_boxes[:, 3] - nall_boxes[:, 1])
            temp[:, 2] = (nsub_boxes[:, 2] - all_centerX) / (nall_boxes[:, 2] - nall_boxes[:, 0])
            temp[:, 3] = (nsub_boxes[:, 3] - all_centerY) / (nall_boxes[:, 3] - nall_boxes[:, 1])
            temp[:, 4] = ((nsub_boxes[:, 2] - nsub_boxes[:, 0]) * (nsub_boxes[:, 3] - nsub_boxes[:, 1])) / ((nall_boxes[:,2] - nall_boxes[:, 0]) * (nall_boxes[:, 3] - nall_boxes[:, 1]))
            
            mask = np.identity(all_boxes.shape[0])  # [N, N]
            mask[top_k:, top_k:] = 0
            mask = 1 - mask
            mask[:, (top_k+1):] = 0
            mask[top_k:, top_k] = 0
            
            mask = mask.astype(np.bool)
            temp = temp.reshape(all_boxes.shape[0], all_boxes.shape[0], -1)
            
            rel_info = temp[mask, :]
            # print(rel_info.shape) # [N*top_k, 5]
            rel_info = rel_info.reshape(all_boxes.shape[0], -1)
            
            rel_pos_per_img.append(rel_info)
        
        rel_pos_per_img = cat([rel_pos_one_img for rel_pos_one_img in rel_pos_per_img])
        # print(rel_pos_per_img.shape) # [N*N, top_k*5]
        # assert 1 == 0
        return rel_pos_per_img

    def compute_abs_pos_old(self, proposals):
        abs_pos_per_img = []

        # rel_pos_per_img = []

        top_k = self.topk


        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach().cpu()
            ih, iw = proposals_one_img.image_size
            abs_pos_one_img = np.empty((len(boxes), self.rel_pos), dtype=np.float32)

            for ix, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2-x1, y2-y1
                cx, cy = x + 0.5 * w, y + 0.5 * h
                
                abs_pos = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)], np.float32)

                abs_pos_one_img[ix] = abs_pos

            abs_pos_per_img.append(abs_pos_one_img)
        
        abs_pos_per_img = cat([torch.tensor(abs_pos_one_img) for abs_pos_one_img in abs_pos_per_img])

        return abs_pos_per_img

    def compute_abs_pos(self, proposals):
        abs_pos_per_img = []

        # rel_pos_per_img = []

        top_k = self.topk


        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach().cpu()
            ih, iw = proposals_one_img.image_size
            abs_pos_one_img = np.zeros((len(boxes), self.rel_pos), dtype=np.float32)
            for ix, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2-x1, y2-y1
                cx, cy = x + 0.5 * w, y + 0.5 * h

                if self.rel_pos == 7:
                    assert self.pos_norm
                    abs_pos = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih), 1./iw, 1./ih], np.float32)

                elif self.rel_pos == 5:
                    if not self.pos_norm :
                        abs_pos = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)], np.float32)
                    else:    
                        abs_pos = np.array([cx/iw, cy/ih, w*h/(iw*ih), 1.0/iw, 1.0/ih], np.float32)  #

                # abs_pos = np.array([cx/iw, cy/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)], np.float32)
                elif self.rel_pos == 3:
                    assert not self.pos_norm
                    abs_pos = np.array([cx/iw, cy/ih, w*h/(iw*ih)], np.float32)  

                else:
                    raise

                abs_pos_one_img[ix] = abs_pos
            
            abs_pos_per_img.append(abs_pos_one_img)
        
        abs_pos_per_img = cat([torch.tensor(abs_pos_one_img) for abs_pos_one_img in abs_pos_per_img])
        return abs_pos_per_img

    def compute_abs_pos_concurrent(self, proposals):
        abs_pos_per_img = []

        # rel_pos_per_img = []

        assert self.rel_pos == 5
        
        for index, proposals_one_img in enumerate(proposals):
            boxes = proposals_one_img.proposal_boxes.tensor.data.detach()
            
            
            ih, iw = proposals_one_img.image_size
            # im_array = np.array([iw, ih, iw, ih, iw*ih])
            im_array = torch.tensor([iw, ih, iw, ih, iw*ih], dtype=torch.float32)
            
            # abs_pos_one_img = np.zeros((len(boxes), self.rel_pos), dtype=np.float32)
            abs_pos_one_img = torch.zeros((len(boxes), self.rel_pos), dtype=torch.float32)
            
            abs_pos_one_img[:, 0] = boxes[:, 0]
            abs_pos_one_img[:, 1] = boxes[:, 1]
            abs_pos_one_img[:, 2] = boxes[:, 2] - 1
            abs_pos_one_img[:, 3] = boxes[:, 3] - 1
            abs_pos_one_img[:, 4] = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
            
            abs_info = abs_pos_one_img / im_array
            
            abs_pos_per_img.append(abs_info)
            
        abs_pos_per_img = cat([abs_pos_one_img for abs_pos_one_img in abs_pos_per_img])
        
        return abs_pos_per_img
    
 


class GATNet(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.1, alpha=0.2, nheads=2):
        super().__init__()
        print("使用GAT")
        
        self.dropout = dropout

        self.attentions = [GATNetLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        self.out_att = GATNetLayer(nhid * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj=None):
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, adj))
        
        # return F.log_softmax(x, dim=1)
        # print(x.shape)
        # assert 1 == 0
        
        return x
       
class GATNetLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        print("使用GAT")
        # assert in_features == out_features, str(in_features) + ", " + str(out_features) # 暂时不改变维度
        
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.FloatTensor(size=(in_features, out_features)))  # 线性变换
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.FloatTensor(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    
    def forward_new(self, node_features, adj=None):
        # print("new gat")
        if adj is None:
            return self.forward_old(node_features)
        # 先投影变换
        Wh = torch.matmul(node_features, self.W) # [B, N, iD] * [iD, oD] -> [B, N, oD]
        attention = adj
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    # forward_batch
    def forward(self, node_features, adj=None):
        
        Wh = torch.matmul(node_features, self.W) # [B, N, iD] * [iD, oD] -> [B, N, oD]
        
        a_input = self._batch_prepare_attentional_mechanism_input(Wh)  # [B, N, N, 2*oD]
        
        e = torch.matmul(a_input, self.a) # [B, N, N, 1]
        
        e = self.leakyrelu(e.squeeze(-1))  # [B, N, N]
        
        # print(node_features.shape, e.shape)
        # if adj is not None:
            # print(adj.shape)
        
        ########################################
        # 需要修改
        # 貌似这个地方有问题
        # 在构造adj的时候我们就已经执行过softmax操作，并且添加了我们自己设计的weight产生公式
        # 所以在这里直接使用即可
        if adj is not None:
            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e
        
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        ########################################
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _batch_prepare_attentional_mechanism_input(self, Wh):
        B, N, D = Wh.shape
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)  # [B, N*N, D]
        
        Wh_repeated_alternating = Wh.repeat(1, N, 1)  # [B, N*N, D]
        
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1) # [B, N*N, 2D]
        
        return all_combinations_matrix.contiguous().view(B, N, N, 2*D)
                
    
    def forward_single(self, node_features, adj=None):
        Wh = torch.mm(node_features, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        print(node_features.shape, e.shape)
        if adj is not None:
            print(adj.shape)
            
        if adj is not None:
            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            
        else:
            # attention = -9e15*torch.ones_like(e)  # 初始化attention
            attention = e
        
        attention = F.softmax(attention, dim=1)  # 公式（3）
        attention = F.dropout(attention, self.dropout, training=self.training)  # [N, N]
        
        node_features_t = torch.matmul(attention, Wh)  # [N, out_features]
        
        # assert node_features.shape[1] == node_features_t.shape[1]
        
        if self.concat:
            return F.elu(node_features_t)
        else:
            return node_features_t

    def _prepare_attentional_mechanism_input(self, Wh):  # 公式（1）
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# from  PGCN
from torch.nn.parameter import Parameter
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        
        # print(input.shape)
        # print(self.weight.shape)
        # 注意torch.matmul、torch bmm，以及torch.mm之间的差别
        # torch.mm的输入一定是二维的，而torch.bmm的输入可以是三维的，但是两个输入一定都同维度
        # torch.matmul输入的维度可以不一（似乎），但是需要满足矩阵乘法的条件
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 一般来说，GCN只需要两层即可
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.8): # 这里的dropout可能需要设置较高（e.g., 0.8）
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x



class GuildNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_dim = 256

        self.guild = nn.Sequential(
            nn.Conv2d(self.input_dim, self.input_dim, 1, bias=False),
            nn.BatchNorm2d(self.input_dim),
            nn.ReLU(),
            nn.Conv2d(self.input_dim, self.input_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_dim),
            nn.ReLU(),
            nn.Conv2d(self.input_dim, self.input_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.input_dim),
            nn.ReLU(),
            nn.Conv2d(self.input_dim, self.input_dim, 1, bias=False),
            nn.BatchNorm2d(self.input_dim),
            nn.ReLU(),
        )

    
    def forward(self, x):
        x = self.guild(x)

        return x


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]