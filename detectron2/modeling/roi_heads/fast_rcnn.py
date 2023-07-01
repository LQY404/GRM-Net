# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple, Union
import torch
import math
from fvcore.nn import giou_loss, smooth_l1_loss
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from torch.nn import functional as F

from typing import Dict, List, Optional, Tuple
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""

category_weights = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    math.log(math.e + 30094./605), 
    1.0, 
    math.log(math.e + 30094./1725),
    1.0, 1.0, 
    math.log(math.e + 30094./716), math.log(math.e + 30094./343),  # a-b
    math.log(math.e + 30094./1205), math.log(math.e + 30094./685),
    1.0, 1.0, 
    math.log(math.e + 30094./836), math.log(math.e + 30094./1886), math.log(math.e + 30094./580), 
    1.0, 1.0, 
    math.log(math.e + 30094./1254), math.log(math.e + 30094./958),   # c-d
    1.0, 1.0, 1.0, 
    math.log(math.e + 30094./1941), math.log(math.e + 30094./184), math.log(math.e + 30094./2209), math.log(math.e + 30094./56), 
    1.0, 
    math.log(math.e + 30094./2655), 
    1.0, 
    math.log(math.e + 30094./188), math.log(math.e + 30094./647), math.log(math.e + 30094./744),  # f-r
    math.log(math.e + 30094./810), 
    1.0, 
    math.log(math.e + 30094./62), 
    1.0, 
    math.log(math.e + 30094./1068), 
    1.0, 
    math.log(math.e + 30094./8504), math.log(math.e + 30094./233),
    1.0, 1.0
]


def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    matching_scores: Optional[List[torch.tensor]] = None,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    if matching_scores is None:
        result_per_image = [
            fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
            )
            for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
        ]
    else:
        result_per_image = [
            fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, matching_score
            )
            for scores_per_image, boxes_per_image, image_shape,matching_score in zip(scores, boxes, image_shapes, matching_scores)
        ]

    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    matching_score = None,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    if matching_score is None:
        filter_mask = scores > score_thresh  # R x K
    else:
        filter_mask = (scores > score_thresh) & (matching_score > 0.5)
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    if matching_score is not None:
        matching_score = matching_score[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    if matching_score is not None:
        matching_score = matching_score[keep]
    
    keep = batched_nms(boxes, scores, torch.Tensor([0]), 0.9)  # 不分类NMS
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    if matching_score is not None:
        matching_score = matching_score[keep]
    

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    if matching_score is not None:
        result.pred_matchs = matching_score

    return result, filter_inds[:, 0]


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        matching_score=None,
        feature_matching_score=None,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.matching_score = matching_score
        if self.matching_score is not None:
            self.matching_score = self.matching_score.reshape(-1, )
        # assert self.matching_score is not None
        # print(self.matching_score.shape)

        self.feature_matching_score = feature_matching_score
        if self.feature_matching_score is not None:
            self.feature_matching_score = self.feature_matching_score.reshape(-1, )

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            
            if proposals[0].has("matching_gt"):
                self.matching_gt = cat([p.matching_gt for p in proposals], dim=0)
                # print(self.matching_gt.shape)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.data.detach().sigmoid().argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # bg_class_ind = 0

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        # fg_inds = self.gt_classes > 0
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        bg_inds = self.gt_classes == bg_class_ind
        num_bg = bg_inds.nonzero().numel()
        bg_ = pred_classes[~fg_inds]
        num_false_positive = (bg_ != bg_class_ind).nonzero().numel()

        num_true_negative = (bg_ == bg_class_ind).nonzero().numel()
        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()

        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()


        # print("fg:", num_fg)
        # print("bg:", num_bg)
        # print("num_instances:", num_instances)
        # print("num_accurate:", num_accurate)
        # print("num_true_positive:", num_true_positive)
        # print("num_false_positive:", num_false_positive)
        # print("num_true_negative:", num_true_negative)
        # print("num_false_negative:", num_false_negative)
        # assert 1 == 0
        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)  # 这个指标可以不要
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)
            
            if num_bg > 0:
                storage.put_scalar("fast_rcnn/false_positive", num_false_positive / num_bg)
                storage.put_scalar("fast_rcnn/bg_cls_accuracy", num_true_negative / num_bg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            print("训练过程中无预测实例。。。")
            return 0.0 * self.pred_class_logits.sum()
        else:
            # self._log_accuracy()
            bg_class_ind = self.pred_class_logits.shape[1] - 1
            fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)

            # add weight only for sketch
            # return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")
            #
            # return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="sum") / max(1, len(fg_inds))
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, weight=torch.tensor(category_weights).to(self.gt_classes.device), reduction="sum") / max(1, len(fg_inds))
            # print(self.pred_class_logits.shape, self.gt_classes.shape)

            gt_matrix = torch.zeros_like(self.pred_class_logits)

            idxs = [(i, v) for i, v in enumerate(self.gt_classes.data.cpu().numpy())]

            for idx in idxs:
                gt_matrix[idx] = 1.0

            # print(self.gt_classes)
            # print(gt_matrix) 
            # assert 1 == 0
            # return sigmoid_focal_loss_jit(self.pred_class_logits, gt_matrix, alpha=0.25, gamma=2.0, reduction="mean")
            return sigmoid_focal_loss_jit(self.pred_class_logits, gt_matrix, alpha=0.25, gamma=2.0, reduction="sum") / max(1, len(fg_inds))  #使用Focal loss

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # bg_class_ind = 0

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        # print(fg_inds)
        # fg_inds = nonzero_tuple(self.gt_classes > 0)[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
            
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg


    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        l = {
            "loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()
        }

        assert self.feature_matching_score is None
            
        assert self.matching_score is  None
            
        return l

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)
    
    def predict_matches(self):
        assert self.matching_score is not None
        matching_scores = self.matching_score.sigmoid()

        return matching_scores.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        if self.matching_score is not None:
            matching_scores = self.predict_matches()
        else:
            matching_scores = None

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, matching_scores
        )


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = True, # 改一下
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        # add matching head
        self.use_match = 0
        if self.use_match:
            self.matching_pred = Linear(input_size, 1)
        else:
            self.matching_pred = None

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        if self.use_match:
            nn.init.normal_(self.matching_pred.weight, std=0.01)
            nn.init.constant_(self.matching_predm, 0)

        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        # print("训练模式？", self.training)
        # if self.training:
        #     x = F.dropout(x, p=0.5)  # 削弱过拟合
        # else:
        #     print("dont need dropout for test")

        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        # proposal_deltas = proposal_deltas_match[:, :-1]
        # matching_score = proposal_deltas_match[:, -1].unsqueeze(-1)
        if self.matching_pred is not None:
            matching_scores = self.matching_pred(x)

            return scores, proposal_deltas, matching_scores
        # print(proposal_deltas.shape)
        # print(matching_score.shape)

        # assert 1 == 0
        return scores, proposal_deltas

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals, feature_matching_score=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        if self.matching_pred is not None:
            scores, proposal_deltas, matching_scores = predictions
        else:
            scores, proposal_deltas = predictions
            matching_scores = None
            
        losses = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            matching_score=matching_scores,
            feature_matching_score=feature_matching_score
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """

        # assert len(predictions) == 3
        # print("筛选")
        # scores, proposal_deltas, matching_score = predictions
        # print(scores.shape, matching_score.shape)
        # keep = matching_score.squeeze(-1).sigmoid() >= 0.5
        # print(matching_score.shape)
        # scores = scores[keep, :]
        # proposal_deltas = proposal_deltas[keep, :]
        # matching_score = matching_score[keep, :]
        # predictions = scores, proposal_deltas, matching_score

        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)

        if self.matching_pred is not None:
            matching_scores = self.predict_matches(predictions, proposals)
        else:
            matching_scores = None

        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            matching_scores=matching_scores
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        
        if self.matching_pred is not None:
            _, proposal_deltas, _ = predictions
        else:
            _, proposal_deltas = predictions

        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        print(predict_boxes.shape, N, B)
        assert 1 == 0
        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []

        if self.matching_pred is not None:
            _, proposal_deltas, _ = predictions
        else:
            _, proposal_deltas = predictions

        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            # ensure fp32 for decoding precision
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        if self.matching_pred is not None:
            scores, _, _ = predictions
        else:
            scores, _ = predictions

        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_matches(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):  
        assert self.matching_pred is not None
        _, _, matching_scores = predictions

        num_inst_per_image = [len(p) for p in proposals]
        matches = matching_scores.sigmoid()
        
        return matches.split(num_inst_per_image, dim=0)
