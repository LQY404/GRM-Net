# Copyright (c) Facebook, Inc. and its affiliates.
import math
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn
import torch

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone
from .build import BACKBONE_REGISTRY
from .resnet import build_resnet_backbone
from .utils.comm import generate_coord

__all__ = ["build_resnet_fpn_backbone", "build_retinanet_resnet_fpn_backbone", "FPN"]


class FPNRef(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum", cfg=None
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPNRef, self).__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        # _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        print(strides)
        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            
            if strides[-2] == strides[-1] and len(in_channels_per_feature)-idx == 1:
                stage = int(math.log2(32))
            else:    
                stage = int(math.log2(strides[idx]))
            
            # print(stage)
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        # print(output_convs)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        # print(strides)
        if strides[-2] == strides[-1]:  # res5 dilation:
            print("res5使用空洞代替下采样")
            self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides[: -1]}  # p2-p4
            assert "p5" not in self._out_feature_strides
            print(self._out_feature_strides)
            self._out_feature_strides['p5'] = strides[-1]
            
        else:
            self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        # print(self._out_feature_strides)
        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

        # Scripting does not support this: https://github.com/pytorch/pytorch/issues/47334
        # have to do it in __init__ instead.
        self.rev_in_features = tuple(in_features[::-1])
        
        
        
        self.use_mm_fpn = True
        if self.use_mm_fpn:
            self.roi_dim = 256
            self.rnn_dim = cfg.REF_RNN_DIM
            bidirectional = 2
            self.hn_dim = self.rnn_dim * bidirectional if not self.use_bert else 512
            self.m_dim = self.roi_dim * 2 + 8
            
            self.mback = nn.Sequential(
                # nn.Conv2d(self.m_dim, self.roi_dim, 1),
                # CoordConv(self.m_dim, self.m_dim, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(self.m_dim, self.roi_dim, 3, bias=False, padding=1),
                nn.BatchNorm2d(self.roi_dim),
                # nn.SyncBatchNorm(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(),
                nn.Conv2d(self.roi_dim, self.roi_dim, 3, bias=False, padding=1),
                nn.BatchNorm2d(self.roi_dim),
                # # nn.SyncBatchNorm(self.roi_dim),
                # # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(),
            )

            self.lfc = nn.Sequential(
                nn.Linear(self.hn_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                # nn.LayerNorm(self.roi_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.roi_dim, self.roi_dim),
                # nn.BatchNorm1d(self.roi_dim),
                # nn.GroupNorm(32, self.roi_dim),
                nn.ReLU(),
            )

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x, sent_dict=None):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # print(x)
        if self.use_mm_fpn:
            assert sent_dict is not None
            hs, hn, embedding, words = sent_dict["hs"], sent_dict["hn"], sent_dict["embedding"], sent_dict["words"]
            tcontext = self.lfc(hn)
            
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        
        # change for mmfusion
        if self.use_mm_fpn:
            prev_features_ = self.output_convs[0](prev_features)
            bs, c, vh, vw = prev_features_.shape
            coord = generate_coord(bs, vh, vw, prev_features_.device)
            hn = tcontext.reshape(bs, -1, 1, 1).repeat(1, 1, vh, vw)
            
            mfea = torch.cat((hn, prev_features_, coord), dim=1)
            mfea = self.mback(mfea)
            mfea = F.normalize(mfea, p=2, dim=1)
            results.append(mfea)
            
        else:
            results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for features, lateral_conv, output_conv in zip(
            self.rev_in_features[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            features = bottom_up_features[features]
            if prev_features.shape[-1] == features.shape[-1] and prev_features.shape[-2] == features.shape[-2]:
                # 此时res5使用空洞
                top_down_features = prev_features
            else:
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                
            # Has to use explicit forward due to https://github.com/pytorch/pytorch/issues/47336
            lateral_features = lateral_conv.forward(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv.forward(prev_features))


        # for P6
        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(list(zip(self._out_features, results)))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPNRef(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        cfg=None
    )
    return backbone
