import torch
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.utils.comm import get_world_size


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total.float() / num_gpus


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, stride, device): 
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2  # 注意+stride//2的操作, 相当于将每个点的位置移到stride中心
    # 返回的尺寸大小为[h*w, 2]
    return locations


def compute_ious(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    return ious, gious

def compute_overlap_xyxy(box1, box2):
    '''
    box: x, y, x, y
    '''
    # 计算两个Box的左上角、右下角
    ulx1, uly1, brx1, bry1 = box1[0], box1[1], box1[2], box1[3]
    ulx2, uly2, brx2, bry2 = box2[0], box2[1], box2[2], box2[3]

    x1 = max(ulx1, ulx2)
    y1 = max(uly1, uly2)
    x2 = min(brx1, brx2)
    y2 = min(bry1, bry2)

    if x1 > x2 or y1 > y2:
        return 0
    return (x2 - x1) * (y2 - y1)

def compute_iou_xyxy(box1, box2):
    overlap_area = compute_overlap_xyxy(box1, box2)
    all_area = (box1[2]-box1[0]) * (box1[3]-box1[1]) + (box2[2]-box2[0]) * (box2[3]-box2[1]) - overlap_area

    return overlap_area / all_area



##########
#  for sentence
##########

#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}


def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
  """
  Tokenize a sequence, converting a string s into a list of (string) tokens by
  splitting on the specified delimiter. Optionally keep or remove certain
  punctuation marks and add start and end tokens.
  """
  if punct_to_keep is not None:
    for p in punct_to_keep:
      s = s.replace(p, '%s%s' % (delim, p))

  if punct_to_remove is not None:
    for p in punct_to_remove:
      s = s.replace(p, '')

  tokens = s.split(delim)
  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')
  return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
  token_to_count = {}
  tokenize_kwargs = {
    'delim': delim,
    'punct_to_keep': punct_to_keep,
    'punct_to_remove': punct_to_remove,
    'add_start_token': False,
    'add_end_token': False,
  }
  for seq in sequences:
#    if delim==';':
#      print('seq={}'.format(seq))
    seq_tokens = tokenize(seq, **tokenize_kwargs)
#    if delim==';':
#      print('seq_tokens={}'.format(seq_tokens))
#      _=input()
    for token in seq_tokens:
      token_to_count[token] = token_to_count.get(token, 0) + 1
      # if token not in token_to_count:
      #   token_to_count[token] = 0
      # token_to_count[token] += 1

  token_to_idx = {}
  for token, idx in SPECIAL_TOKENS.items():  #添加开始、结束符
    token_to_idx[token] = idx
    
  for token, count in sorted(token_to_count.items()):
    if count >= min_token_count:
      token_to_idx[token] = len(token_to_idx)

  return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
  seq_idx = []
  for token in seq_tokens:
    if token not in token_to_idx.keys():
      if allow_unk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
  # print(idx_to_token)
  tokens = []
  for idx in seq_idx:
    # print(idx)
    if type(idx) != int:
      #idx = int(idx.data.cpu().numpy())
      idx = int(idx)
      # print(idx)
    tokens.append(idx_to_token[idx])
    if stop_at_end and tokens[-1] == '<END>':
      break
  if delim is None:
    return tokens
  else:
    return delim.join(tokens)


##############
##  for ref
#############
import json

def invert_dict(d):
  return {v: k for k, v in d.items()}


def load_vocab(path):
  with open(path, 'r') as f:
    vocab = json.load(f)
    vocab['refexp_idx_to_token'] = invert_dict(vocab['refexp_token_to_idx'])
    
  # Sanity check: make sure <NULL>, <START>, and <END> are consistent
  assert vocab['refexp_token_to_idx']['<NULL>'] == 0
  assert vocab['refexp_token_to_idx']['<START>'] == 1
  assert vocab['refexp_token_to_idx']['<END>'] == 2
  
  return vocab

def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 8), dtype=np.float32)

    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val
    
def generate_coord(batch, height, width, device):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).to(device))
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

def compute_locations_ref(h, w, stride): 
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2  # 注意+stride//2的操作, 相当于将每个点的位置移到stride中心
    # 返回的尺寸大小为[h*w, 2]
    return locations

def g2(N, vh, vw, stride):
    spatial_batch_val = np.zeros((N, vh, vw, 8), dtype=np.float32)
    
    H, W = vh * stride, vw * stride
    
    locs = compute_locations(vh, vw, stride).reshape(vh, vw, -1)  # [vh, vw, 2]
    
    for h in range(vh):
        for w in range(vw):
            
            xmin = locs[h, w, 0] / H * 2 - 1
            xmax = (locs[h, w, 0] + stride // 2) / H * 2 - 1
            
            xctr = (xmin + xmax) / 2
            
            ymin = locs[h, w, 1] / H * 2 - 1
            ymax = (locs[h, w, 1] + stride // 2) / H * 2 - 1
            
            yctr = (ymin + ymax) / 2
            
            spatial_batch_val[: h, w, :] = [xmin, ymin, xmax, ymax, xctr, yctr, 1/H, 1/W]
        
    
    return spatial_batch_val
###################
########  visualization
###################
import os
import numpy as np
# from skimage.measure import find_contours
import matplotlib.pyplot as plt
from PIL import Image
import imgviz

if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')

def apply_mask(image, mask, color=None, alpha=0.7, show=False):
    """Apply the given mask to the image.
    """
    maskv = np.unique(mask)
    # print(maskv)
    # if show:
    #     print("categories in this image:" + str(maskv))
    if color is None:
        color = np.array([220, 20, 60])
        color = color.astype(np.int32)

    for c in range(3):
         for v in maskv:
            if v == 0:
                continue
            # print(image[:, :, c] * (1 - alpha).shape)
            # print(alpha * color[c])
            image[:, :, c] = np.where(mask == v,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def apply_center(image, center, alpha=0.1): #alpha越大越实
    # center[center != 1] = 0
    # center = center.astype(np.uint8)
    index = np.where(center > 0.1)
    index = tuple(zip(index[0], index[1]))
    for x, y in index: 
        # for c in range(3): 
            # image[x, y, c] = image[x, y, c] * (1 - center[x, y]) + center[x, y] * 255
        image[x, y, :] = [0,255,0] 

    return image

def display_mask(mask):
    plt.imshow(mask)
    plt.show()

def display_instances(image, masks, center=None, colors=None, title="",
                      figsize=(10, 10), ax=None, show=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    
    

    # Generate random colors
    

    # Show area outside image boundaries.

    masked_image = image.astype(np.uint32).copy()
    # print(masked_image.shape)
    print(len(masks))
    for i in range(len(masks)):
        if colors:
            color = colors[i]['color']
            cate_name = colors[i]['name']
        else:
            color = None
            cate_name = None
        # Mask
        mask = masks[i]
        # print(mask.shape)
        
        # center_y, center_x = np.mean(np.where(mask!=0)[0]), np.mean(np.where(mask!=0)[1])
        # ax.text(center_x, center_y, str(cate_name), color='red', size=8, backgroundcolor="none")
        masked_image = apply_mask(masked_image, mask, color, show=show)

   
    if center is not None:
        apply_center(masked_image, center[0])

    if show:
        if not ax:
            _, ax = plt.subplots(1)
        height, width = image.shape[:2]
    
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)
        
        ax.imshow(masked_image.astype(np.uint32))
        plt.show()

    return masked_image.astype(np.uint32)


def saveMask(mask, save_dir):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())

    lbl_pil.save(save_dir)
    
    

import torch.nn as nn
class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output



