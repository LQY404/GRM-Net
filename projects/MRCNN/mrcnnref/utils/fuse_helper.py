import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math

def _make_conv(input_dim, output_dim, k, stride=1):
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, (k, k), padding=(pad, pad), stride=(stride, stride)),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True)
    )
    
def _make_mlp(input_dim, output_dim, drop):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True),
                         nn.Dropout(drop),
                         nn.Linear(output_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True))

def _make_coord(batch, height, width):
    # relative position encoding
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0), \
                                               xv_max.unsqueeze(0), yv_max.unsqueeze(0), \
                                               xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0), \
                                               hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord
  
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
  
