# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.layers import nonzero_tuple

__all__ = ["subsample_labels"]



def subsample_pos_labels(positive, need_pos):
    perm1 = torch.randperm(len(positive), dtype=torch.int64)[:need_pos]
    # print("正样本序号：", perm1)
    perm1 = perm1.to(positive.device)

    pos_idx = positive[perm1]
    return pos_idx

def subsample_neg_labels(negative, need_neg):
    perm2 = torch.randperm(len(negative), dtype=torch.int64)[:need_neg]
    # print("负样本序号：", perm2)
    perm2 = perm2.to(negative.device)
    neg_idx = negative[perm2]
    return neg_idx

def subsample_labels_old(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.
    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.
    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx

def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    # print(labels.shape, labels)
    # labels是所有的anchor数目，总量很大


    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)  # 最多256 * 0.5
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos  # 256 - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)


    # if num_neg / num_pos > 3:  # 保持正负样本比例不会太悬殊，
    #     num_neg = min(num_pos * 3, negative.numel())  # 一般来说negative.numel()都很大

    # print("正负样本比例：", num_neg / num_pos)
    # print("正样本数目：", positive.numel(), len(positive))
    # print("负样本数目：", negative.numel(), len(negative))
    # print(num_pos, num_neg)
    # randomly select positive and negative examples
    # perm1 = torch.randperm(len(positive), device=positive.device, dtype=torch.int64)[:num_pos]
    # print("采样得到的正样本序号：", perm1)
    # perm2 = torch.randperm(len(negative), device=negative.device, dtype=torch.int64)[:num_neg]
    # print("采样得到的负样本序号：", perm2)

    # num = -1
    # while torch.sum(perm2) == 0 or not torch.all(perm2 < num_neg):
    #     print("重新生成")
    #     if num == -1:
    #         num = len(negative)

    #     num = max(num //2, num_neg)
        
    #     perm2 = torch.randperm(num, device=negative.device, dtype=torch.int64)[:num_neg]

    # print(perm2)   

    # pos_idx = positive[perm1]
    # neg_idx = negative[perm2]
    pos_idx = subsample_pos_labels(positive, num_pos)
    neg_idx = subsample_neg_labels(negative, num_neg)

    return pos_idx, neg_idx
