# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    print(meta_arch)
    print(META_ARCH_REGISTRY)
    print(cfg)
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    # model = torch.nn.parallel.DistributedDataParallel(
    #     module=model.to(cfg.MODEL.DEVICE), broadcast_buffers=False,
    #     device_ids=[0], output_device=0, find_unused_parameters=True)

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
