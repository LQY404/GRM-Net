## Multi-instance Referring Image Segmentation of Scene Sketches based on Global Reference Mechanism
this is the source code of our work [GRM-Net](https://diglib.eg.org/handle/10.2312/pg20221238). In this work, we solve the problem of Multi-instance referring segmentation in Sketch Scenes.

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install and build libs
```
git clone https://github.com/PeizeSun/SparseR-CNN.git
cd SparseR-CNN
python setup.py build develop
```

2. Train SparseR-CNN
```
python projects/MRCNN/train_net.py --num-gpus 1 \
    --config-file projects/MRCNN/configs/maskrcnn_r_101_3x.yaml
```

3. Evaluate SparseR-CNN
```
python projects/MRCNN/train_net.py --num-gpus 1 \
    --config-file projects/MRCNN/configs/maskrcnn_r_101_3x.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```



## Citing

If you use GRM-Net in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@inproceedings {10.2312:pg.20221238,
booktitle = {Pacific Graphics Short Papers, Posters, and Work-in-Progress Papers},
editor = {Yang, Yin and Parakkat, Amal D. and Deng, Bailin and Noh, Seung-Tak},
title = {{Multi-instance Referring Image Segmentation of Scene Sketches based on Global Reference Mechanism}},
author = {Ling, Peng and Mo, Haoran and Gao, Chengying},
year = {2022},
publisher = {The Eurographics Association},
ISBN = {978-3-03868-190-8},
DOI = {10.2312/pg.20221238}
}
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}

```
