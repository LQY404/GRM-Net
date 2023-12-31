"""
Define all the file paths used for data input and model output.
All paths are constructed based on the root path of this repository (PhraseCutDataset).
Suggested usage: run from project root path and use default settings below.
"""

from pathlib import Path
import os

# set up api path
# f_path = Path.resolve(Path(__file__))
# api_path = f_path.parent.parent
# assert api_path.match('*/PhraseCutDataset')

# dataset_dir and paths to files
# dataset_dir = api_path.joinpath('data/VGPhraseCut_v0')
dataset_dir = '/home/lingpeng/project/PhraseCutDataset/data/VGPhraseCut_v0/'
# img_fpath = dataset_dir.joinpath('images')
img_fpath = os.path.join(dataset_dir, 'images', 'VG_100K')
# name_att_rel_count_fpath = dataset_dir.joinpath('name_att_rel_count.json')
name_att_rel_count_fpath = os.path.join(dataset_dir, 'name_att_rel_count.json')
# img_info_fpath = dataset_dir.joinpath('image_data_split.json')
img_info_fpath = os.path.join(dataset_dir, 'image_data_split.json')
# skip_fpath = dataset_dir.joinpath('skip.json')
skip_fpath = os.path.join(dataset_dir, 'skip.json')

refer_fpaths = dict()
refer_input_fpaths = dict()
vg_scene_graph_fpaths = dict()
for split in ['train', 'val', 'test', 'miniv']:
    # refer_fpaths[split] = dataset_dir.joinpath('refer_%s.json' % split)
    refer_fpaths[split] = os.path.join(dataset_dir, 'refer_%s.json' % split)
    # refer_input_fpaths[split] = dataset_dir.joinpath('refer_input_%s.json' % split)
    refer_input_fpaths[split] = os.path.join(dataset_dir, 'refer_input_%s.json' % split)
    # vg_scene_graph_fpaths[split] = dataset_dir.joinpath('scene_graphs_%s.json' % split)
    vg_scene_graph_fpaths[split] = os.path.join(dataset_dir, 'scene_graphs_%s.json' % split)

# output paths
# output_path = api_path.joinpath('output')
output_path = os.path.join(dataset_dir, 'output')
# summary_path = output_path.joinpath('eval_refvg')
summary_path = os.path.join(dataset_dir, 'eval_refvg')

# visualization paths
# gt_plot_path_color = dataset_dir.joinpath('visualizations/color')
gt_plot_path_color = os.path.join(dataset_dir, 'visualizations/color')
# gt_plot_path_gray = dataset_dir.joinpath('visualizations/gray')
gt_plot_path_gray = os.path.join(dataset_dir, 'visualizations/gray')
