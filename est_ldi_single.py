import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
import mesh
from utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering

import utils_extra

def load_models(config):
    device = utils_extra.set_device(config)
    torch.cuda.empty_cache()
    depth_edge_model = Inpaint_Edge_Net(init_weights=True)
    depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                   map_location=torch.device(device))
    depth_edge_model.load_state_dict(depth_edge_weight)
    depth_edge_model = depth_edge_model.to(device)
    depth_edge_model.eval()
    depth_feat_model = Inpaint_Depth_Net()
    depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                   map_location=torch.device(device))
    depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
    depth_feat_model = depth_feat_model.to(device)
    depth_feat_model.eval()
    depth_feat_model = depth_feat_model.to(device)
    rgb_model = Inpaint_Color_Net()
    rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                 map_location=torch.device(device))
    rgb_model.load_state_dict(rgb_feat_weight)
    rgb_model.eval()
    rgb_model = rgb_model.to(device)
    return rgb_model, depth_edge_model, depth_feat_model


def edit_sizes(config):
    frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
    config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
    config['original_h'], config['original_w'] = config['output_h'], config['output_w']
    return config


def run_sample(config, im, depth, ldi):
    rgb_model, depth_edge_model, depth_feat_model = load_models(config)
    if os.path.exists(ldi):
        return
    image = imageio.imread(im)
    int_mtx = utils_extra.int_mtx_CPY(image)
    if depth.split('.')[-1] == 'npy':
        config['output_h'], config['output_w'] = np.load(depth).shape[:2]
        config = edit_sizes(config)
        depth = read_MiDaS_depth(depth, 3.0, config['output_h'], config['output_w'])
    else:
        config['output_h'], config['output_w'] = cv2.imread(depth).shape[:2]
        config['output_h'], config['output_w'] = 540, 960#720
        config = edit_sizes(config)
        depth = utils_extra.read_depth(depth,
                                        config['depth_rescale'],
                                        config['output_h'],
                                        config['output_w'],
                                        config['inv_depth'])
    if image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
        config['gray_image'] = True
    else:
        config['gray_image'] = False
    image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
    mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
    vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
    depth = vis_depths[-1]
    rt_info = mesh.write_ply(image,
                             depth,
                             int_mtx,
                             ldi,
                             config,
                             rgb_model,
                             depth_edge_model,
                             depth_edge_model,
                             depth_feat_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    parser.add_argument('--im', type=str, help='Video to process', required=True)
    parser.add_argument('--depth', type=str, help='Video to process', required=True)
    parser.add_argument('--ldi', type=str, help='Video to process', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    run_sample(config, args.im, args.depth, args.ldi)
