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

def load_models(config, clock):
    device = utils_extra.set_device(config)
    torch.cuda.empty_cache()
    depth_edge_model = Inpaint_Edge_Net(init_weights=True)
    depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                   map_location=torch.device(device))
    depth_edge_model.load_state_dict(depth_edge_weight)
    depth_edge_model = depth_edge_model.to(device)
    depth_edge_model.eval()
    if config['verbose']:
        print("Loaded edge model in: " + clock.run_time())
    depth_feat_model = Inpaint_Depth_Net()
    depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                   map_location=torch.device(device))
    depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
    depth_feat_model = depth_feat_model.to(device)
    depth_feat_model.eval()
    depth_feat_model = depth_feat_model.to(device)
    if config['verbose']:
        print("Loaded depth model in: " + clock.run_time())
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


def run_samples(samples, config):
    clock = utils_extra.timer()
    rgb_model, depth_edge_model, depth_feat_model = load_models(config, clock)
    print('Estimating Frames...')
    for id in tqdm(range(samples.data_num)):
        idx = samples.frame_num[id]
        # try:
        image = imageio.imread(samples.im_file[idx])
        int_mtx = utils_extra.int_mtx_CPY(image)
        if samples.depth_file[idx].split('.')[-1] == 'npy':
            config['output_h'], config['output_w'] = np.load(samples.depth_file[idx]).shape[:2]
            config = edit_sizes(config)
            depth = read_MiDaS_depth(samples.depth_file[idx], 3.0, config['output_h'], config['output_w'])
        else:
            config['output_h'], config['output_w'] = cv2.imread(samples.depth_file[idx]).shape[:2]
            config['output_h'], config['output_w'] = 540, 960#720
            config = edit_sizes(config)
            depth = utils_extra.read_depth(samples.depth_file[idx],
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
        if config['verbose']:
            print("Loaded rgb model in: " + clock.run_time())
            print("======= Starting LDI estimation ======")
        clock.reset()
        rt_info = mesh.write_ply(image,
                                 depth,
                                 int_mtx,
                                 samples.ldi_file[idx],
                                 config,
                                 rgb_model,
                                 depth_edge_model,
                                 depth_edge_model,
                                 depth_feat_model)
        # torch.cuda.empty_cache()
        if config['verbose']:
            print(samples.im_file[idx])
            print(samples.depth_file[idx])
            print(samples.ldi_file[idx])
            print("Estimated LDI in: " + clock.run_time())
        # except:
        #     print('Error with file:'+samples.depth_file[idx])
    print("Estimated frames in: " + clock.total_time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    parser.add_argument('--vid', type=str, help='Specific video to process')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    samples = utils_extra.data_files(config['src_dir'],
                                     config['tgt_dir'],
                                     args.vid)
    samples.collect_depth()
    run_samples(samples, config)
