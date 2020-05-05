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
from mesh import write_ply, read_ply, output_3d_photo
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

def run_samples(samples, config):
    clock = utils_extra.timer()
    device = utils_extra.set_device(config)
    print("Set up time: " + clock.run_time())

    for idx in range(samples.data_num):
        print(samples.im_file[idx])
        print(samples.depth_file[idx])
        print(samples.ldi_file[idx])
        image = imageio.imread(samples.im_file[idx])
        int_mtx = utils_extra.int_mtx_CPY(image)

        config['output_h'], config['output_w'] = np.load(samples.depth_file[idx]).shape[:2]
        frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
        config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
        config['original_h'], config['original_w'] = config['output_h'], config['output_w']
        if image.ndim == 2:
            image = image[..., None].repeat(3, -1)
        if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
            config['gray_image'] = True
        else:
            config['gray_image'] = False
        image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
        depth = read_MiDaS_depth(samples.depth_file[idx], 3.0, config['output_h'], config['output_w'])
        mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
            vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
            depth = vis_depths[-1]
            model = None
            torch.cuda.empty_cache()
            print("Start Running 3D_Photo ...")
            print(f"Loading edge model at {time.time()}")
            depth_edge_model = Inpaint_Edge_Net(init_weights=True)
            depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                           map_location=torch.device(device))
            depth_edge_model.load_state_dict(depth_edge_weight)
            depth_edge_model = depth_edge_model.to(device)
            depth_edge_model.eval()

            print(f"Loading depth model at {time.time()}")
            depth_feat_model = Inpaint_Depth_Net()
            depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                           map_location=torch.device(device))
            depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
            depth_feat_model = depth_feat_model.to(device)
            depth_feat_model.eval()
            depth_feat_model = depth_feat_model.to(device)
            print(f"Loading rgb model at {time.time()}")
            rgb_model = Inpaint_Color_Net()
            rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                         map_location=torch.device(device))
            rgb_model.load_state_dict(rgb_feat_weight)
            rgb_model.eval()
            rgb_model = rgb_model.to(device)
            graph = None

            print(f"Writing depth ply (and basically doing everything) at {time.time()}")
            rt_info = write_ply(image,
                                  depth,
                                  int_mtx,
                                  samples.ldi_file[idx],
                                  config,
                                  rgb_model,
                                  depth_edge_model,
                                  depth_edge_model,
                                  depth_feat_model)

            if rt_info is False:
                continue
            rgb_model = None
            color_feat_model = None
            depth_edge_model = None
            depth_feat_model = None
            torch.cuda.empty_cache()
        if config['save_ply'] is True or config['load_ply'] is True:
            verts, colors, faces, Height, Width, hFov, vFov = read_ply(samples.ldi_file[idx])
        else:
            verts, colors, faces, Height, Width, hFov, vFov = rt_info




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    samples = utils_extra.data_files(config['src_folder'],
                                  config['depth_folder'],
                                  config['mesh_folder'])
    samples.collect_depth()
    run_samples(samples, config)
