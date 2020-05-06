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
    print('Contructing Video...')
    for idx in range(samples.data_num):
        config = utils_extra.vid_meta_CPY(config, samples.depth_file[idx])
        image = imageio.imread(samples.im_file[idx])
        depth = read_MiDaS_depth(samples.depth_file[idx], 3.0, config['output_h'], config['output_w'])
        mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        # load video params (maker nice - currently lifted from orig code)
        int_mtx = utils_extra.int_mtx_CPY(image)
        tgts_poses = utils_extra.tgts_poses_CPY(config)
        tgt_name = utils_extra.tgt_name_CPY(samples.depth_file[idx])
        tgt_pose = utils_extra.tgt_pose_CPY()
        ref_pose = np.eye(4)
        normal_canvas, all_canvas = None, None
        # load LDI
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(samples.ldi_file[idx])
        if config['verbose']:
            print("Loaded LDI in: " + clock.run_time())
        videos_poses, video_basename = copy.deepcopy(tgts_poses), tgt_name
        top = (config.get('original_h') // 2 - int_mtx[1, 2] * config['output_h'])
        left = (config.get('original_w') // 2 - int_mtx[0, 2] * config['output_w'])
        down, right = top + config['output_h'], left + config['output_w']
        border = [int(xx) for xx in [top, down, left, right]]
        normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                            copy.deepcopy(tgt_pose), config['video_postfix'], copy.deepcopy(ref_pose), copy.deepcopy(config['video_folder']),
                            image.copy(), copy.deepcopy(int_mtx), config, image,
                            videos_poses, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                            mean_loc_depth=mean_loc_depth)
        if config['verbose']:
            print("Constructed video in: " + clock.run_time())
    print("Constructed videos in: " + clock.total_time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    samples = utils_extra.data_files(config['src_folder'],
                                  config['depth_folder'],
                                  config['mesh_folder'])
    samples.collect_ldi()
    run_samples(samples, config)
