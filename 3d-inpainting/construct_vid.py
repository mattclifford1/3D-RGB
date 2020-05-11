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

def run_samples(samples, config):
    clock = utils_extra.timer()
    print('Contructing Video...')
    constructer = mesh.frame_constucter(config)
    for idx in range(samples.data_num):
        print(samples.ldi_file[idx])
        output_h, output_w = utils_extra.vid_meta_CPY(config, samples.depth_file[idx])
        original_h, original_w = output_h, output_w  # elim this eventually
        image = imageio.imread(samples.im_file[idx])
        depth = read_MiDaS_depth(samples.depth_file[idx], 3.0, config['output_h'], config['output_w'])
        mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        # load video params (maker nice - currently lifted from orig code)
        int_mtx = utils_extra.int_mtx_CPY(image)
        # load LDI
        verts, colors, faces, Height, Width, hFov, vFov = mesh.read_ply(samples.ldi_file[idx])
        if config['verbose']:
            print("Loaded LDI in: " + clock.run_time())
        top = (original_h // 2 - int_mtx[1, 2] * output_h)
        left = (original_w // 2 - int_mtx[0, 2] * output_w)
        down, right = top + output_h, left + output_w
        border = [int(xx) for xx in [top, down, left, right]]
        constructer.get_frame(samples.ldi_file[idx],
                              copy.deepcopy(int_mtx),
                              output_h, output_w,
                              border=border,
                              depth=depth,
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
