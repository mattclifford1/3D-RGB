import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
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


def run_samples(samples):
    clock = utils_extra.timer()

    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
    else:
        device = "cpu"

    print("Running on Device: {device}")
    print("Set up time: " + clock.run_time())

    for idx in range(samples.num_ims):
        print(samples.im_file[idx])
        depth = run_depth(samples.im_file[idx],
                          config['src_folder'],
                          config['depth_folder'],
                          config['MiDaS_model_ckpt'],
                          MonoDepthNet,
                          MiDaS_utils,
                          target_w=640.0)
        np.save(samples.depth_file[idx], depth)

        print("Depth estimated in: " + clock.run_time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    samples = utils_extra.im_data(config['src_folder'],
                                  config['depth_folder'],
                                  config['mesh_folder'])

    run_samples(samples)
