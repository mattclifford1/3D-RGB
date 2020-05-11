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
    constructer = mesh.frame_constucter(config, samples.im_file[0], samples.depth_file[0])
    for idx in range(samples.data_num):
        print(samples.ldi_file[idx])
        constructer.get_frame(samples.ldi_file[idx],
                              samples.frame_num[idx],
                              samples.depth_file[idx])
        if config['verbose']:
            print("Constructed frame in: " + clock.run_time())
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
