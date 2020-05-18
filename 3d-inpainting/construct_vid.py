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
import render_3D

def run_samples(samples, config):
    clock = utils_extra.timer()
    for track_type in config['video_postfix']:
        os.makedirs(os.path.join(samples.video_dir, track_type), exist_ok=True)
    print('Contructing Video...')
    constructer = render_3D.frame_constucter(config, samples.im_file[0], samples.depth_file[0], len(samples.frame_num))
    for id in tqdm(range(samples.data_num)):
        idx = samples.frame_num[id]
        # constructer.load_ply(samples.ldi_file[idx])
        frames_dict = constructer.get_frame(samples.ldi_file[idx],
                                            samples.frame_num[idx],
                                            samples.depth_file[idx])
        if config['verbose']:
            print("Constructed frame in: " + clock.run_time())
        for track_type, frame in frames_dict.items():
            write_file = os.path.join(samples.video_dir, track_type, str(samples.frame_num[idx])+config['img_format'])
            print(write_file)
            cv2.imwrite(write_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print("Constructed videos in: " + clock.total_time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    parser.add_argument('--vid', type=str, help='Specific video to process')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    if config['server']:
        print('Using Mesa graphics backend for headless server')
        vispy.use('osmesa')
    samples = utils_extra.data_files(config['src_dir'],
                                     config['tgt_dir'],
                                     args.vid)
    samples.collect_ldi()
    run_samples(samples, config)
