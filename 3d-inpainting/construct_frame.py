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

def run_sample(samples, config, id):
    for track_type in config['video_postfix']:
        os.makedirs(os.path.join(samples.video_dir, track_type), exist_ok=True)
    first_file = samples.frame_num[0]
    constructer = render_3D.frame_constucter(config,
                                            samples.im_file[first_file],
                                            samples.depth_file[first_file],
                                            samples.data_num)
    idx = samples.frame_num[id]
    # constructer.load_ply(samples.ldi_file[idx])
    frames_dict = constructer.get_frame(samples.ldi_file[idx],
                                        idx,
                                        samples.depth_file[idx])
    for track_type, frame in frames_dict.items():
        write_file = os.path.join(samples.video_dir, track_type, str(idx)+config['img_format'])
        cv2.imwrite(write_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    parser.add_argument('--vid', type=str, help='Video to process', required=True)
    parser.add_argument('--frame', type=int, help='Frame to proces')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    samples = utils_extra.data_files(config['src_dir'],
                                     config['tgt_dir'],
                                     args.vid)
    samples.collect_ldi()
    run_sample(samples, config, args.frame)
