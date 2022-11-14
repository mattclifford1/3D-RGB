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
import sys; sys.path.append('..'); sys.path.append('.')
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

def run_sample(config, im, depth, ldi, idx, vid_length, video_dir):
    for track_type in config['video_postfix']:
        os.makedirs(os.path.join(video_dir, track_type), exist_ok=True)
    constructer = render_3D.frame_constucter(config,
                                            im,
                                            depth,
                                            vid_length)
    # constructer.load_ply(samples.ldi_file[idx])
    frames_dict = constructer.get_frame(ldi,
                                        idx,
                                        depth)
    for track_type, frame in frames_dict.items():
        write_file = os.path.join(video_dir, track_type, str(idx)+config['img_format'])
        cv2.imwrite(write_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    parser.add_argument('--im', type=str, help='Video to process', required=True)
    parser.add_argument('--depth', type=str, help='Video to process', required=True)
    parser.add_argument('--ldi', type=str, help='Video to process', required=True)
    parser.add_argument('--video_dir', type=str, help='Video to process', required=True)
    parser.add_argument('--frame_num', type=int, help='Video to process', required=True)
    parser.add_argument('--vid_length', type=int, help='Video to process', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    run_sample(config, args.im, args.depth, args.ldi, args.frame_num, args.vid_length, args.video_dir)
