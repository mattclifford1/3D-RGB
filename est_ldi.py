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
from p_tqdm import p_map

def call_thread(id, config, samples, sh_file='./run_single-ldi.sh'):
    # run in separate call to stop memory leakage bug
    idx = samples.frame_num[id]
    os.system(sh_file+' '+config+' '+samples.im_file[idx]
                                +' '+samples.depth_file[idx]
                                +' '+samples.ldi_file[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    parser.add_argument('--vid', type=str, help='Specific video to process')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    samples = utils_extra.data_files(config['src_dir'],
                                     config['tgt_dir'],
                                     args.vid)
    samples.collect_depth()
    print('Estimating LDI ...')
    # in parrelel
    r = p_map(partial(call_thread,
                      config=args.config,
                      samples=samples),
              list(range(samples.data_num)),
              num_cpus=config['threads'])
    # sequencial
    # for id in tqdm(range(samples.data_num)):
    #     call_thread(id, args.config, args.vid)
