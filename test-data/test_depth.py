import argparse
import os
import sys
sys.path.append('..')
import utils_extra
import utils
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_depth(samples):
    found_npy = False
    found_png = False
    for id in samples.frame_num:
        if samples.depth_file[id].split('.')[-1] == 'npy':
            depth = utils.read_MiDaS_depth(samples.depth_file[id], 3.0)
            npy_shape = len(depth.shape)
            found_npy = True
        else:
            depth = utils_extra.read_depth(samples.depth_file[id], midas_process=False, bits=16)
            png_shape = len(depth.shape)
            found_png = True
        if found_npy and found_png:
            if png_shape == npy_shape:
                print('++++++++++ PASS +++++++++++')
            else:
                print('--------- Fail array depth shape -----------')
            break
    print(type(depth))
    print(depth.dtype)
    depth = depth.astype(np.uint16)
    plt.imshow(depth)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yaml to control run')
    parser.add_argument('--vid', type=str, help='Specific video to process')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    samples = utils_extra.data_files(config['src_dir'],
                                  config['tgt_dir'],
                                  args.vid)
    samples.collect_depth()

    show_depth(samples)
