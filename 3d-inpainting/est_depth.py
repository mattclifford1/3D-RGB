import argparse
import yaml
import numpy as np
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
import utils_extra
import cv2


def run_samples(samples, config):
    clock = utils_extra.timer()
    device  = utils_extra.set_device(config)
    print("Set up time: " + clock.run_time())

    for idx in range(samples.data_num):
        print(samples.im_file[idx])
        depth = run_depth(samples.im_file[idx],
                          config['src_folder'],
                          config['depth_folder'],
                          config['MiDaS_model_ckpt'],
                          MonoDepthNet,
                          MiDaS_utils,
                          target_w=640.0)
        # save results
        np.save(samples.depth_file[idx], depth)
        depth = (depth+np.min(depth))/np.max(depth)
        depth = depth*255
        cv2.imwrite(samples.depth_file[idx].split('.')[0]+'.png', depth)

        print("Depth estimated in: " + clock.run_time())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    samples = utils_extra.data_files(config['src_folder'],
                                  config['depth_folder'],
                                  config['mesh_folder'])
    samples.collect_rgb()
    run_samples(samples, config)
