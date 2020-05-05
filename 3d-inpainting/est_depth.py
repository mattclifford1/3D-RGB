import argparse
import yaml
import numpy as np
from MiDaS.run import run_depth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
import utils_extra


def run_samples(samples):
    clock = utils_extra.timer()

    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
    else:
        device = "cpu"

    print("Running on Device: {device}")
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
        np.save(samples.depth_file[idx], depth)

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
    run_samples(samples)
