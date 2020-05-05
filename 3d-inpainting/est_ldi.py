
import utils_extra

def run_samples(samples, config):
    clock = utils_extra.timer()
    device = utils_extra.set_device(config)
    print("Set up time: " + clock.run_time())

    for idx in range(samples.data_num):
        print(samples.depth_file[idx])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    samples = utils_extra.data_files(config['src_folder'],
                                  config['depth_folder'],
                                  config['mesh_folder'])
    samples.collect_depth()
    run_samples(samples, config)
