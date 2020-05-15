import argparse
import os
import sys
sys.path.append('..')
import utils_extra
import yaml


def compare_not_run(samples):
    '''
    make sure to not include files that have already been
    compututed in a previous run and are therefore
    already located in the file directory
    '''
    already_run = os.listdir(samples.ldi_dir)
    already_run = [os.path.join(samples.ldi_dir, file) for file in already_run]
    fail_count = 0
    for file in already_run:
        if file in samples.ldi_file:
            fail_count += 1
            print('--------- Fail: '+file+ '-----------')
    if fail_count == 0:
        print('++++++++++ PASS +++++++++++')


def test_removed_input(samples):
    fail_count = 0
    already_run = os.listdir(samples.ldi_dir)
    depth_base = [(file.split('.')[0]).split('/')[-1] for file in samples.depth_file]
    for file in already_run:
        if file.split('.')[0] in depth_base:
            fail_count += 1
            print('--------- Fail: '+file+ '-----------')
    if fail_count == 0:
        print('++++++++++ PASS +++++++++++')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yaml to control run')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    samples = utils_extra.data_files(config['src_folder'],
                                  config['depth_folder'],
                                  config['mesh_folder'])
    samples.collect_depth()

    compare_not_run(samples)
    test_removed_input(samples)
    print(samples.depth_file)
    print(samples.ldi_file)
