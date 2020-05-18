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
    for file in already_run:
        if file.split('.')[0] in samples.depth_file.keys():
            fail_count += 1
            print('--------- Fail: '+file+ '-----------')
    if fail_count == 0:
        print('++++++++++ PASS +++++++++++')


def sort_lists(samples):
    samples.collect_ldi()
    fail_count = 0
    for id in samples.frame_num:
        if diff_base(samples.im_file[id], samples.depth_file[id], samples.ldi_file[id]):
            fail_count += 1
            print('============****************************===========')
            print('--------- Fail: '+samples.im_file[id]+ '-----------')
            print('--------- Fail: '+samples.depth_file[id]+ '-----------')
            print('--------- Fail: '+samples.ldi_file[id]+ '-----------')
            print('============****************************===========')
        if fail_count == 0:
            print('++++++++++ PASS +++++++++++')


def diff_base(file1, file2, file3):
    if file1.split('/')[-1].split('.')[0] != file2.split('/')[-1].split('.')[0]:
        return True
    elif file1.split('/')[-1].split('.')[0] != file3.split('/')[-1].split('.')[0]:
        return True
    elif file2.split('/')[-1].split('.')[0] != file3.split('/')[-1].split('.')[0]:
        return True
    else:
        return False


def run_ordering(samples):
    samples.collect_ldi()
    for id in range(samples.data_num):
        # print(samples.frame_num[id])
        id = samples.frame_num[id]
        # print(samples.im_file[id])
        # print(samples.depth_file[id])
        # print(samples.ldi_file[id])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yaml to control run')
    parser.add_argument('--vid', type=str, help='Specific video to process')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    samples = utils_extra.data_files(config['src_dir'],
                                  config['tgt_dir'],
                                  args.vid)
    samples.collect_depth()

    compare_not_run(samples)
    test_removed_input(samples)
    samples = utils_extra.data_files(config['src_dir'],
                                  config['tgt_dir'],
                                  args.vid)
    sort_lists(samples)
    run_ordering(samples)
