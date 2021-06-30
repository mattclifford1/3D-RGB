import cv2
import os
import argparse
from tqdm import tqdm
import yaml
import numpy as np


def get_ims(files_list):
    im_list = [cv2.imread(file) for file in files_list]
    if len(files_list) == 4:
        top = np.concatenate([im_list[0], im_list[1]], axis=1)
        bottom = np.concatenate([im_list[2], im_list[3]], axis=1)
        return np.concatenate([top, bottom], axis=0)
    else:
        return np.concatenate(im_list, axis=1)

def list_to_vid(files_dict, config, vid_name):
    img = cv2.imread(files_dict[0][0])
    height, width, layers = img.shape
    num_cams = len(files_dict[0])
    if num_cams == 4: # 2x2 videos
        size = (width*2, height*2)
    else:                         # tile horizontally
        size = (width*num_cams, height*num_cams)

    out_file = os.path.join(config['save_vid_dir'], vid_name+config['vid_format'])
    os.makedirs(config['save_vid_dir'], exist_ok=True)
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), config['fps'], size)

    print('======= Writing Video =======')
    num_list = sorted(list(files_dict.keys()))
    for id in tqdm(range(len(num_list))):
        img = get_ims(files_dict[num_list[id]])
        out.write(img)
    out.release()

def get_files_dict(base_dir, vid, inner_dir, cam_type):
    vid_dir = os.path.join(base_dir, vid, inner_dir)
    base_files = os.listdir(os.path.join(vid_dir, cam_type[0]))
    file_dict = {}
    for file in base_files:
        multi_files = []
        for cam in cam_type:
            multi_files.append(os.path.join(vid_dir, cam, file))
        file_dict[int(file.split('.')[0])] = multi_files
    return file_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='postprocess.yml',help='Configure of post processing')
    parser.add_argument('--vid', type=str, help='Specific video to process')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    files_dict = get_files_dict(config['tgt_dir'],
                                args.vid,
                                config['vid_folder'],
                                config['vid_types'])
    list_to_vid(files_dict, config, args.vid)
