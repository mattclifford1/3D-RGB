import cv2
import os
import argparse
from tqdm import tqdm
import yaml


def list_to_vid(files_dict, config, vid_name):
    img = cv2.imread(files_dict[0])
    height, width, layers = img.shape
    size = (width,height)
    out_file = os.path.join(config['save_dir'], vid_name+config['vid_format'])
    os.makedirs(config['save_dir'], exist_ok=True)
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), config['fps'], size)
    print('======= Writing Video =======')
    for id in tqdm(range(len(files_dict))):
        img = cv2.imread(files_dict[id])
        out.write(img)
    out.release()

def get_files_dict(vids_dir, vid):
    vid_dir = os.path.join(vids_dir, vid)
    file_list = os.listdir(vid_dir)
    file_dict = {}
    for file in file_list:
        file_dict[int(file.split('.')[0])] = os.path.join(vid_dir, file)
    return file_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    parser.add_argument('--vid', type=str, help='Specific video to process')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))

    files_dict = get_files_dict(config['input_dir'], args.vid)
    list_to_vid(files_dict, config, args.vid)
