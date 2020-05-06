import numpy as np
import os
from PIL import Image
import cv2
from colourisation.utils import vid_utils
from colourisation.utils import im_utils
import multiprocessing
import argparse
import yaml


def get_list_of_ims(dir, filename, resize=None):
    end_of_video = False
    capture = cv2.VideoCapture(os.path.join(dir,filename))
    im_list = []
    while not end_of_video:
        ret, BGR = capture.read()
        if ret:
            frame = cv2.cvtColor(BGR, cv2.COLOR_RGB2BGR)
            if resize:
                frame = im_utils.resize_im_to(frame, resize)
            im_list.append(frame)
        else:
            capture.release()
        end_of_video = not ret
    return im_list

class simple_writer():
    def __init__(self, im_list, save_dir, filename, im_format='.jpg'):
        self.im_list = im_list
        self.save_dir = save_dir
        if im_format[0] != '.':
            im_format = '.' + im_format
        self.im_format = im_format
        self.new_dir = filename.split('.')[0]
        if not os.path.isdir(os.path.join(self.save_dir, self.new_dir)):
            os.mkdir(os.path.join(self.save_dir, self.new_dir))
        # pool = multiprocessing.Pool(processes=40)
        # pool.map(self.save_single, range(len(im_list)))
        # pool.close()
        for i in range(len(im_list)):
            self.save_single(i)


    def save_single(self, i):
        im = Image.fromarray(self.im_list[i])
        im.save(os.path.join(self.save_dir, self.new_dir, str(i)+self.im_format))


#TODO:   see if there is a ffmpeg command or similar for this
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='preprocess_vid.yml',help='Configure of video preprocessing')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    dir = '/home/nulightai/datasets/kinetics/processed'
    save_dir = '/home/nulightai/datasets/kinetics/frames'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        print('Already some frames processed')
    alread_processed = os.listdir(save_dir)
    vid_list = os.listdir(dir)
    for vid in vid_list:
        if vid.split('.')[0] in alread_processed:
            print('Skipping '+vid)
            continue
        print('read')
        im_list = get_list_of_ims(dir, vid)
        print('write')
        simple_writer(im_list, save_dir, vid)
