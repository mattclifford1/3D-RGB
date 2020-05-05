import time
import os
import numpy as np

def set_device(config):
    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
    else:
        device = "cpu"
    print("Running on Device: " + device)
    return device


def int_mtx_CPY(image):
    H, W = image.shape[:2]
    int_mtx = np.array([[max(H, W), 0, W//2], [0, max(H, W), H//2], [0, 0, 1]]).astype(np.float32)
    if int_mtx.max() > 1:
        int_mtx[0, :] = int_mtx[0, :] / float(W)
        int_mtx[1, :] = int_mtx[1, :] / float(H)
    return int_mtx


class timer:
    def __init__(self):
        self.start_time = time.time()
        self.temp_time = time.time()

    def run_time(self):
        tmp = time.time() - self.temp_time
        self.reset()
        return "{:.2f}".format(tmp)

    def reset(self):
        self.temp_time = time.time()

    def total_time(self):
        return "{:.2f}".format(time.time() - self.start_time)


class data_files:
    '''
    for giving directory names and collecting all the relevent files
    from each to read/ write data
    '''
    def __init__(self, im_dir, depth_dir, ldi_dir):
        self.im_dir = im_dir
        self.depth_dir = depth_dir
        self.ldi_dir = ldi_dir
        self.im_formats=['jpg', 'png', 'jpeg']
        self.depth_formats=['npy']
        self.ldi_formats=['ply']
        self.make_dirs(im_dir, depth_dir, ldi_dir)
        self.depth_write = '.npy'
        self.ldi_write = '.ply'


    def make_dirs(self, *args):
        for dir in args:
            os.makedirs(dir, exist_ok=True)


    def collect_rgb(self):
        self.im_file = os.listdir(self.im_dir)
        self.im_file = self.clean(self.im_file, self.im_formats, self.im_dir)
        self.depth_file = []
        self.ldi_file = []
        self.make_extra_files([[self.depth_file, self.depth_write, self.depth_dir],
                               [self.ldi_file, self.ldi_write, self.ldi_dir]])


    def collect_depth(self):
        self.depth_file = os.listdir(self.depth_dir)
        self.depth_file = self.clean(self.depth_file, self.depth_formats, self.depth_dir)
        self.ldi_file = []
        self.make_extra_files([[self.ldi_file, self.ldi_write, self.ldi_dir]])
        self.im_file = self.collect_extra_files(self.im_dir)


    def collect_ldi(self):
        self.ldi_file = os.listdir(self.ldi_dir)
        self.ldi_file = self.clean(self.ldi_file, self.ldi_formats, self.ldi_dir)
        self.im_file = self.collect_extra_files(self.im_dir)
        self.depth_file = self.collect_extra_files(self.depth_dir)


    def clean(self, file_list, supported_formats, dir):
        '''
        remove all non-supported file types and add the directory
        into the file list
        '''
        for file in file_list:
            if file.split('.')[-1] not in supported_formats:
                file_list.remove(file)
        self.base_file = [file.split('.')[0] for file in file_list]
        file_list = [os.path.join(dir, file) for file in file_list]
        self.data_num = len(file_list)
        return file_list


    def make_extra_files(self, list_data):
        for tmp_list, file_format, file_dir in list_data:
            for base in self.base_file:
                tmp_list.append(os.path.join(file_dir, base+file_format))


    def collect_extra_files(self, dir):
        '''
        WARNING: this quick and easy method assumes os.listdir returns
        in the exact same order for each dir
        '''
        tmp_list = os.listdir(dir)
        for file in tmp_list:
            if file.split('.')[0] not in self.base_file:
                tmp_list.remove(file)
        tmp_list = [os.path.join(dir, file) for file in tmp_list]
        return tmp_list
