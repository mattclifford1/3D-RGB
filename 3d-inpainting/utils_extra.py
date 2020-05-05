import time
import os

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
        self.make_extra_names([[self.depth_file, self.depth_write, self.depth_dir],
                               [self.ldi_file, self.ldi_write, self.ldi_dir]])


    def collect_depth(self):
        self.depth_file = os.listdir(self.depth_dir)
        self.depth_file = self.clean(self.depth_file, self.depth_formats, self.depth_dir)
        self.ldi_file = []
        self.make_extra_names([[self.ldi_file, self.ldi_write, self.ldi_dir]])


    def collect_ldi(self):
        self.ldi_file = os.listdir(self.ldi_dir)
        self.ldi_file = self.clean(self.ldi_file, self.ldi_formats, self.ldi_dir)


    def clean(self, file_list, supported_formats, dir):
        '''
        remove all non-supported file types and add the directory
        into the file list
        '''
        for ind, file in enumerate(file_list):
            if file.split('.')[-1] not in supported_formats:
                file_list.pop(ind)
        self.base_file = [file.split('.')[0] for file in file_list]
        file_list = [os.path.join(dir, file) for file in file_list]
        self.data_num = len(file_list)
        return file_list


    def make_extra_names(self, list_data):
        for tmp_list, file_format, file_dir in list_data:
            for base in self.base_file:
                tmp_list.append(os.path.join(file_dir, base+file_format))
