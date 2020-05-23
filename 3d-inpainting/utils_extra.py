import time
import os
import numpy as np
import utils
import cv2

def set_device(config):
    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
        dev = 'GPU'
    else:
        device = "cpu"
        dev = "CPU"
    print("Running on Device: " + dev)
    return device


def int_mtx_CPY(image):
    H, W = image.shape[:2]
    int_mtx = np.array([[max(H, W), 0, W//2], [0, max(H, W), H//2], [0, 0, 1]]).astype(np.float32)
    if int_mtx.max() > 1:
        int_mtx[0, :] = int_mtx[0, :] / float(W)
        int_mtx[1, :] = int_mtx[1, :] / float(H)
    return int_mtx

def tgts_poses_CPY(config):
    '''
    get the pos for camera (i think) for all frames in a pose
    '''
    tgts_poses = []
    generic_pose = np.eye(4)
    for traj_idx in range(len(config['traj_types'])):
        tgt_poses = []
        sx, sy, sz = utils.path_planning(config['num_frames'], config['x_shift_range'][traj_idx], config['y_shift_range'][traj_idx],
                                   config['z_shift_range'][traj_idx], path_type=config['traj_types'][traj_idx])
        for xx, yy, zz in zip(sx, sy, sz):
            tgt_poses.append(generic_pose * 1.)
            tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
        tgts_poses += [tgt_poses]
    return tgts_poses


def tgt_name_CPY(depth_file):
    tgt_name = [os.path.splitext(os.path.basename(depth_file))[0]]
    return tgt_name


def tgt_pose_CPY():
    generic_pose = np.eye(4)
    tgt_pose = [[generic_pose * 1]]
    tgt_pose = generic_pose * 1
    return tgt_pose


def vid_meta_CPY(config, depth_file):
    config['output_h'], config['output_w'] = np.load(depth_file).shape[:2]
    frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
    output_h, output_w = int(config['output_h'] * frac), int(config['output_w'] * frac)
    return output_h, output_w

def vid_meta_CPY_legacy(config, depth_file):
    config['output_h'], config['output_w'] = np.load(depth_file).shape[:2]
    frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
    config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
    config['original_h'], config['original_w'] = config['output_h'], config['output_w']
    return config


def read_depth(file, depth_rescale=3.0, out_height=None, out_width=None, midas_process=False, bits=16):
    '''
    Taken from base code that reads MiDaS estimated depth
    '''
    if bits == 16:
        depth = cv2.imread(file, -1)
        # depth = (depth>>3)|(depth<<13)  # if bit shifted
    else:
        depth = cv2.imread(file)
    if len(depth.shape) == 3:
        if (depth[:,:,0]==depth[:,:,1]).all() and (depth[:,:,1]==depth[:,:,2]).all():
            depth = depth[:,:,0]
        else:
            depth = np.average(depth, axis=2)
    if midas_process:
        depth = depth - depth.min()
        depth = cv2.blur(depth / depth.max(), ksize=(3, 3)) * depth.max()
        depth = (depth / depth.max()) * depth_rescale
    if out_height is not None and out_width is not None:
        depth = resize(depth / depth.max(), (out_height, out_width), order=1) * depth.max()
    if midas_process:
        depth = 1. / np.maximum(depth, 0.05)
    return depth


class timer:
    def __init__(self):
        self.start_time = time.time()
        self.temp_time = time.time()

    def run_time(self):
        tmp = time.time() - self.temp_time
        self.reset()
        return self.format_str(tmp)

    def reset(self):
        self.temp_time = time.time()

    def total_time(self):
        return self.format_str(time.time() - self.start_time)

    def format_str(self, time_float):
        if time_float > 60:
            time_float /= 60
            return "{:.2f}".format(time_float) + ' mins'
        else:
            return "{:.2f}".format(time_float) + ' secs'


class data_files:
    '''
    for giving directory names and collecting all the relevent files
    from each to read/ write data
    '''
    def __init__(self, src_dir, tgt_dir, vid):
        self.im_dir = os.path.join(src_dir, vid)
        self.depth_dir = os.path.join(tgt_dir, vid, 'depth')
        self.ldi_dir = os.path.join(tgt_dir, vid, 'ldi')
        self.video_dir = os.path.join(tgt_dir, vid, 'video-frames')
        self.im_formats=['jpg', 'png', 'jpeg']
        self.depth_formats=['npy', 'png']
        self.ldi_formats=['ply']
        self.make_dirs(self.im_dir, self.depth_dir, self.ldi_dir, self.video_dir)
        self.depth_write = '.npy'
        self.ldi_write = '.ply'


    def make_dirs(self, *args):
        for dir in args:
            os.makedirs(dir, exist_ok=True)


    def collect_rgb(self):
        self.im_file = self.remove_computed(self.im_dir, self.depth_dir)
        self.im_file = self.clean(self.im_file, self.im_formats, self.im_dir)
        [self.depth_file, self.ldi_file] = self.make_extra_files([[self.depth_write, self.depth_dir],
                                                                  [self.ldi_write, self.ldi_dir]])


    def collect_depth(self):
        self.depth_file = self.remove_computed(self.depth_dir, self.ldi_dir)
        self.depth_file = self.clean(self.depth_file, self.depth_formats, self.depth_dir)
        [self.ldi_file] = self.make_extra_files([[self.ldi_write, self.ldi_dir]])
        self.im_file = self.collect_extra_files(self.im_dir, self.im_formats)


    def collect_ldi(self):
        self.ldi_file = self.remove_computed(self.ldi_dir, self.video_dir)
        self.ldi_file = self.clean(self.ldi_file, self.ldi_formats, self.ldi_dir)
        self.im_file = self.collect_extra_files(self.im_dir, self.im_formats)
        self.depth_file = self.collect_extra_files(self.depth_dir, self.depth_formats)


    def clean(self, file_list, supported_formats, dir):
        '''
        remove all non-supported file types and add the directory
        into the file list
        '''
        file_list = self.remove_upsupported(file_list, supported_formats, type='format')
        self.base_file = [file.split('.')[0] for file in file_list]
        self.frame_num = [int(file) for file in self.base_file]
        self.frame_num = sorted(self.frame_num)
        self.data_num = len(file_list)
        file_dict = {}
        for file in file_list:
            file_dict[int(file.split('.')[0])] = os.path.join(dir, file)
        return file_dict


    def make_extra_files(self, list_data, computed_ok = True):
        out_dicts = []
        for file_format, file_dir in list_data:
            tmp_dict = {}
            if not computed_ok:
                print('Removing already computed files from list')
                already_run = os.listdir(file_dir)
                already_run = [os.path.join(file_dir, file) for file in already_run]
                for base in self.base_file:
                    file = base + file_format
                    if os.path.join(file_dir, file) in already_run:
                        print('Already compututed: '+file)
                    else:
                        tmp_dict[int(file.split('.')[0])] = os.path.join(file_dir, file)
            else:
                for base in self.base_file:
                    tmp_dict[int(base)] = os.path.join(file_dir, base+file_format)
            out_dicts.append(tmp_dict)
        return out_dicts



    def collect_extra_files(self, dir, supported_formats):
        '''
        WARNING: this quick and easy method assumes os.listdir returns
        in the exact same order for each dir
        '''
        file_list = os.listdir(dir)
        file_list = self.remove_upsupported(file_list, supported_formats, type='format')
        file_list = self.remove_upsupported(file_list, self.base_file, type='file')
        file_dict = {}
        for file in file_list:
            file_dict[int(file.split('.')[0])] = os.path.join(dir, file)
        return file_dict


    def remove_computed(self, data_dir, already_run_dir):
        data_list = os.listdir(data_dir)
        already_run = os.listdir(already_run_dir)
        base_already_run = [file.split('.')[0] for file in already_run]
        remove_list = []
        for file in data_list:
            if file.split('.')[0] in base_already_run:
                remove_list.append(file)
        for remove_file in remove_list:
            data_list.remove(remove_file)
        return data_list


    def remove_upsupported(self, file_list, supported_formats, type='format'):
        if type == 'format':
            type = -1
        elif type == 'file':
            type = 0
        remove_list = []
        for file in file_list:
            if file.split('.')[type] not in supported_formats:
                remove_list.append(file)
        for remove_file in remove_list:
            file_list.remove(remove_file)
        return file_list
