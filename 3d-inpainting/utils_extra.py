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


class im_data:
    def __init__(self, im_dir, depth_dir, mesh_dir):
        self.supported_formats = ['jpg', '.png', '.jpeg']
        self.im_dir = im_dir
        self.depth_dir = depth_dir
        self.mesh_dir = mesh_dir
        self.make_dirs(im_dir, depth_dir, mesh_dir)
        self.im_file = os.listdir(self.im_dir)
        self.clean()
        self.num_ims = len(self.im_file)
        self.make_extra_names()


    def make_dirs(self, *args):
        for dir in args:
            os.makedirs(dir, exist_ok=True)


    def clean(self, supported_formats=['jpg', 'png', 'jpeg']):
        clean_list = []
        for file in self.im_file:
            if file.split('.')[-1] in supported_formats:
                clean_list.append(file)
        self.base_file = [file.split('.')[0] for file in clean_list]
        self.im_file = [os.path.join(self.im_dir,file) for file in clean_list]


    def make_extra_names(self, depth_format='.npy', mesh_format='.ply'):
        self.depth_file = []
        self.mesh_file = []
        for base in self.base_file:
            self.depth_file.append(os.path.join(self.depth_dir, base+depth_format))
            self.mesh_file.append(os.path.join(self.mesh_dir, base+mesh_format))
