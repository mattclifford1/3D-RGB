import os
import numpy as np
try:
    import cynetworkx as netx
except ImportError:
    import networkx as netx
import matplotlib.pyplot as plt
from functools import partial
from vispy import scene, io
from vispy.scene import visuals
from vispy.visuals.filters import Alpha
import cv2
from moviepy.editor import ImageSequenceClip
from skimage.transform import resize
import time
import copy
import torch
import os
from utils import path_planning, open_small_mask, clean_far_edge, refine_depth_around_edge
from utils import refine_color_around_edge, filter_irrelevant_edge_new, require_depth_edge, clean_far_edge_new
from utils import create_placeholder, refresh_node, find_largest_rect
from mesh_tools import get_depth_from_maps, get_map_from_ccs, get_edge_from_nodes, get_depth_from_nodes, get_rgb_from_nodes, crop_maps_by_size, convert2tensor, recursive_add_edge, update_info, filter_edge, relabel_node, depth_inpainting
from mesh_tools import refresh_bord_depth, enlarge_border, fill_dummy_bord, extrapolate, fill_missing_node, incomplete_node, get_valid_size, dilate_valid_size, size_operation
import transforms3d
import random
from functools import reduce
import copy
import utils
import utils_extra
import imageio
from mesh import *

'''
class adaptation for frame constructiong using
base code from output_3d_photo funciton above
'''
class frame_constucter:
    def __init__(self, config, im_file, depth_file, num_frames):
        self.ref_pose = np.eye(4)
        self.config = config
        self.num_frames = num_frames
        self.video_traj_types = self.config['video_postfix']
        self.make_tgts_poses()
        self.im_count = 0
        self.setup(im_file, depth_file)

    def setup(self, im_file, depth_file):
        '''
        set border and output sizes etc
        '''
        image = imageio.imread(im_file)
        self.int_mtx = utils_extra.int_mtx_CPY(image)
        self.output_h, self.output_w = utils_extra.vid_meta_CPY(self.config, depth_file)
        original_h, original_w = self.output_h, self.output_w  # elim this eventually
        top = (original_h // 2 - self.int_mtx[1, 2] * self.output_h)
        left = (original_w // 2 - self.int_mtx[0, 2] * self.output_w)
        down, right = top + self.output_h, left + self.output_w
        self.border = [int(xx) for xx in [top, down, left, right]]

    def make_tgts_poses(self):
        '''
        get the pos for camera (i think) for all frames in a pose
        '''
        self.videos_poses = []
        generic_pose = np.eye(4)   #copy from ref_pose???
        for traj_idx in range(len(self.config['traj_types'])):
            tgt_poses = []
            sx, sy, sz = utils.path_planning(self.num_frames,
                                             self.config['x_shift_range'][traj_idx],
                                             self.config['y_shift_range'][traj_idx],
                                             self.config['z_shift_range'][traj_idx],
                                             path_type=self.config['traj_types'][traj_idx])
            for xx, yy, zz in zip(sx, sy, sz):
                tgt_poses.append(generic_pose * 1.)
                if self.config['traj_types'][traj_idx] != 'control':
                    '''change if we require camera movement'''
                    tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
            self.videos_poses += [tgt_poses]

    def load_ply(self, file_name):
        '''
        take from run scipt - not sure if copy needed or not yet
        LOOK into memory usage here.......
        '''
        ply = read_ply(file_name)
        self.verts = ply[0].copy()
        self.colors = ply[1].copy()
        self.colors = self.colors[..., :3]
        self.faces = ply[2].copy()
        self.Height = copy.deepcopy(ply[3])
        self.Width = copy.deepcopy(ply[4])
        # self.hFov = copy.deepcopy(ply[5])
        # self.vFov = copy.deepcopy(ply[6])
        self.extract_info()

    def extract_info(self):
        '''
        TODO: save to self needed items
        '''
        self.cam_mesh_graph = {}
        self.cam_mesh_graph['H'] = self.Height
        self.cam_mesh_graph['W'] = self.Width
        self.cam_mesh_graph['original_H'] = self.original_H
        self.cam_mesh_graph['original_W'] = self.original_W
        int_mtx_real_x = self.int_mtx[0] * self.Width
        int_mtx_real_y = self.int_mtx[1] * self.Height
        self.cam_mesh_graph['hFov'] = 2 * np.arctan((1. / 2.) * ((self.cam_mesh_graph['original_W']) / int_mtx_real_x[0]))
        self.cam_mesh_graph['vFov'] = 2 * np.arctan((1. / 2.) * ((self.cam_mesh_graph['original_H']) / int_mtx_real_y[1]))

        self.fov_in_rad = max(self.cam_mesh_graph['vFov'], self.cam_mesh_graph['hFov'])
        self.fov = (self.fov_in_rad * 180 / np.pi)
        self.init_factor = 1
        if self.config.get('anti_flickering') is True:
            self.init_factor = 3
        if (self.cam_mesh_graph['original_H'] is not None) and (self.cam_mesh_graph['original_W'] is not None):
            canvas_w = self.cam_mesh_graph['original_W']
            canvas_h = self.cam_mesh_graph['original_H']
        else:
            canvas_w = self.cam_mesh_graph['W']
            canvas_h = self.cam_mesh_graph['H']
        self.canvas_size = max(canvas_h, canvas_w)

    def get_canvas(self):
        normal_canvas = Canvas_view(self.fov,
                                    canvas_size=self.canvas_size,
                                    factor=self.init_factor,
                                    bgcolor='gray',
                                    proj='perspective')
        return normal_canvas

    def get_anchor_plane(self, normal_canvas):
        img = normal_canvas.render()
        img = cv2.resize(img, (int(img.shape[1] / self.init_factor), int(img.shape[0] / self.init_factor)), interpolation=cv2.INTER_AREA)
        if self.border is None:
            self.border = [0, img.shape[0], 0, img.shape[1]]
        if (self.cam_mesh_graph['original_H'] is not None) and (self.cam_mesh_graph['original_W'] is not None):
            aspect_ratio = self.cam_mesh_graph['original_H'] / self.cam_mesh_graph['original_W']
        else:
            aspect_ratio = self.cam_mesh_graph['H'] / self.cam_mesh_graph['W']
        if aspect_ratio > 1:
            img_h_len = self.cam_mesh_graph['H'] if self.cam_mesh_graph.get('original_H') is None else self.cam_mesh_graph['original_H']
            img_w_len = img_h_len / aspect_ratio
            anchor = [0,
                      img.shape[0],
                      int(max(0, int((img.shape[1])//2 - img_w_len//2))),
                      int(min(int((img.shape[1])//2 + img_w_len//2), (img.shape[1])-1))]
        elif aspect_ratio <= 1:
            img_w_len = self.cam_mesh_graph['W'] if self.cam_mesh_graph.get('original_W') is None else self.cam_mesh_graph['original_W']
            img_h_len = img_w_len * aspect_ratio
            anchor = [int(max(0, int((img.shape[0])//2 - img_h_len//2))),
                      int(min(int((img.shape[0])//2 + img_h_len//2), (img.shape[0])-1)),
                      0,
                      img.shape[1]]
        anchor = np.array(anchor)
        plane_width = np.tan(self.fov_in_rad/2.) * np.abs(self.mean_loc_depth)
        return anchor, plane_width

    def get_frame(self, ply_file, num, depth_file):
        self.original_H, self.original_W = self.output_h, self.output_w
        depth = utils.read_MiDaS_depth(depth_file, 3.0, self.config['output_h'], self.config['output_w'])
        self.mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        self.load_ply(ply_file)
        # print('Running for all poses')
        output_dir = os.path.join(self.config['tgt_dir'], 'video-frames')
        os.makedirs(output_dir, exist_ok=True)
        normal_canvas = self.get_canvas()
        anchor, plane_width = self.get_anchor_plane(normal_canvas)
        normal_canvas.add_data(self.verts,
                               self.faces,
                               self.colors)
        frames_dict = {}
        for video_pose, video_traj_type in zip(self.videos_poses, self.video_traj_types):
            img_gen = self.gen_rgb(video_pose[num],
                                   video_traj_type,
                                   plane_width,
                                   self.fov,
                                   normal_canvas,
                                   anchor,
                                   self.border)
            frames_dict[video_traj_type] = img_gen
            # write_file = os.path.join(output_dir,str(num)+'-'+video_traj_type+'.jpg')
            # cv2.imwrite(write_file, cv2.cvtColor(img_gen, cv2.COLOR_RGB2BGR))
           # cv2.imwrite(os.path.join(output_dir, 'crop_stereo'+str(num)+video_traj_type+'.jpg'), cv2.cvtColor((stereo[atop:abuttom, aleft:aright, :3] * 1).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return frames_dict

    def gen_rgb(self, tp, video_traj_type, plane_width, fov, normal_canvas, anchor, border):
        rel_pose = np.linalg.inv(np.dot(tp, np.linalg.inv(self.ref_pose)))
        axis, angle = transforms3d.axangles.mat2axangle(rel_pose[0:3, 0:3])
        normal_canvas.rotate(axis=axis, angle=(angle*180)/np.pi)
        normal_canvas.translate(rel_pose[:3,3])
        new_mean_loc_depth = self.mean_loc_depth - float(rel_pose[2, 3])
        if 'dolly' in video_traj_type:
            new_fov = float((np.arctan2(plane_width, np.array([np.abs(new_mean_loc_depth)])) * 180. / np.pi) * 2)
            normal_canvas.reinit_camera(new_fov)
        else:
            normal_canvas.reinit_camera(fov)
        normal_canvas.view_changed()
        img = normal_canvas.render()
        img = cv2.GaussianBlur(img,(int(self.init_factor//2 * 2 + 1), int(self.init_factor//2 * 2 + 1)), 0)
        img = cv2.resize(img, (int(img.shape[1] / self.init_factor), int(img.shape[0] / self.init_factor)), interpolation=cv2.INTER_AREA)
        img = img[anchor[0]:anchor[1], anchor[2]:anchor[3]]
        img = img[int(border[0]):int(border[1]), int(border[2]):int(border[3])]

        if any(np.array(self.config['crop_border']) > 0.0):
            H_c, W_c, _ = img.shape
            o_t = int(H_c * self.config['crop_border'][0])
            o_l = int(W_c * self.config['crop_border'][1])
            o_b = int(H_c * self.config['crop_border'][2])
            o_r = int(W_c * self.config['crop_border'][3])
            img = img[o_t:H_c-o_b, o_l:W_c-o_r]
            img = cv2.resize(img, (W_c, H_c), interpolation=cv2.INTER_CUBIC)
            '''
            below is the final constructed image, given the
            ldi and camera coords/angle
            '''
        normal_canvas.translate(-rel_pose[:3,3])
        normal_canvas.rotate(axis=axis, angle=-(angle*180)/np.pi)
        normal_canvas.view_changed()
        crop = False
        if crop:
            stereo = img[..., :3]
            atop = 0; abuttom = img.shape[0] - img.shape[0] % 2; aleft = 0; aright = img.shape[1] - img.shape[1] % 2
            cropped = (stereo[atop:abuttom, aleft:aright, :3] * 1).astype(np.uint8)
        # print('saving frame')
        # cv2.imwrite('tmp/frame'+str(self.im_count)+'.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.im_count += 1
        return img
