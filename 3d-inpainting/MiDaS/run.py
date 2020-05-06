"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
# from monodepth_net import MonoDepthNet
# import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio


def run_depth(img_name, input_path, output_path, model_path, Net, utils, target_w=None, device='cpu'):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    # select device
    device = torch.device(device)

    # load network
    model = Net(model_path)
    model.to(device)
    model.eval()

    img = utils.read_image(img_name)
    w = img.shape[1]
    scale = target_w / max(img.shape[0], img.shape[1])
    target_height = int(round(img.shape[0] * scale))
    target_width = int(round(img.shape[1] * scale))
    img_input = utils.resize_image(img)
    img_input = img_input.to(device)
    # compute
    with torch.no_grad():
        out = model.forward(img_input)

    depth = utils.resize_depth(out, target_width, target_height)
    img = cv2.resize((img * 255).astype(np.uint8),
                     (target_width, target_height),
                     interpolation=cv2.INTER_AREA)
    return depth


# if __name__ == "__main__":
#     # set paths
#     INPUT_PATH = "image"
#     OUTPUT_PATH = "output"
#     MODEL_PATH = "model.pt"

#     # set torch options
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True

#     # compute depth maps
#     run_depth(INPUT_PATH, OUTPUT_PATH, MODEL_PATH, Net, target_w=640)
