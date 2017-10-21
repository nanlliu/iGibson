import numpy as np
import ctypes as ct
import cv2
import sys
import argparse
from realenv.data.datasets import ViewDataSet3D
from completion import CompletionNet
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from numpy import cos, sin
import utils
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from multiprocessing import Pool, cpu_count
from scipy.signal import convolve2d
from scipy.interpolate import griddata
import scipy
import torch.nn.functional as F
from torchvision import transforms


dll=np.ctypeslib.load_library('../core/render/render_cuda_f','.')

# In[6]:

def render(imgs, depths, pose, poses, tdepth):
    global fps
    t0 = time.time()
    showsz = imgs[0].shape[0]
    nimgs = len(imgs)
    show=np.zeros((nimgs, showsz,showsz * 2,3),dtype='uint8')
    target_depth = tdepth[:,:,0]

    imgs = np.array(imgs)
    depths = np.array(depths)
    
    pose_after = [pose.dot(np.linalg.inv(poses[0])).dot(poses[i]).astype(np.float32) for i in range(len(imgs))]
    pose_after = np.array(pose_after)
    
    dll.render(ct.c_int(len(imgs)),
               ct.c_int(imgs[i].shape[0]),
               ct.c_int(imgs[i].shape[1]),
               ct.c_int(1),
               imgs.ctypes.data_as(ct.c_void_p),
               depths.ctypes.data_as(ct.c_void_p),
               pose_after.ctypes.data_as(ct.c_void_p),
               show.ctypes.data_as(ct.c_void_p),
               target_depth.ctypes.data_as(ct.c_void_p)
              )

    return show, target_depth

# In[7]:

def generate_data(args):

    idx  = args[0]
    print(idx)
    d    = args[1]
    outf = args[2]

    print(idx)
    data = d[idx]   ## This operation stalls 95% of the time, CPU heavy
    sources = data[0]
    target = data[1]
    source_depths = data[2]
    target_depth = data[3]
    poses = [item.numpy() for item in data[-1]]
    show, _ =  render(sources, source_depths, poses[0], poses, target_depth)

    return show, target_depth, target



parser = argparse.ArgumentParser()
parser.add_argument('--debug'  , action='store_true', help='debug mode')
parser.add_argument('--dataroot'  , required = True, help='dataset path')
parser.add_argument('--outf'  , type = str, default = '', help='path of output folder')
opt = parser.parse_args()


d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 5, off_3d = False, train = False)

for i in range(len(d)):

    filename = "%s/data_%d.npz" % (opt.outf, i)
    if not os.path.isfile(filename):
        generate_data([i, d, opt.outf])


