#!/usr/bin/env python3

import numpy as np
from common import *
from depth_error import DepthError
import matplotlib.pyplot as plt
from PIL import Image


import glob, os
depth_fn = []
path = '/home/liu/workspace/VisualLidar/depth/'
os.chdir(path)
for file in glob.glob("*.npy"):
    depth_fn.append(path + file)
depth_fn.sort()

path = '/media/liu/SSD-PSMU3/kitti_slam/00/'
left_fn = []
os.chdir(path + 'image_2/')
for file in glob.glob("*.png"):
    left_fn.append(path + 'image_2/' + file)
left_fn.sort()


if __name__ == "__main__":
    fx = 721.538
    fy = 721.538
    cx = 609.559
    cy = 172.854
    K = np.array([[fx,0, cx], [0, fy, cy], [0,0,1]])
    T = v2T([0,0,0,0,0,0])
    fig, axes= plt.subplots(3, figsize=(8.0, 10.0))
    for i in range(1, len(depth_fn)):
        print('process image: %d'%i)
        ref_depth = np.load(depth_fn[i-1])
        tar_depth = np.load(depth_fn[i])
        left_img = np.ascontiguousarray(Image.open(left_fn[i]))

        matcher = DepthError(ref_depth, tar_depth, K) 
        matcher.setT(T)
        matcher.track(max_err=5, sampling=4)
        matcher.track(max_err=1, sampling=1, remove_outlier=True)
        T = matcher.T
        pts = matcher.get_good_pts(left_img)
        np.save('/home/liu/workspace/VisualLidar/mapping/T/T_%04d'%i, matcher.T)
        np.save('/home/liu/workspace/VisualLidar/mapping/pts/pts_%04d'%i, pts)
        if(True):
            axes[0].cla()
            axes[1].cla()
            axes[2].cla()
            axes[0].imshow(matcher.img0, vmin=0, vmax=3)
            axes[0].set_title('reproj error',loc='left')
            axes[1].imshow(matcher.img1, vmin=0, vmax=30)
            axes[1].set_title('ref reproj to tar depth',loc='left')
            axes[2].imshow(matcher.tar_depth, vmin=0, vmax=30)
            axes[2].set_title('tar depth',loc='left')
            plt.savefig('/home/liu/workspace/VisualLidar/mapping/img/%04d.png'%i)
            plt.pause(0.01)
    