#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from common import *
import PIL
import scipy.signal


    
def PixSelect(img, want = 30, block_size = 32):
    dx = np.roll(img,-1,axis=1) - img
    dy = np.roll(img,-1,axis=0) - img
    dx[:,-1] = 0.
    dx[-1,:] = 0.
    dy[:,-1] = 0.
    dy[-1,:] = 0.
    grad =  np.sqrt((dx * dx + dy * dy))
    pt_list = []
    h, w = img.shape
    bw = w//block_size
    bh = h//block_size
    ths = np.zeros([bh, bw])
    blocks = []
    for x in range(bh):
        for y in range(bw):
            xs = block_size*x
            xe = block_size*(x+1)
            ys = block_size*y
            ye = block_size*(y+1)
            if(x == bh - 1):
                xe = h
            if(y == bw - 1):
                ye = w
            blocks.append([xs , xe, ys , ye , x, y])
    for b in blocks:
        xs , xe, ys , ye , x, y = b
        block_grad = grad[xs : xe, ys : ye]
        ths[x, y] = np.median(block_grad)
    kernel = np.ones([3,3])/9.
    ths_smooth = scipy.signal.fftconvolve(ths, kernel, mode='same')
    for b in blocks:
        xs , xe, ys , ye , x, y = b
        th = ths_smooth[ x, y]
        block_grad = grad[xs : xe, ys : ye]
        s = 1.
        while(True):
            N = np.sum(block_grad > s * th)
            if(N >  want):
                s += 0.5
            else:
                s -= 0.5
                break
        x, y = np.where(block_grad > s * th)
        #print(np.sum(block_grad > s * th))
        x += xs
        y += ys


        pt_list.append(np.vstack([y,x]))
    pts = np.hstack(pt_list)
    return pts


if __name__ == "__main__":
    img = PIL.Image.open('/media/liu/SSD-PSMU3/kitti_slam/00/image_2/000000.png').convert('L')
    pts = PixSelect(np.asarray(img))
    img = np.asarray(img).astype(float)
    img[pts[1,:], pts[0,:]] = np.nan
    plt.imshow(img)
    plt.show()
    
