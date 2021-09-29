from line_profiler import LineProfiler

import numpy as np
from common import *
from optimizer import DepthError
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == "__main__":
    tar_depth = np.load('/home/liu/workspace/VisualLidar/depth/0000.npy')
    ref_depth = np.load('/home/liu/workspace/VisualLidar/depth/0001.npy')
    fx = 721.538
    fy = 721.538
    cx = 609.559
    cy = 172.854
    K = np.array([[fx,0, cx], [0, fy, cy], [0,0,1]])
    matcher = DepthError(tar_depth, ref_depth, K, True) 
    matcher.setT(v2T([0,0,-1,0,0,0]))
    #matcher.track(max_err=3, sampling=1)


    prof = LineProfiler()
    prof.add_function(matcher.track)
    prof.runcall(matcher.track)
    prof.print_stats()