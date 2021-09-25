#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from reprojection import depth2pts
from math_tools import v2T


    
class esm:
    def __init__(self, ref_depth, tar_depth):
        self.H, self.W = ref_depth.shape
        self.K = np.array([[718.,0, 607], [0, 718, 185], [0,0,1]])

        self.ref_depth = ref_depth
        self.tar_depth = tar_depth
        pts = depth2pts(tar_depth, self.K)
        self.tar_pts = np.ones((4, pts.shape[1]))
        self.tar_pts[0:3, :] = pts
        self.T = v2T([2,-2,1.3, 0.1, 0.4, -0.3])
        #self.T[1,3] = 0.2
        #self.T[0,3] = 0.2
        #self.T[1,3] = 4
        #self.T[2,3] = 4
        self.last_err = np.inf
        iter = 0
        self.calc_Jg()
        pix =  np.dot(self.K, self.tar_pts[0:3,:])
        pix /= pix[2,:]
        pix = pix[0:2,:]
        pix = pix.T.reshape(-1)

        while(True):
            pts_trans = np.dot(self.T, self.tar_pts)
            reproj_pix =  np.dot(self.K, pts_trans[0:3,:])
            reproj_pix /= reproj_pix[2,:]
            reproj_pix = reproj_pix[0:2,:]
            reproj_pix = reproj_pix.T.reshape(-1)

            self.calc_Jw(pts_trans)
            residuals = reproj_pix - pix
            err = np.sqrt(np.nansum(residuals*residuals)/(residuals.shape[0]))
            #err, residuals = self.residuals(self.ref_depth, reproj_pix, d)
            print("iter:%d, err:%f"%(iter,err))
            iter+=1
            if  err < 0.01:
                print("OK!")
                break

            JwJg = np.matmul(self.Jw, self.Jg)

            J = JwJg

            #self.last_err = err

            J = J.reshape(-1,6)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T, residuals)
            dx = np.dot(hessian_inv,temp)
            #self.T[0,3] += dx[0]
            #self.T[1,3] += dx[1]
            #self.T[2,3] += dx[2]

            #print(self.T)
            dT = getMatrix(dx)
            #print(self.T)
            self.T = np.dot(self.T, dT) 


    def calc_Jg(self):
        A1 = np.array([0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0. ]).reshape([3,4])
        A2 = np.array([0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 0. ]).reshape([3,4])
        A3 = np.array([0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 1. ]).reshape([3,4])
        A4 = np.array([0, 0, 0, 0,  0, 0,-1, 0,  0, 1, 0, 0. ]).reshape([3,4])
        A5 = np.array([0, 0, 1, 0,  0, 0, 0, 0, -1, 0, 0, 0. ]).reshape([3,4])
        A6 = np.array([0,-1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0. ]).reshape([3,4])
        self.Jg = np.vstack([A1.flatten(),
            A2.flatten(),
            A3.flatten(),
            A4.flatten(),
            A5.flatten(),
            A6.flatten()]).T


    def residuals(self, depth, pix, d):
        residuals = depth[pix[1], pix[0]] - d
        residuals = np.nan_to_num(residuals)
        m = np.nansum(residuals*residuals)
        return np.sqrt(m/(d.shape[0])), residuals

    def image_gradient(self, pix):
        dx = (self.ref_depth[pix[1], pix[0] + 1] - self.ref_depth[pix[1], pix[0]]).reshape(-1,1,1)
        dy = (self.ref_depth[pix[1] + 1, pix[0]] - self.ref_depth[pix[1], pix[0]]).reshape(-1,1,1)
        return np.dstack([dx,dy])


    def calc_Jw(self, pts):
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]
        z2 = z*z
        xy = x*y
        x2 = x*x
        y2 = y*y
        fx = 718.
        fy = 718.

        self.Jw = np.zeros([self.tar_pts.shape[1], 2, 12])
        self.Jw[:,0, 0]  = fx*x/z
        self.Jw[:,0, 1]  = fx*y/z
        self.Jw[:,0, 2]  = fx
        self.Jw[:,0, 3]  = fx/z
        self.Jw[:,0, 8]  = -fx*x2/z2
        self.Jw[:,0, 9]  = -fx*xy/z2
        self.Jw[:,0,10] = -fx*x/z
        self.Jw[:,0,11] = -fx*x/z2

        self.Jw[:,1, 4] = fy*x/z
        self.Jw[:,1, 5] = fy*y/z
        self.Jw[:,1, 6] = fy
        self.Jw[:,1, 7] = fy/z
        self.Jw[:,1, 8] = -fy*xy/z2
        self.Jw[:,1, 9] = -fy*y2/z2
        self.Jw[:,1,10] = -fy*y/z
        self.Jw[:,1,11] = -fy*y/z2
        

    
if __name__ == "__main__":
    ref_depth = np.load('/Users/liuyang/workspace/VisualLidar/depth/0000.npy')
    tar_depth = np.load('/Users/liuyang/workspace/VisualLidar/depth/0000.npy')
    esm = esm(ref_depth, tar_depth) #x,y,weight,height
    #esm.track(tar_depth, False)
    plt.show()