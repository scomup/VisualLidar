#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from common import *

"""
The cost function is defined as
fi = Dr(C(x)pt_i) - tz(x)pt_i
where:
pt_i: The point i observed by camera A
C(x) = K*T(x): Camera projection matrix
K: intrinsics matrix
T(x): Transform matrix for target to reference camera
tz(x): the z(3rd) row of T
x: 6DoF transform parameter
Dr: The depth image from reference camera
"""

class DepthMatcher:
    def __init__(self, ref_depht, tar_depth, K, max_err = 3):
        self.ref_depth = ref_depht
        self.tar_depth = tar_depth
        self.H, self.W = ref_depth.shape
        self.tar_pts = depth2pts(tar_depth, K)
        self.K = K
        self.max_err = max_err
        self.T = v2T([0,0,0,0,0,0])
        self.dTdx = self.calc_dTdx()

    def track(self):
        #self.img0, self.img1 = reprojection_error_image(self.ref_depth, self.tar_depth, self.T, self.K)
        #return

        last_err = np.inf
        iter = 0
        while(True):
            #transform the target points to reference coordinate
            cur_pts = transform(self.T, self.tar_pts)
            #Projection points to camera
            reproj_pix =  projection(self.K, cur_pts)
            #Make sure all points are located inside the camera boundaries
            check = range_check(reproj_pix, self.H, self.W)
            #check = random_mask(check, 1000)
            cur_pts = cur_pts[:,check]
            reproj_pix = reproj_pix[:,check]

            #Calcate the partial derivative
            dCdT = self.calc_dCdT(cur_pts)
            dtzdT = self.calc_dtzdT(cur_pts)
            #Calcate the residuals
            reproj_depth = cur_pts[2]
            err, residuals = self.residuals(self.ref_depth, reproj_pix, reproj_depth)
            
            print("iter:%d, err:%f"%(iter,err))
            iter+=1
            if  err < 0.01:
                print("OK!")
                print(self.T)
                break
            if err > last_err:
                self.img0, self.img1 = reprojection_error_image(self.ref_depth, self.tar_depth, self.T, self.K)
                break

            #Calcate the jacobian
            dDdC = self.calc_dDdC(self.ref_depth, reproj_pix)
            dCdx = np.matmul(dCdT, self.dTdx)
            dDdx = np.matmul(dDdC, dCdx)
            dtzdx = np.matmul(dtzdT, self.dTdx)
            #J = dDdx - dtzdx
            J = dDdx

            #Gauss-nowton method
            J = J.reshape(-1,6)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T, residuals)
            dx = np.dot(hessian_inv,temp)
            dT = v2T(dx)
            self.T = np.dot(self.T, dT) 
            last_err = err

    def calc_dTdx(self):
        A1 = np.array([0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0. ]).reshape([3,4])
        A2 = np.array([0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 0. ]).reshape([3,4])
        A3 = np.array([0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 1. ]).reshape([3,4])
        A4 = np.array([0, 0, 0, 0,  0, 0,-1, 0,  0, 1, 0, 0. ]).reshape([3,4])
        A5 = np.array([0, 0, 1, 0,  0, 0, 0, 0, -1, 0, 0, 0. ]).reshape([3,4])
        A6 = np.array([0,-1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0. ]).reshape([3,4])
        dTdx = np.vstack([A1.flatten(),
            A2.flatten(),
            A3.flatten(),
            A4.flatten(),
            A5.flatten(),
            A6.flatten()]).T
        return dTdx

    def residuals(self, depth, pix, reproj_depth):
        residuals = depth[pix[1], pix[0]] - reproj_depth
        residuals = np.nan_to_num(residuals)
        residuals = np.clip(residuals, -self.max_err, self.max_err)
        m = np.nansum(residuals*residuals)
        return np.sqrt(m/(reproj_depth.shape[0])), residuals

    def calc_dDdC(self, depth, pix):
        dx = (depth[pix[1], pix[0] + 1] - depth[pix[1], pix[0]]).reshape(-1,1,1)
        dy = (depth[pix[1] + 1, pix[0]] - depth[pix[1], pix[0]]).reshape(-1,1,1)
        dDdC = np.nan_to_num(np.dstack([dx,dy]))
        return dDdC

    def calc_dCdT(self, pts):
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]
        z2 = z*z
        xy = x*y
        x2 = x*x
        y2 = y*y
        fx = self.K[0,0]
        fy = self.K[1,1]
        dCdT = np.zeros([pts.shape[1], 2, 12])
        dCdT[:,0, 0]  = fx*x/z
        dCdT[:,0, 1]  = fx*y/z
        dCdT[:,0, 2]  = fx
        dCdT[:,0, 3]  = fx/z
        dCdT[:,0, 8]  = -fx*x2/z2
        dCdT[:,0, 9]  = -fx*xy/z2
        dCdT[:,0,10] = -fx*x/z
        dCdT[:,0,11] = -fx*x/z2
        dCdT[:,1, 4] = fy*x/z
        dCdT[:,1, 5] = fy*y/z
        dCdT[:,1, 6] = fy
        dCdT[:,1, 7] = fy/z
        dCdT[:,1, 8] = -fy*xy/z2
        dCdT[:,1, 9] = -fy*y2/z2
        dCdT[:,1,10] = -fy*y/z
        dCdT[:,1,11] = -fy*y/z2
        return dCdT

    def calc_dtzdT(self, pts):
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]
        dtzdT = np.zeros([pts.shape[1], 1, 12])
        dtzdT[:,0, 8] = x
        dtzdT[:,0, 9] = y
        dtzdT[:,0,10] = z
        dtzdT[:,0,11] = 1
        return dtzdT

    
if __name__ == "__main__":
    ref_depth = np.load('/Users/liuyang/workspace/VisualLidar/depth/0000.npy')
    tar_depth = np.load('/Users/liuyang/workspace/VisualLidar/depth/0001.npy')
    K = np.array([[718.,0, 607], [0, 718, 185], [0,0,1]])
    max_err = 2
    matcher = DepthMatcher(ref_depth, tar_depth, K, max_err) 
    matcher.track()
    fig, axes= plt.subplots(4)
    axes[0].imshow(matcher.img0, vmin=0, vmax=max_err)
    axes[0].set_title('reproj error',loc='left')
    axes[1].imshow(matcher.img1, vmin=0, vmax=30)
    axes[1].set_title('tar reproj to ref depth')
    axes[2].imshow(ref_depth, vmin=0, vmax=30)
    axes[2].set_title('ref depth')
    axes[3].imshow(tar_depth, vmin=0, vmax=30)
    axes[3].set_title('tar depth')
    print(matcher.T)
    plt.show()