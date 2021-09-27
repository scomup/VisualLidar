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
    def __init__(self, ref_depth, tar_depth, K):
        self.ref_depth = ref_depth
        self.tar_depth = tar_depth
        self.H, self.W = ref_depth.shape
        self.K = K
        self.T = v2T([0,0,0,0,0,0])
        self.dTdx = self.calc_dTdx()
        self.border = 2

    def track(self, max_err=np.inf, sampling = 1, remove_outlier = False):

        ref_depth = self.ref_depth
        tar_depth = self.tar_depth
        K = self.K
        W = self.W
        H = self.H

        tar_pts = depth2pts(tar_depth, K ,sampling)
        #tar_pts = tar_pts[:,np.random.choice(tar_pts.shape[1],100000,replace=False)]
        self.max_err = max_err
        last_err = np.inf
        iter = 0
        while(True):
            #transform the target points to reference coordinate
            cur_pts = transform(self.T, tar_pts)
            #Projection points to camera
            reproj_pix =  projection(K, cur_pts)
            #Make sure all points are located inside the camera boundaries
            check = range_check(reproj_pix, H, W, self.border)
            cur_pts = cur_pts[:,check]
            reproj_pix = reproj_pix[:,check]
            reproj_d = cur_pts[2]
            #check inlier
            if(remove_outlier):
                mask = self.inlier(ref_depth, reproj_pix, reproj_d)
                cur_pts = cur_pts[:,mask]
                reproj_pix = reproj_pix[:,mask]
                reproj_d = cur_pts[2]

            #Calcate the partial derivative
            dCdT = self.calc_dCdT(K, cur_pts)
            dtzdT = self.calc_dtzdT(cur_pts)
            #Calcate the residuals
            err, residuals, _, _ = self.residuals(ref_depth, reproj_pix, reproj_d)
            
            print("iter:%d, err:%f"%(iter,err))
            iter+=1
            if  err < 0.01:
                print("OK!")
                print(self.T)
                break
            if err > last_err:
                self.T = self.last_T
                #_, _, self.img0, self.img1 = self.residuals(self.ref_depth, reproj_pix, reproj_d, True)
                self.img0, self.img1 = reprojection_error_image(ref_depth, tar_depth, self.T, K)
                break

            #Calcate the jacobian
            dDdC = self.calc_dDdC(ref_depth, reproj_pix)
            dCdx = np.matmul(dCdT, self.dTdx)
            dDdx = np.matmul(dDdC, dCdx)
            dtzdx = np.matmul(dtzdT, self.dTdx)
            J = dDdx - dtzdx

            #Gauss-nowton method
            J = J.reshape(-1,6)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T, residuals)
            dx = np.dot(hessian_inv,temp)
            dT = v2T(dx)
            self.last_T = self.T
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

    def residuals(self, depth, pix, reproj_d, get_proj_err_image = False):
        residuals = depth[pix[1], pix[0]] - reproj_d
        residuals = np.nan_to_num(residuals)
        residuals = np.clip(residuals, -self.max_err, self.max_err)
        r = np.abs(residuals)
        e = np.nansum(r)
        error_image = np.zeros_like(depth)
        reproj_image = np.zeros_like(depth)
        if(get_proj_err_image):
            reproj_image.fill(np.nan)
            error_image.fill(np.nan)
            error_image[pix[1], pix[0]] = r
            reproj_image[pix[1], pix[0]] = reproj_d
        return e/(residuals.shape[0]), residuals, error_image, reproj_image

    def inlier(self, depth, pix, reproj_d):
        residuals = depth[pix[1], pix[0]] - reproj_d
        residuals = np.nan_to_num(residuals)
        residuals = np.clip(residuals, -self.max_err, self.max_err)
        r = np.abs(residuals)
        return np.where(r < self.max_err)[0]


    def calc_dDdC(self, depth, pix):
        dx = (depth[pix[1], pix[0] + 1] - depth[pix[1], pix[0]]).reshape(-1,1,1)
        dy = (depth[pix[1] + 1, pix[0]] - depth[pix[1], pix[0]]).reshape(-1,1,1)
        dDdC = np.nan_to_num(np.dstack([dx,dy]))
        return dDdC

    def calc_dCdT(self, K, pts):
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]
        z2 = z*z
        xy = x*y
        x2 = x*x
        y2 = y*y
        fx = K[0,0]
        fy = K[1,1]
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
    ref_depth = np.load('/home/liu/workspace/VisualLidar/depth/0000.npy')
    tar_depth = np.load('/home/liu/workspace/VisualLidar/depth/0001.npy')
    fx = 721.538
    fy = 721.538
    cx = 609.559
    cy = 172.854
    K = np.array([[fx,0, cx], [0, fy, cy], [0,0,1]])
    matcher = DepthMatcher(ref_depth, tar_depth, K) 
    matcher.track(max_err=3, sampling=4)
    matcher.track(max_err=1, sampling=2, remove_outlier=True)
    fig, axes= plt.subplots(3)
    axes[0].imshow(matcher.img0, vmin=0, vmax=3)
    axes[0].set_title('reproj error',loc='left')
    axes[1].imshow(matcher.img1, vmin=0, vmax=30)
    axes[1].set_title('tar reproj to ref depth',loc='left')
    axes[2].imshow(matcher.ref_depth, vmin=0, vmax=30)
    axes[2].set_title('truth ref depth',loc='left')

    print(matcher.T)

    if(True):
        import open3d as o3d
        pcd0 = o3d.geometry.PointCloud()
        ref_pts = depth2pts(ref_depth, K)
        pcd0.points = o3d.utility.Vector3dVector(ref_pts.T)
        c = np.zeros_like(ref_pts.T)
        c[:,2] = 1
        pcd0.colors = o3d.utility.Vector3dVector(c)
        pcd1 = o3d.geometry.PointCloud()
        tar_pts = depth2pts(tar_depth, K)
        tar_pts = transform(matcher.T, tar_pts)
        pcd1.points = o3d.utility.Vector3dVector(tar_pts.T)
        c = np.zeros_like(tar_pts.T)
        c[:,1] = 1
        pcd1.colors = o3d.utility.Vector3dVector(c)
        o3d.visualization.draw_geometries([pcd0,pcd1])

    plt.show()