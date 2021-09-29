#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from common import *

"""
The cost function is defined as
fi = Dt(C(x)pt_i) - tz(x)pr_i
where:
pt_i: The point i observed by ref camera
C(x) = K*T(x): Camera projection matrix
K: intrinsics matrix
T(x): Transform matrix for reference to target camera
tz(x): the z(3rd) row of T
x: 6DoF transform parameter
Dt: The depth image from target camera
"""

class DepthError:
    def __init__(self, ref_depth, tar_depth, K, show_log = False):
        self.tar_depth = tar_depth
        self.ref_depth = ref_depth
        self.H, self.W = tar_depth.shape
        self.K = K
        self.T = v2T([0,0,0,0,0,0])
        self.dTdx = self.calc_dTdx()
        self.border = 2
        self.show_log = show_log

    def setT(self, T):
        self.T = T

    def track(self, max_err=np.inf, sampling = 1, remove_outlier = False):
        
        tar_depth = self.tar_depth
        ref_depth = self.ref_depth
        K = self.K
        W = self.W
        H = self.H

        ref_pts = depth2pts(ref_depth, K ,sampling)
        num = 50000
        if(num != 0 and ref_pts.shape[1] > num):
            ref_pts = ref_pts[:,np.random.choice(ref_pts.shape[1],num,replace=False)]
        self.max_err = max_err
        last_err = np.inf
        iter = 0
        while(True):
            #transform the reference points to target coordinate
            cur_pts = transform(self.T, ref_pts)
            #Projection points to camera
            reproj_pix =  projection(K, cur_pts)
            #Make sure all points are located inside the camera boundaries
            check = range_check(reproj_pix, H, W, self.border)
            cur_pts = cur_pts[:,check]
            reproj_pix = reproj_pix[:,check]
            reproj_d = cur_pts[2]

            #Calcate the residuals
            err, residuals = self.residuals(tar_depth, reproj_pix, reproj_d)

            #check inlier
            if(remove_outlier):
                mask = np.where(np.abs(residuals) < self.max_err)[0]
                cur_pts = cur_pts[:,mask]
                reproj_pix = reproj_pix[:,mask]
                reproj_d = cur_pts[2]
                err, residuals = self.residuals(tar_depth, reproj_pix, reproj_d)

            #Calcate the partial derivative
            dCdT = self.calc_dCdT(K, cur_pts)
            dtzdT = self.calc_dtzdT(cur_pts)
            if(self.show_log):
                print("iter:%d, err:%f"%(iter,err))
            iter+=1
            if  err < 0.01:
                break
            if err > last_err:
                self.T = self.last_T
                #_, _, self.img0, self.img1 = self.get_proj_err_image(self.tar_depth, reproj_pix, reproj_d)
                self.img0, self.img1 = reprojection_error_image(tar_depth, ref_depth, self.T, K)
                break

            #Calcate the jacobian
            dDdC = self.calc_dDdC(tar_depth, reproj_pix)
            dCdx = np.matmul(dCdT, self.dTdx)
            dDdx = np.matmul(dDdC, dCdx)
            dtzdx = np.matmul(dtzdT, self.dTdx)
            J = dDdx - dtzdx
            #J = dDdx
            residuals = residuals * sampling
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

    def residuals(self, depth, pix, reproj_d):
        residuals = interpn(depth, pix) - reproj_d
        residuals = np.nan_to_num(residuals)
        residuals = np.clip(residuals, -self.max_err, self.max_err)
        r = np.abs(residuals)
        e = np.nansum(r)
        return e/(residuals.shape[0]), residuals

    def get_proj_err_image(self, depth, pix, reproj_d):
        residuals = interpn(depth, pix) - reproj_d
        residuals = np.nan_to_num(residuals)
        residuals = np.clip(residuals, -self.max_err, self.max_err)
        r = np.abs(residuals)
        error_image = np.zeros_like(depth)
        reproj_image = np.zeros_like(depth)
        reproj_image.fill(np.nan)
        error_image.fill(np.nan)
        error_image[pix[1], pix[0]] = r
        reproj_image[pix[1], pix[0]] = reproj_d
        return error_image, reproj_image

    def get_good_pts(self, color, sampling = 1):
        ref_pts = depth2pts(self.ref_depth, self.K, sampling)
        cur_pts = transform(self.T, ref_pts)
        pix =  projection(self.K, cur_pts)
        check = range_check(pix, self.H, self.W)
        cur_pts = cur_pts[:,check]
        reproj_d = cur_pts[2]
        pix = pix[:,check]
        _, residuals, = self.residuals(self.tar_depth, pix, reproj_d)
        mask = np.where(np.abs(residuals) < 0.3)[0]
        pts = ref_pts[:,check]
        pts = pts[:, mask]
        pix = pix[:,mask]
        pix = np.around(pix)
        pix = pix.astype(int)
        c = color[pix[1],pix[0]].T
        pts = np.vstack([pts, c])
        return pts



    def calc_dDdC(self, depth, pix):
        pix_x1y0 = pix.copy()
        pix_x1y0[0] = pix_x1y0[0] + 1.
        pix_x0y1 = pix.copy()
        pix_x0y1[1] = pix_x0y1[1] + 1.
        x0y0 = interpn(depth, pix)
        x1y0 = interpn(depth, pix_x1y0)
        x0y1 = interpn(depth, pix_x0y1)
        dx = x1y0 - x0y0
        dy = x0y1 - x0y0
        dDdC = np.nan_to_num(np.dstack([dx.reshape(-1,1),dy.reshape(-1,1)]))
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
    tar_depth = np.load('/home/liu/workspace/VisualLidar/depth/0000.npy')
    ref_depth = np.load('/home/liu/workspace/VisualLidar/depth/0001.npy')
    
    fx = 721.538
    fy = 721.538
    cx = 609.559
    cy = 172.854
    K = np.array([[fx,0, cx], [0, fy, cy], [0,0,1]])

    map_pts = depth2pts(ref_depth, K ,4)

    matcher = DepthError(tar_depth, ref_depth, K, True) 
    matcher.setT(v2T([0,0,0,0,0,0]))
    matcher.track(max_err=5, sampling=4)
    matcher.track(max_err=1, sampling=1, remove_outlier=True)
    fig, axes= plt.subplots(3)
    axes[0].imshow(matcher.img0, vmin=0, vmax=3)
    axes[0].set_title('reproj error',loc='left')
    axes[1].imshow(matcher.img1, vmin=0, vmax=30)
    axes[1].set_title('ref reproj to tar depth',loc='left')
    axes[2].imshow(matcher.tar_depth, vmin=0, vmax=30)
    axes[2].set_title('tar depth',loc='left')

    print(matcher.T)

    if(True):
        import open3d as o3d
        pcd0 = o3d.geometry.PointCloud()
        tar_pts = depth2pts(tar_depth, K)
        pcd0.points = o3d.utility.Vector3dVector(tar_pts.T)
        c = np.zeros_like(tar_pts.T)
        c[:,2] = 1
        pcd0.colors = o3d.utility.Vector3dVector(c)
        pcd1 = o3d.geometry.PointCloud()
        ref_pts = depth2pts(ref_depth, K)
        ref_pts = transform(np.linalg.inv(matcher.T), ref_pts)
        pcd1.points = o3d.utility.Vector3dVector(ref_pts.T)
        c = np.zeros_like(ref_pts.T)
        c[:,1] = 1
        pcd1.colors = o3d.utility.Vector3dVector(c)
        o3d.visualization.draw_geometries([pcd0, pcd1])
    

    plt.show()