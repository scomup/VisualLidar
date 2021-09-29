#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from common import *
import PIL
from pix_select import PixSelect
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

class PhotomericError:
    def __init__(self, ref_pts, tar_img, K, show_log = False):
        self.tar_img = tar_img
        self.ref_pts = ref_pts
        self.H, self.W = tar_img.shape
        self.K = K
        self.T = v2T([0,0,0,0,0,0])
        self.dTdx = self.calc_dTdx()
        self.border = 2
        self.show_log = show_log
        

    def setT(self, T):
        self.T = T

    def track(self, max_err = 255):
        self.max_err = max_err
        tar_img = self.tar_img
        ref_pts = self.ref_pts
        K = self.K
        W = self.W
        H = self.H

        last_err = np.inf
        iter = 0
        while(True):
            #transform the reference points to target coordinate
            cur_pts = transform(self.T, ref_pts[0:3,:])
            #Projection points to camera
            reproj_pix =  projection(K, cur_pts)
            #Make sure all points are located inside the camera boundaries
            check = range_check(reproj_pix, H, W, self.border)
            cur_pts = cur_pts[:,check]
            reproj_pix = reproj_pix[:,check]
            ref_intensity = ref_pts[3,check]

            #Calcate the residuals
            err, residuals = self.residuals(tar_img, reproj_pix, ref_intensity)

            #Calcate the partial derivative
            dCdT = self.calc_dCdT(K, cur_pts)
            if(self.show_log):
                print("iter:%d, err:%f"%(iter,err))
            iter+=1
            if  err < 0.01:
                break
            if last_err - err < 0.001:
                self.T = self.last_T
                self.img0, self.img1 = self.get_proj_err_image(self.tar_img, reproj_pix, ref_intensity)
                #self.img0, self.img1 = reprojection_error_image(tar_depth, ref_depth, self.T, K)
                break

            #Calcate the jacobian
            dDdC = self.calc_dDdC(tar_img, reproj_pix)
            dCdx = np.matmul(dCdT, self.dTdx)
            dDdx = np.matmul(dDdC, dCdx)
            J = dDdx

            residuals = residuals
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

    def residuals(self, img, pix, intensity):
        residuals = interpn(img, pix) - intensity
        residuals = np.nan_to_num(residuals)
        residuals = np.clip(residuals, -self.max_err, self.max_err)
        r = np.abs(residuals)
        e = np.nansum(r)
        return e/(residuals.shape[0]), residuals

    def get_proj_err_image(self, img, pix, intensity):
        residuals = interpn(img, pix) - intensity
        residuals = np.nan_to_num(residuals)
        r = np.abs(residuals)
        error_image = np.zeros_like(img).astype(float)
        reproj_image = np.zeros_like(img).astype(float)
        #reproj_image.fill(np.nan)
        #error_image.fill(np.nan)
        pix = np.around(pix)
        pix = pix.astype(int)
        error_image[pix[1], pix[0]] = r
        reproj_image[pix[1], pix[0]] = intensity
        return error_image, reproj_image

    def calc_dDdC(self, img, pix):
        pix_x1y0 = pix.copy()
        pix_x1y0[0] = pix_x1y0[0] + 1.
        pix_x0y1 = pix.copy()
        pix_x0y1[1] = pix_x0y1[1] + 1.
        x0y0 = interpn(img, pix)
        x1y0 = interpn(img, pix_x1y0)
        x0y1 = interpn(img, pix_x0y1)
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
    fx = 721.538
    fy = 721.538
    cx = 609.559
    cy = 172.854
    K = np.array([[fx,0, cx], [0, fy, cy], [0,0,1]])
    ref_depth = np.load('/home/liu/workspace/VisualLidar/depth/0000.npy')
    ref_img = np.asarray(PIL.Image.open('/media/liu/SSD-PSMU3/kitti_slam/00/image_2/000000.png').convert('L'))
    tar_img = np.asarray(PIL.Image.open('/media/liu/SSD-PSMU3/kitti_slam/00/image_2/000001.png').convert('L'))
    pix = PixSelect(ref_img)
    ref_pts = getpts(ref_depth, ref_img, pix, K ,1)
    ref_pts = depth2ptsI(ref_depth, ref_img, K ,2)

    matcher = PhotomericError(ref_pts, tar_img, K, True) 
    matcher.setT(v2T([0,0,0,0,0,0]))
    matcher.track()
    matcher.track(10)

    ref_pts = depth2ptsI(ref_depth, ref_img, K ,1)
    cur_pts = transform(matcher.T, ref_pts[0:3,:])
    reproj_pix =  projection(K, cur_pts)
    check = range_check(reproj_pix, tar_img.shape[0], tar_img.shape[1], 1)
    cur_pts = cur_pts[:,check]
    reproj_pix = reproj_pix[:,check]
    ref_intensity = ref_pts[3,check]
    img0, img1 = matcher.get_proj_err_image(matcher.tar_img, reproj_pix, ref_intensity)

    fig, axes= plt.subplots(3)
    axes[0].imshow(img0, vmin=0, vmax=30)
    axes[0].set_title('reproj error',loc='left')
    axes[1].imshow(img1, vmin=0, vmax=255)
    axes[1].set_title('ref reproj to tar depth',loc='left')
    axes[2].imshow(matcher.tar_img, vmin=0, vmax=255)
    axes[2].set_title('tar depth',loc='left')

    print(matcher.T)

    if(False):
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