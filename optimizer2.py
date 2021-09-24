#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.spatial.transform import Rotation
from reprojection import depth2pts
from scipy.interpolate import griddata

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

epsilon = 1e-5

def getMatrix(x):
    M = np.eye(4)
    M[0:3,3] = x[0:3]
    #M[0:3,0:3] = exp(x[3:6])
    return M
    
# so3 to 3d Rotation Matrix
def exp(v):
    theta_sq = np.dot(v, v)
    imag_factor = 0.
    real_factor = 0.
    if (theta_sq < epsilon * epsilon):
        theta_po4 = theta_sq * theta_sq
        imag_factor = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4
        real_factor = 1. - (1.0 / 8.0) * theta_sq +   (1.0 / 384.0) * theta_po4
    else:
        theta = np.sqrt(theta_sq)
        half_theta = 0.5 * theta
        sin_half_theta = np.sin(half_theta)
        imag_factor = sin_half_theta / theta
        real_factor = np.cos(half_theta)
    quat = np.array([imag_factor*v[0], imag_factor*v[1], imag_factor*v[2], real_factor])
    rot = Rotation.from_quat(quat)
    return rot.as_dcm()

class esm:
    def __init__(self, ref_depth, tar_depth):
        self.H, self.W = ref_depth.shape
        self.K = np.array([[718.,0, 607], [0, 718, 185], [0,0,1]])

        self.ref_depth = ref_depth
        self.tar_depth = tar_depth
        pts = depth2pts(tar_depth, self.K)
        self.tar_pts = np.ones((4, pts.shape[1]))
        self.tar_pts[0:3, :] = pts

        #self.ref_dxdy = self.image_gradient(self.ref_depth)
        self.T = np.eye(4)
        #self.T[0,3] = 0.01
        self.T[2,3] = 0.5
        self.last_err = np.inf

        while(True):
            
            pts_trans = np.dot(self.T, self.tar_pts)
            self.calc_Jw(pts_trans)
            reproj_pix =  np.dot(self.K, pts_trans[0:3,:])
            reproj_pix /= reproj_pix[2,:]
            #reproj_pix[0] = np.around(reproj_pix[0])
            #reproj_pix[1] = np.around(reproj_pix[1])
            #reproj_pix = reproj_pix.astype(int)

            check = np.logical_and(np.logical_and(
                reproj_pix[0] < self.W - 2, reproj_pix[0] > 2),
                np.logical_and(reproj_pix[2] < self.H - 1, reproj_pix[1] > 2))

            d = pts_trans[2, check]
            self.calc_Jw(pts_trans)
            Jw = self.Jw[check, :, :]

            reproj_pix = reproj_pix[0:2, check]

            err, residuals = self.residuals(self.ref_depth, reproj_pix, d)
            print(err)
            if  err < 0.00001:
                print("OK!")
                break

            Ji = self.image_gradient(reproj_pix)
            Ji = np.nan_to_num(Ji)
            JiJw = np.matmul(Ji, Jw)

            J = JiJw - self.Jw2[check, :, :]


            self.last_err = err

            J = J.reshape(-1,3)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T, residuals)
            dx = np.dot(hessian_inv,temp)
            #print(self.T)
            dT = getMatrix(dx)
            self.T = np.dot(self.T, dT) 





    def residuals(self, depth, pix, d):
        residuals = depth[pix[1], pix[0]] - d
        residuals = np.nan_to_num(residuals)
        m = np.nansum(residuals*residuals)
        return np.sqrt(m/(d.shape[0])), residuals

    def image_gradient(self, pix):

        #dx = np.roll(img,-1,axis=1) - img
        dx = (self.ref_depth[pix[1], pix[0] + 1] - self.ref_depth[pix[1], pix[0]]).reshape(-1,1,1)
        dy = (self.ref_depth[pix[1] + 1, pix[0]] - self.ref_depth[pix[1], pix[0]]).reshape(-1,1,1)
        return np.dstack([dx,dy])


    def calc_Jw(self, pts):
        x = pts[0,:]
        y = pts[1,:]
        z = pts[2,:]

        self.K = np.array([[718.,0, 607], [0, 718, 185], [0,0,1]])
        fx = 718.
        fy = 718.
        cx = 607.
        cy = 185.

        self.Jw = np.zeros([self.tar_pts.shape[1], 2, 3])
        self.Jw[:,0,0] = fx/z
        self.Jw[:,0,1] = 0
        self.Jw[:,0,2] = -(cx + fx *x)/(z*z)

        self.Jw[:,1,0] = 0
        self.Jw[:,1,1] = fy/z
        self.Jw[:,1,2] = -(cy + fy *y)/(z*z)

        self.Jw2 = np.zeros([self.tar_pts.shape[1], 1, 3])
        self.Jw2[:,0,0] = x
        self.Jw2[:,0,1] = y
        self.Jw2[:,0,2] = z

    
if __name__ == "__main__":
    ref_depth = np.load('/home/liu/workspace/VisualLidar/depth/0000.npy')
    tar_depth = np.load('/home/liu/workspace/VisualLidar/depth/0000.npy')
    esm = esm(ref_depth, tar_depth) #x,y,weight,height
    #esm.track(tar_depth, False)
    plt.show()