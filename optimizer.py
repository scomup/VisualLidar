#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys

from numpy.lib.arraysetops import ediff1d
from reprojection import depth2pts, reprojection
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass


class esm:
    def __init__(self, ref_depth, tar_depth):
        self.H, self.W = ref_depth.shape
        self.K = np.array([[718.,0, 607], [0, 718, 185], [0,0,1]])

        self.ref_depth = ref_depth
        self.tar_depth = tar_depth
        pts = depth2pts(tar_depth, self.K)
        self.tar_pts = np.ones((4, pts.shape[1]))
        self.tar_pts[0:3, :] = pts

        self.calc_Jg()
        self.calc_Jw()
        #self.ref_dxdy = self.image_gradient(self.ref_depth)
        self.T = np.eye(4)
        self.last_err = np.inf

        while(True):


            pts_trans = np.dot(self.T, self.tar_pts)
            reproj_pix =  np.dot(self.K, pts_trans[0:3,:])
            reproj_pix /= reproj_pix[2,:]
            reproj_pix[0] = np.around(reproj_pix[0])
            reproj_pix[1] = np.around(reproj_pix[1])
            reproj_pix = reproj_pix.astype(int)

            check = np.logical_and(np.logical_and(
                reproj_pix[0] < self.W - 1, reproj_pix[0] > 0),
                np.logical_and(reproj_pix[1] < self.H - 1, reproj_pix[1] > 0))

            d = pts_trans[2, check]
            reproj_pix = reproj_pix[0:2, check]

            Ji = self.image_gradient(reproj_pix)

            self.ref_pts = np.dot(self.T, self.ref_pts)

            err,residuals = self.residuals(self.ref_pts)
            if self.last_err - err < 0.0000001:
                print("OK!")
                break
            else:
                if(show):
                    self.ax2.cla()
            self.last_err = err
            print('itr %d, err:%f'%(itr,err))
            Ji = (self.image_gradient(cur_img) + self.ref_dxdy)/2.
            J = np.zeros([self.rect[2],self.rect[3],8])
            for u in range(self.rect[2]):
                for v in range(self.rect[3]):
                    J[v,u] = np.dot(Ji[v,u,:], self.JwJg[v,u,:,:])
            J = J.reshape(-1,8)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T,residuals)
            x0 = np.dot(hessian_inv,temp)

            A = np.zeros([3,3])
            for i in range(len(self.A)):
                A += x0[i] * self.A[i]

            dH = self.exp(A)

            self.H = np.dot(self.H,dH)
            itr+=1



    def track(self, depth, show=False):
        self.cur_depth = depth
        itr = 0
        if(show):
            fig = plt.figure()
            self.ax1 = fig.add_subplot(211)
            self.ax2 = fig.add_subplot(212)
            

        while(True):
            if(show):
                self.show_process()

            self.ref_pts = np.dot(self.T, self.ref_pts)
            err,residuals = self.residuals(self.ref_pts)
            if self.last_err - err < 0.0000001:
                print("OK!")
                break
            else:
                if(show):
                    self.ax2.cla()
            self.last_err = err
            print('itr %d, err:%f'%(itr,err))
            Ji = (self.image_gradient(cur_img) + self.ref_dxdy)/2.
            J = np.zeros([self.rect[2],self.rect[3],8])
            for u in range(self.rect[2]):
                for v in range(self.rect[3]):
                    J[v,u] = np.dot(Ji[v,u,:], self.JwJg[v,u,:,:])
            J = J.reshape(-1,8)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T,residuals)
            x0 = np.dot(hessian_inv,temp)

            A = np.zeros([3,3])
            for i in range(len(self.A)):
                A += x0[i] * self.A[i]

            dH = self.exp(A)

            self.H = np.dot(self.H,dH)
            itr+=1

    def exp(self,A):
        G = np.zeros([3,3])
        A_factor = np.eye(3)
        i_factor = 1.
        for i in range(9):
            G += A_factor/i_factor
            A_factor = np.dot(A_factor, A)
            i_factor*= float(i+1) 
        return G

    def residuals(self, ref_pts):
        reprojection_depth = reprojection(ref_pts, [self.H, self.W], self.T, self.K)
        residuals = self.cur_depth - reprojection_depth
        m = np.nansum(residuals*residuals)
        return np.sqrt(m/(self.W*self.H)), residuals.reshape(-1)

    def image_gradient(self, pix):

        #dx = np.roll(img,-1,axis=1) - img
        dx = self.ref_depth[pix[1], pix[0] + 1] - self.ref_depth[pix[1], pix[0]]
        dy = self.ref_depth[pix[1] + 1, pix[0]] - self.ref_depth[pix[1], pix[0]]
        return np.dstack([dx,dy])




    def calc_Jg(self):
        A1 = np.array([0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0. ]).reshape([3,4])
        A2 = np.array([0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 0. ]).reshape([3,4])
        A3 = np.array([0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 1. ]).reshape([3,4])
        A4 = np.array([0, 0, 0, 0,  0, 0,-1, 0,  0, 1, 0, 0. ]).reshape([3,4])
        A5 = np.array([0, 0, 1, 0,  0, 0, 0, 0, -1, 0, 0, 0. ]).reshape([3,4])
        A6 = np.array([0,-1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0. ]).reshape([3,4])
        KA1 = np.dot(self.K, A1) 
        KA2 = np.dot(self.K, A2) 
        KA3 = np.dot(self.K, A3) 
        KA4 = np.dot(self.K, A4) 
        KA5 = np.dot(self.K, A5) 
        KA6 = np.dot(self.K, A6) 
        self.Jg = np.vstack([KA1.flatten(),
            KA2.flatten(),
            KA3.flatten(),
            KA4.flatten(),
            KA5.flatten(),
            KA6.flatten()]).T
        self.Jg2 = np.vstack([KA1.flatten(),
            A2.flatten(),
            A3.flatten(),
            A4.flatten(),
            A5.flatten(),
            A6.flatten()]).T


    def calc_Jw(self):
        x = self.tar_pts[0,:]
        y = self.tar_pts[1,:]
        z = self.tar_pts[2,:]
        self.Jw = np.zeros([self.tar_pts.shape[1], 2, 12])
        self.Jw[:,0,0] = x/z
        self.Jw[:,0,1] = y/z
        self.Jw[:,0,2] = 1
        self.Jw[:,0,3] = 1/z
        self.Jw[:,0,8] = -x*x/(z*z)
        self.Jw[:,0,9] = -x*y/(z*z)
        self.Jw[:,0,10] = -x/z
        self.Jw[:,0,11] = -x/(z*z)

        self.Jw[:,1,4] = x/z
        self.Jw[:,1,5] = y/z
        self.Jw[:,1,6] = 1
        self.Jw[:,1,7] = 1/z
        self.Jw[:,1,8] = -y*x/(z*z)
        self.Jw[:,1,9] = -y*y/(z*z)
        self.Jw[:,1,10] = -y/z
        self.Jw[:,1,11] = -y/(z*z)

        #self.JwJg = np.dot(self.Jw,self.Jg)






    
if __name__ == "__main__":
    ref_depth = np.load('/Users/liuyang/workspace/ViusalLidar/depth/0000.npy')
    tar_depth = np.load('/Users/liuyang/workspace/ViusalLidar/depth/0001.npy')
    esm = esm(ref_depth, tar_depth) #x,y,weight,height
    #esm.track(tar_depth, False)
    print(esm.H)
    plt.show()