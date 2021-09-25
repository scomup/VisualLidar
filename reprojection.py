import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

K = np.array([[718.,0, 607], [0, 718, 185], [0,0,1]])

def getT(x, y, z, roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw])
    M = np.zeros([3, 4])
    M[0:3, 0:3] = r.as_matrix()
    M[0,3] = x
    M[1,3] = y
    M[2,3] = z
    return M

def depth2pts(depth, K):
    H, W = depth.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    x, y = np.meshgrid(x, y)
    pts = np.ones((3, W * H), dtype=np.float32)
    d = depth.flatten()
    pts[0, :] = x.flatten()
    pts[1, :] = y.flatten()
    Kinv = np.linalg.inv(K)
    ptsn = np.matmul(Kinv, pts)
    Xn = np.reshape(ptsn[0, :], newshape=(H, W))
    Yn = np.reshape(ptsn[1, :], newshape=(H, W))
    pts3d = np.ones((3, W * H), dtype=np.float32)
    pts3d[0, :] = (d*Xn.flatten())
    pts3d[1, :] = (d*Yn.flatten())
    pts3d[2, :] = depth.flatten()
    pts3d = pts3d[:, np.logical_not(np.isnan(pts3d[2, :]))]
    return pts3d

def reprojection(pts3d, img_size, T, K):
    if(pts3d.shape[0] == 3):
        pts3dh = np.ones((4, pts3d.shape[1]))
        pts3dh[0:3, :] = pts3d
    else:
        pts3dh = pts3d
    img = np.zeros([img_size[0] , img_size[1]])
    img.fill(np.nan)
    pts3dh_trans = np.dot(T, pts3dh)
    reproj_pix =  np.dot(K, pts3dh_trans[0:3,:])
    reproj_pix /= reproj_pix[2,:]
    x_idx = np.around(reproj_pix[0]).astype(int)
    y_idx = np.around(reproj_pix[1]).astype(int)
    check = np.logical_and(np.logical_and(x_idx < img_size[1], x_idx > 0),
            np.logical_and(y_idx < img_size[0], y_idx > 0))
    x_idx = x_idx[check]
    y_idx = y_idx[check]
    d = pts3d[2, check]
    img[y_idx, x_idx] = d

    return img

#if __main__
#pts3d = depth2pts(depth, K)
#T = getT(0, 0, 1, 0, 0, 0)
#reproj_depth = reprojection(pts3d, depth.shape, T, K)
#
#
#plt.imshow(reproj_depth, cmap="rainbow")
#plt.show()



