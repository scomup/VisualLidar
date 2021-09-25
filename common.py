import numpy as np
from scipy.spatial.transform import Rotation

epsilon = 1e-5

# so3 to 3d Rotation Matrix
def v2R(v):
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

def v2T(x):
    M = np.eye(4)
    M[0:3,3] = x[0:3]
    M[0:3,0:3] = v2R(x[3:6])
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

def transform(T, pts):
    ptshd = np.ones((4, pts.shape[1]))
    ptshd[0:3, :] = pts        
    pts_trans = np.matmul(T, ptshd)
    return pts_trans[0:3, :]

def projection(K, pts):
    pix =  np.dot(K, pts)
    pix /= pix[2,:]
    pix[0] = np.around(pix[0])
    pix[1] = np.around(pix[1])
    pix = pix.astype(int)
    return pix

def range_check(pix, H, W):
    b = 1
    check = np.logical_and(np.logical_and(
        pix[0] < W - b, pix[0] > b),
        np.logical_and(pix[1] < H - b, pix[1] > b))
    return check
