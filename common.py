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
    return rot.as_matrix()

def v2T(x):
    M = np.eye(4)
    M[0:3,3] = x[0:3]
    M[0:3,0:3] = v2R(x[3:6])
    return M

def depth2pts(depth, K, sampling = 1):
    assert type(sampling) == int, "Incorrect sampling data"

    H, W = depth.shape 
    x = np.arange(0, W - 1, sampling)
    y = np.arange(0, H - 1, sampling)
    N = x.shape[0] * y.shape[0]
    x, y = np.meshgrid(x, y)
    pts = np.ones((3, N), dtype=np.float32)
    d = depth[y,x].flatten()
    pts[0, :] = x.flatten()
    pts[1, :] = y.flatten()
    Kinv = np.linalg.inv(K)
    ptsn = np.matmul(Kinv, pts)
    pts3d = np.ones((3, N), dtype=np.float32)
    pts3d = ptsn * d
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

def reprojection_error_image(ref_depth, tar_depth, T, K):
    tar_pts = depth2pts(tar_depth, K)
    tar_pts = tar_pts[:, np.argsort(tar_pts[2])]
    tar_pts = tar_pts[:,::-1]
    ref_pts = transform(T, tar_pts)
    pix = projection(K, ref_pts)
    check = range_check(pix,ref_depth.shape[0],ref_depth.shape[1])
    ref_pts = ref_pts[:,check]
    pix = pix[:,check]
    d = ref_pts[2,:]
    r = ref_depth[pix[1], pix[0]] - d
    e = np.sqrt(r*r)
    error_image = np.zeros_like(ref_depth)
    reproj_image = np.zeros_like(ref_depth)
    reproj_image.fill(np.nan)
    error_image.fill(np.nan)
    error_image[pix[1], pix[0]] = e
    reproj_image[pix[1], pix[0]] = d
    return error_image, reproj_image

def range_check(pix, H, W, border=1):
    check = np.logical_and(np.logical_and(
        pix[0] < W - border, pix[0] > border),
        np.logical_and(pix[1] < H - border, pix[1] > border))
    return check

def random_mask(check, num):
    mask_size = check.shape[0]
    if(mask_size < num):
        return check
    mask = np.zeros(mask_size, dtype=bool)
    idx = np.random.choice(np.arange(mask_size)[check],num,replace=False)
    mask[idx] = True
    return mask