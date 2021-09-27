from __future__ import print_function, division
from math import nan
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import skimage
import skimage.io
import skimage.transform
import numpy as np
from PIL import Image
from datasets.data_io import get_transform
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
import open3d as o3d
import matplotlib.pyplot as plt

path = '/media/liu/SSD-PSMU3/kitti_slam/03/'
model = BGNet_Plus().cuda()

checkpoint = torch.load('models/Sceneflow-IRS-BGNet-Plus.pth',map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint) 
model.eval()

import glob, os
left_fn = []
right_fn = []
os.chdir(path + 'image_2/')
for file in glob.glob("*.png"):
    left_fn.append(path + 'image_2/' + file)
    right_fn.append(path + 'image_3/' + file)
left_fn.sort()
right_fn.sort()

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(left_fn)):
    left_img = Image.open(left_fn[i]).convert('L')
    right_img = Image.open(right_fn[i]).convert('L')
    w, h = left_img.size
    print(left_fn[i])
    h1 = h % 64
    w1 = w % 64
    h1 = h  - h1
    w1 =  w - w1
    h1 = int(h1)
    w1 = int(w1)
    left_img_color = Image.open(left_fn[i]).resize((w1, h1),Image.ANTIALIAS)


    left_img = left_img.resize((w1, h1),Image.ANTIALIAS)
    right_img = right_img.resize((w1, h1),Image.ANTIALIAS)
    preprocess = get_transform()
    left_img_tensor = preprocess(np.ascontiguousarray(left_img, dtype=np.float32))
    right_img_tensor = preprocess(np.ascontiguousarray(right_img, dtype=np.float32))
    pred,_ = model(left_img_tensor.unsqueeze(0).cuda(), right_img_tensor.unsqueeze(0).cuda()) 
    pred = pred[None, :, :]
    pred = torch.nn.functional.interpolate(pred, (h, w), mode='nearest')
    pred = pred[0, 0].data.cpu().numpy()

    fx = 721.538
    fy = 721.538
    cx = 609.559
    cy = 172.854
    baseline = 0.54
    max_depth = 30
    depth = baseline * fx / pred
    depth =  np.where(depth > max_depth, nan, depth)


    """
    color = o3d.geometry.Image(np.array(left_img_color))
    depth = o3d.geometry.Image(depth.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, 718, 718, 607, 185)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
    """
    np.save('/home/liu/workspace/VisualLidar/depth/%04d'%i, depth)

    plt.cla()
    ax.imshow(depth, cmap="rainbow")
    
    plt.pause(0.01)