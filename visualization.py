#!/usr/bin/env python3

import numpy as np
from common import *
from optimizer import DepthError
import matplotlib.pyplot as plt
import rospy
import struct
import PIL
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs.msg import Image
import tf
from tf.transformations import *

def decompose_transform(transform):
    rotation = quaternion_from_matrix(transform)
    pose = np.array(transform[0:3,3]).flatten()
    return (pose, rotation)

def get_points_msg(points):
    pointsColor = np.zeros( (points.shape[1], 1), \
        dtype={ 
            "names": ( "x", "y", "z", "rgba" ), 
            "formats": ( "f4", "f4", "f4", "u4" )} )
    pointsColor["x"] = points[0, :].reshape((-1, 1))
    pointsColor["y"] = points[1, :].reshape((-1, 1))
    pointsColor["z"] = points[2, :].reshape((-1, 1))
    pointsColor["rgba"] = points[3, :].reshape((-1, 1))
    header = Header()
    header.stamp = rospy.Time().now()
    header.frame_id = "map"

    msg = PointCloud2()
    msg.header = header

    msg.height = 1
    msg.width = points.shape[1]

    msg.fields = [
        PointField('x',  0, PointField.FLOAT32, 1),
        PointField('y',  4, PointField.FLOAT32, 1),
        PointField('z',  8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
        ]

    msg.is_bigendian = False
    msg.point_step   = 16
    msg.row_step     = msg.point_step * points.shape[0]
    msg.is_dense     = int( np.isfinite(points).all() )
    msg.data         = pointsColor.tostring()

    return msg



def get_rgbd_points(points, color):

    C = np.zeros((points.shape[1], 4), dtype=np.uint8) 

    C[:, 0] = color[2,:].astype(np.uint8)
    C[:, 1] = color[1,:].astype(np.uint8)
    C[:, 2] = color[0,:].astype(np.uint8)

    C = C.view("uint32")

    pp = np.zeros([4, points.shape[1]])
    pp[0:3,:] = points
    pp[3] = C.flatten()
    return pp


if __name__ == "__main__":
    rospy.init_node("create_cloud_xyzrgb")
    pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)
    br = tf.TransformBroadcaster() 
    img_pub = rospy.Publisher("/image_raw", Image,  queue_size=10)
    depth_pub = rospy.Publisher("/depth", Image,  queue_size=10)
    #self.br.sendTransform(pose, rotation, rospy.Time.now(), 'base_link' ,'map')
    fx = 721.538
    fy = 721.538
    cx = 609.559
    cy = 172.854
    K = np.array([[fx,0, cx], [0, fy, cy], [0,0,1]])
    curT = np.eye(4)
    fig, axes= plt.subplots(3)
    r = rospy.Rate(10) # 10hz
    i = 1
    pi2 = np.pi/2
    V = quaternion_matrix(quaternion_from_euler(-np.pi/2,0,-np.pi/2))

    lmap = []
    while not rospy.is_shutdown():
        if(i > 800):
            break
        #print('process image: %d'%i)
        dT = np.load('/home/liu/workspace/VisualLidar/mapping/T/T_%04d.npy'%i)
        pts = np.load('/home/liu/workspace/VisualLidar/mapping/pts/pts_%04d.npy'%i)
        img = PIL.Image.open('/media/liu/SSD-PSMU3/kitti_slam/00/image_2/%06d.png'%i)
        depth = np.load('/home/liu/workspace/VisualLidar/depth/%04d.npy'%i)
        baseline = 0.54
        max_depth = 30
        disp = baseline * fx / depth

        mask = np.arange(0, pts.shape[1] - 1, 2)
        pts = pts[:, mask]
        curT = np.matmul(curT, np.linalg.inv(dT))

        color = pts[3:6, :]
        vT = np.matmul(V, curT)
        pts_t = transform(np.matmul(V, curT), pts[0:3,:])
        points = get_rgbd_points(pts_t, color)
        pose, rotation = decompose_transform(vT)
        br.sendTransform(pose, rotation, rospy.Time.now(), 'base_link' ,'map')
        lmap.append(points)
        img_msg = bridge.cv2_to_imgmsg(np.asarray(img), "rgb8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = 'image'
        img_pub.publish(img_msg)
        depth_msg = bridge.cv2_to_imgmsg(disp, 'passthrough')
        depth_msg.header.stamp = rospy.Time.now()
        depth_msg.header.frame_id = 'image'
        depth_pub.publish(depth_msg)

        if(i%1 == 0):
            map_points = np.hstack(lmap)
            print('process image: %d'%i)
            print(map_points.shape)
            msg = get_points_msg(map_points)
            pub.publish(msg)
            lmap = []
            r.sleep()
        i += 1
    