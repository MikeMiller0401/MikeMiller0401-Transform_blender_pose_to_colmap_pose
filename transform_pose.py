import os
import cv2
import numpy as np
from pathlib import Path
import OpenEXR
import Imath
import array
from tqdm import tqdm
import re
from PIL import Image

def read_depth_image(filepath):
    """读取EXR文件"""
    try:
        # 打开EXR文件
        exr_file = OpenEXR.InputFile(filepath)
        
        # 获取文件头信息
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # 获取通道名称
        channels = list(header['channels'].keys())
        
        # 定义像素类型
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # 读取所有通道
        channel_data = {}
        for channel in channels:
            channel_str = exr_file.channel(channel, FLOAT)
            channel_array = array.array('f', channel_str)
            channel_data[channel] = np.array(channel_array, dtype=np.float32)
            channel_data[channel] = channel_data[channel].reshape(height, width)
        
        # 组合通道
        if 'R' in channels and 'G' in channels and 'B' in channels:
            # RGB图像（法向图）
            img = np.stack([channel_data['B'], channel_data['G'], channel_data['R']], axis=-1)
        elif 'Y' in channels:
            # 单通道深度图
            img = channel_data['Y']
        elif len(channels) == 1:
            # 单通道
            img = list(channel_data.values())[0]
        else:
            # 多通道，按顺序组合
            img = np.stack([channel_data[ch] for ch in channels], axis=-1)
        
        return img
        
    except Exception as e:
        return None

def depth_to_pcd(color_dir, depth_dir, fx, fy, cx, cy, output_dir):
    """
    Convert depth images to point clouds using camera intrinsics.
    Each .EXR depth image should be transformed to .npy format file.
    Save each frame as a separate pcd file.
    
    Args:
        color_dir: Directory containing color images (.jpg)
        depth_dir: Directory containing depth images (.exr)
        fx, fy, cx, cy: Camera intrinsic parameters
        output_dir: Directory to save the point clouds (npy format)
    """
    os.makedirs(output_dir, exist_ok=True)

    exr_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith('.exr')])
    for index, filename in enumerate(tqdm(exr_files, desc="EXR to NPY", unit="file")):
        depth_data = read_depth_image(os.path.join(depth_dir, filename))
        if depth_data.ndim == 3:
            depth = depth_data[:, :, 0]
        else:
            return None
        
        h, w = depth.shape
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        z = depth
        x_3d = (xx - cx) * z / fx
        y_3d = (yy - cy) * z / fy
        z_3d = z
        pts = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3)
        index = index + 1
        output_filename = f"{index:04d}.npy"
        output_file = os.path.join(output_dir, output_filename)
        np.save(output_file, pts.astype(np.float32))

def transfrom_pose(poses_file, output_dir):
    T_world = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ])
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])

    os.makedirs(output_dir, exist_ok=True)
    with open(poses_file, 'r') as f:
        lines = f.readlines()
    saved_count = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('frame'):
            frame_id = int(line.split()[1])

            pose_lines = []
            for j in range(1, 5):
                if i + j < len(lines):
                    pose_lines.append(lines[i + j].strip())

            if len(pose_lines) == 4:
                pose_raw = np.array([[float(x) for x in l.split()] for l in pose_lines])
                R_raw = pose_raw[:3, :3]
                t_raw = pose_raw[:3, 3]
                scale = np.linalg.norm(R_raw[:, 0])
                R = R_raw / scale
                t = t_raw / 1000.0
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t
                pose = T_world @ pose @ flip_yz

                frame_name = f"{frame_id:04d}"
                output_file = os.path.join(output_dir, f"{frame_name}.txt")
                np.savetxt(output_file, pose, fmt='%.8f')
                saved_count += 1

            i += 5
        else:
            i += 1

def main():
    base_dir = "./tunnel"

    depth_to_pcd(color_dir=os.path.join(base_dir, "color"),
                 depth_dir=os.path.join(base_dir, "depth"),
                 fx=525.0, fy=525.0, cx=319.5, cy=239.5,
                 output_dir=os.path.join(base_dir, "output/pointclouds"))
    
    transfrom_pose(poses_file=os.path.join(base_dir, "camera_poses_mm.txt"),
                   output_dir=os.path.join(base_dir, "output/poses_colmap"))

    

if __name__ == "__main__":
    print("开始转换深度图为点云...")
    
    main()