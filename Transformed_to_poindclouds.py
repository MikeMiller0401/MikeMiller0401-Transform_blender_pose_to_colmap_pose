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
    img = np.stack([channel_data['B'], channel_data['G'], channel_data['R']], axis=-1)
    
    
    return img   

def transfrom_pose(poses_file):
    T_world = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ])
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])   
    with open(poses_file, 'r') as f:
        lines = f.readlines()
        
    saved_count = 0
    i = 0
    pose_all = {}
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

                # frame_name = f"{frame_id:04d}"
                # output_file = os.path.join(output_dir, f"{frame_name}.txt")
                # np.savetxt(output_file, pose, fmt='%.8f')
                saved_count += 1
                pose_all[frame_id] = pose
            i += 5
        else:
            i += 1
            
            
    return pose_all

def save_ply(filename, points):
    """保存点云为 PLY 格式"""
    has_color = points.shape[1] == 6
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for point in points:
            if has_color:
                x, y, z, r, g, b = point
                r = int(np.clip(r * 255, 0, 255))
                g = int(np.clip(g * 255, 0, 255))
                b = int(np.clip(b * 255, 0, 255))
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            else:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
                
def voxel_downsample(points, voxel_size):
    """体素下采样"""
    if voxel_size <= 0:
        return points
    positions = points[:, :3]
    voxel_indices = np.floor(positions / voxel_size).astype(np.int32)
    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        if key not in voxel_dict:
            voxel_dict[key] = i
    return points[list(voxel_dict.values())]

def generate_point_clouds(pcd_pose, depth_dir,output_dir, fx, fy, cx, cy, voxel_size, start_frame, end_frame):
    """
    Convert depth images to point clouds using camera intrinsics.
    Each .EXR depth image should be transformed to .npy format file.
    Save each frame as a separate pcd file.
    
    Args:
        pose: List of camera poses
        depth_dir: Directory containing depth images (.exr)
        fx, fy, cx, cy: Camera intrinsic parameters
        output_dir: Directory to save the point clouds (npy format)
        voxel_size: Size of the voxel for downsampling
        start_frame: Starting frame number
        end_frame: Ending frame number
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pose = pcd_pose
    
    all_points = []
    
    exr_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith('.exr')])
    
    for index, filename in enumerate(
            tqdm(exr_files[start_frame:end_frame], desc="EXR to NPY", unit="file")):
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

        pts = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3) #
        
        pts = voxel_downsample(pts, voxel_size)
        
        positions = pts[:, :3]
        ones = np.ones((positions.shape[0], 1))
        positions_homo = np.hstack([positions, ones])
        i = index + 1
        transfromed_pose = (pose[i] @ positions_homo.T).T[:, :3] 
        all_points.append(transfromed_pose)
        
        if index % 20 == 0:
            print(f"Processed: {index} frames")
    print("Start gobal down-sample")   
    merged_points = np.vstack(all_points)
    merged_points = voxel_downsample(merged_points, voxel_size=voxel_size)
    ply_path = os.path.join(output_dir, "merged.ply")
    save_ply(ply_path, merged_points)
    npy_path = ply_path.replace('.ply', '.npy')
    np.save(npy_path, merged_points)
    print(f"The merged point cloud is saved in {ply_path}")
    print("Finished")

def main():
    base_dir = "./tunnel"
    pose = transfrom_pose(poses_file=os.path.join(base_dir, "camera_poses_mm.txt"),)
    generate_point_clouds(pcd_pose=pose,
                        depth_dir=os.path.join(base_dir, "depth"),
                        output_dir=os.path.join(base_dir, "output"),
                        fx=640.0, fy=640.0, cx=480.0, cy=270.0,
                        voxel_size=0.01,
                        start_frame=1,
                        end_frame=100)

if __name__ == "__main__":
    main()