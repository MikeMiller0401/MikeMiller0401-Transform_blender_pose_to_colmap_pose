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

        i = index + 1
        output_filename = f"{i:04d}.npy"
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

def merge_point_clouds(pcd_dir, pose_dir, output_path,start_frame, end_frame, voxel_size):
    pcd_dir = Path(pcd_dir)
    pose_dir = Path(pose_dir)

    print(f"点云拼接")
    print(f"帧范围: {start_frame} - {end_frame}")
    print(f"体素大小: {voxel_size}m")
    print("-" * 40)

    all_points = []
    processed = 0

    for frame_id in range(start_frame, end_frame + 1):
        frame_name = f"{frame_id:04d}"
        pcd_file = pcd_dir / f"{frame_name}.npy"
        pose_file = pose_dir / f"{frame_name}.txt"

        if not pcd_file.exists() or not pose_file.exists():
            continue

        # 加载点云
        points = np.load(str(pcd_file))

        # 单帧下采样
        if voxel_size > 0:
            points = voxel_downsample(points, voxel_size)
            
        # 加载 pose 并变换（pose 已由 01 烘焙好所有变换）
        pose = np.loadtxt(str(pose_file))

        positions = points[:, :3]
        ones = np.ones((positions.shape[0], 1))
        positions_homo = np.hstack([positions, ones])       # 转换为齐次形式
        transformed_points = (pose @ positions_homo.T).T[:, :3]

        all_points.append(transformed_points)
        processed += 1

        if processed % 20 == 0:
            print(f"已处理: {processed} 帧")

    print(f"\n共处理 {processed} 帧")

    merged_points = np.vstack(all_points)
    print(f"合并后点数: {len(merged_points)}")
    
    # 全局下采样
    if voxel_size > 0:
        merged_points = voxel_downsample(merged_points, voxel_size)
        print(f"最终点数: {len(merged_points)}")

    print(f"\n保存到: {output_path}")
    save_ply(output_path, merged_points)

    npy_path = output_path.replace('.ply', '.npy')
    np.save(npy_path, merged_points)
    print(f"同时保存: {npy_path}")

    print("\n完成！")
    return merged_points
    

def main():
    base_dir = "./tunnel"

    # depth_to_pcd(color_dir=os.path.join(base_dir, "color"),
    #              depth_dir=os.path.join(base_dir, "depth"),
    #              fx=640.0, fy=640.0, cx=480.0, cy=270.0,
    #              output_dir=os.path.join(base_dir, "output/pointclouds"))
    
    # transfrom_pose(poses_file=os.path.join(base_dir, "camera_poses_mm.txt"),
    #                output_dir=os.path.join(base_dir, "output/poses_colmap"))

    merge_point_clouds(
            pcd_dir=os.path.join(base_dir, "output/pointclouds"),
            pose_dir=os.path.join(base_dir, "output/poses_colmap"),
            output_path=os.path.join(base_dir, "output/new_merged_pcd.ply"),
            start_frame=1,
            end_frame=100,
            voxel_size=0.01
        )
if __name__ == "__main__":
    
    main()