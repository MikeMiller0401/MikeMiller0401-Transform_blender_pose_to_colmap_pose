import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def depth_to_pcd(color_dir, depth_dir, fx, fy, cx, cy, output_dir):
    """
    Convert depth images to point clouds using camera intrinsics.
    Save each frame as a separate pcd file.
    
    Args:
        color_dir: Directory containing color images
        depth_dir: Directory containing depth images
        fx, fy, cx, cy: Camera intrinsic parameters
        output_dir: Directory to save the point clouds (npy format)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(('.png', '.jpg', '.npy'))])
    
    print(f"处理 {len(depth_files)} 个深度图...")
    
    for depth_file in tqdm(depth_files, desc="生成点云"):
        depth_path = os.path.join(depth_dir, depth_file)
        
        # Load depth
        if depth_file.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path), dtype=np.float32)
        
        # 如果是多通道，取第一个通道
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        
        # Load corresponding color
        color_file = depth_file.replace('.npy', '.jpg').replace('.png', '.jpg')
        color_path = os.path.join(color_dir, color_file)
        if os.path.exists(color_path):
            color = np.array(Image.open(color_path), dtype=np.uint8)
        else:
            color = np.ones_like(depth, dtype=np.uint8) * 128
        
        # Depth to 3D points
        h, w = depth.shape
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        z = depth
        x_3d = (xx - cx) * z / fx
        y_3d = (yy - cy) * z / fy
        z_3d = z
        
        # Flatten and stack (只保存xyz坐标)
        pts = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3)
        
        # 保存为单独的npy文件，保持原文件名（转换为.npy格式）
        output_filename = os.path.splitext(depth_file)[0] + '.npy'
        output_file = os.path.join(output_dir, output_filename)
        np.save(output_file, pts.astype(np.float32))
    
    print(f"所有点云已保存到 {output_dir}")
    print(f"共生成 {len(depth_files)} 个点云文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DK数据集的相机内参（根据实际情况调整）
    # parser.add_argument("--fx", type=float, default=640, help="相机内参 fx")
    # parser.add_argument("--fy", type=float, default=640, help="相机内参 fy")
    # parser.add_argument("--cx", type=float, default=480, help="相机内参 cx")
    # parser.add_argument("--cy", type=float, default=270, help="相机内参 cy")
    
    # Replica数据集的相机内参（根据实际情况调整）
    parser.add_argument("--fx", type=float, default=640, help="相机内参 fx")
    parser.add_argument("--fy", type=float, default=640, help="相机内参 fy")
    parser.add_argument("--cx", type=float, default=480, help="相机内参 cx")
    parser.add_argument("--cy", type=float, default=270, help="相机内参 cy")

    args = parser.parse_args()
    
    base_dir = "./tunnel"
    color_dir = os.path.join(base_dir, "jpg")
    depth_dir = os.path.join(base_dir, "output/depth_npy")
    output_dir = os.path.join(base_dir, "output/pointcloud_single")

    depth_to_pcd(color_dir, depth_dir, args.fx, args.fy, args.cx, args.cy, output_dir)