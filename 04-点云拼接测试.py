import numpy as np
import os
from pathlib import Path

# ============================================================
# 03 - 读取位姿，拼接点云
# 依赖 01 已将所有坐标系变换烘焙进 pose，
# 此处直接将原始点云乘以 pose 即可，无需任何额外坐标系处理。
# 等价变换：P_world = pose_saved @ p_local
# ============================================================

def load_pcd_from_npy(npy_path):
    """从 .npy 文件加载点云（Nx3 或 Nx6）"""
    return np.load(npy_path)

def load_pose(pose_path):
    """加载由 01 保存的 4x4 位姿矩阵"""
    return np.loadtxt(pose_path)

def transform_points(points, pose):
    """
    将点云变换到世界坐标系。
    pose 由 01 保存，已包含所有坐标系变换（real->blender + y/z翻转），
    此处直接乘以原始点坐标即可。
    """
    if points.shape[1] >= 6:
        positions = points[:, :3]
        colors = points[:, 3:6]
        has_color = True
    else:
        positions = points[:, :3]
        colors = None
        has_color = False

    ones = np.ones((positions.shape[0], 1))
    positions_homo = np.hstack([positions, ones])       # 转换为齐次形式
    transformed_positions = (pose @ positions_homo.T).T[:, :3]

    if has_color:
        return np.hstack([transformed_positions, colors])
    else:
        return transformed_positions

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

def merge_point_clouds(pcd_dir, pose_dir, output_path,
                       start_frame=1, end_frame=100, voxel_size=0.01):
    """
    读取 01 保存的 pose，将各帧点云变换到世界坐标系后拼接。

    Args:
        pcd_dir:     点云 .npy 文件目录
        pose_dir:    01 保存的 pose .txt 文件目录
        output_path: 输出 .ply 路径
        start_frame: 起始帧编号
        end_frame:   结束帧编号
        voxel_size:  体素下采样大小（米），<=0 则不下采样
    """
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
        points = load_pcd_from_npy(str(pcd_file))

        # 单帧下采样
        if voxel_size > 0:
            points = voxel_downsample(points, voxel_size)

        # 加载 pose 并变换（pose 已由 01 烘焙好所有变换）
        pose = load_pose(str(pose_file))
        transformed_points = transform_points(points, pose)

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


if __name__ == "__main__":
    base_dir = "./tunnel/output"
    merge_point_clouds(
        pcd_dir=os.path.join(base_dir, "pointclouds"),
        pose_dir=os.path.join(base_dir, "poses_colmap"),
        output_path=os.path.join(base_dir, "new_merged_pcd.ply"),
        start_frame=1,
        end_frame=100,
        voxel_size=0.01
    )