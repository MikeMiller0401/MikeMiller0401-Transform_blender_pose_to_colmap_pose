import os
import numpy as np

# ============================================================
# 01 - 提取并保存位姿
# 将 camera_poses_mm.txt 中的位姿提取出来，完成以下处理后保存：
#   1. 归一化旋转矩阵
#   2. 平移从毫米转米
#   3. 坐标系转换 real -> blender  （左乘 T_world）
#   4. 烘焙点云 y/z 翻转          （右乘 flip_yz）
# 这样 03 直接将原始点云乘以保存的 pose 即可，无需任何额外处理。
# 等价变换：P_world = T_world @ pose @ flip_yz @ p_local
# ============================================================

def real_to_blender(pose):
    """世界坐标系(X右,Y下,Z前) -> Blender世界系(X右,Y前,Z上)，左乘改变世界系基"""
    T_world = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ])
    
    print(T_world @ pose)
    return T_world @ pose

def parse_and_save_poses(poses_file, output_dir, start_frame=1, end_frame=None):
    """
    从 camera_poses_mm.txt 读取位姿，处理后逐帧保存为 {frame_id:04d}.txt

    保存的 pose 已烘焙所有坐标系变换，03 可直接使用。
    """
    # 点云局部坐标系 y/z 取反矩阵（与 TB 中 transform_points_blender 等价）
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

            # 跳过不在范围内的帧
            if end_frame is not None and frame_id > end_frame:
                i += 5
                continue
            if frame_id < start_frame:
                i += 5
                continue

            pose_lines = []
            for j in range(1, 5):
                if i + j < len(lines):
                    pose_lines.append(lines[i + j].strip())

            if len(pose_lines) == 4:
                pose_raw = np.array([[float(x) for x in l.split()] for l in pose_lines])

                R_raw = pose_raw[:3, :3]
                t_raw = pose_raw[:3, 3]

                # 1. 归一化旋转矩阵
                scale = np.linalg.norm(R_raw[:, 0])
                R = R_raw / scale

                # 2. 平移从毫米转米
                t = t_raw / 1000.0

                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t

                # 3. 坐标系转换 real -> blender（左乘）
                # 4. 烘焙点云 y/z 翻转（右乘），使 03 直接使用原始点云
                pose = (real_to_blender(pose) ) @ flip_yz

                frame_name = f"{frame_id:04d}"
                output_file = os.path.join(output_dir, f"{frame_name}.txt")
                np.savetxt(output_file, pose, fmt='%.8f')
                saved_count += 1

            i += 5
        else:
            i += 1

    print(f"完成! 共保存 {saved_count} 个位姿到 {output_dir}")


if __name__ == "__main__":
    base_dir = "./tunnel"
    parse_and_save_poses(
        poses_file=os.path.join(base_dir, "camera_poses_mm.txt"),
        output_dir=os.path.join(base_dir, "output/pose"),
        start_frame=1,
        end_frame=400
    )