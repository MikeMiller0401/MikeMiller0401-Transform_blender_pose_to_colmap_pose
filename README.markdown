# Transform Blender Pose to Colmap Pose

这个仓库用于处理相机位姿和深度图像，并将其转换为统一世界坐标系下的三维点云。它适用于包含深度 EXR 序列与相机位姿文件的数据集，例如 tunnel 示例数据。

## 项目目标

- 读取相机位姿文件 camera_poses_mm.txt
- 将位姿从毫米换算为米
- 完成坐标系转换，适配 Blender / COLMAP 风格的世界坐标系
- 从深度 EXR 图像反投影为 3D 点
- 对点云进行体素下采样并融合为最终点云
- 输出格式为 .ply 和 .npy，便于后续可视化或重建

## 主要脚本

- [Transformed_to_pointclouds.py](Transformed_to_pointclouds.py)
  - 主要流程脚本
  - 读取 depth 目录下的 EXR 深度图
  - 使用相机内参反投影为点云
  - 生成 merged.ply 和 merged.npy

- [TB_test_support_real_data.py](TB_test_support_real_data.py)
  - 用于基于已生成的 .npy 点云和相机位姿做拼接与融合

- [code_byHWT](code_byHWT)
  - 存放早期实验和辅助脚本

## 依赖

建议使用 Python 3.8+。需要安装以下依赖：

```bash
pip install numpy opencv-python OpenEXR Imath Pillow tqdm
```

如果你的环境中 OpenEXR 安装遇到问题，可以先查看系统是否已安装相关编译依赖，或改用 conda 环境安装。

## 快速开始

1. 准备数据
   - 将深度图放在 tunnel/depth 目录下
   - 将相机位姿文件放在 tunnel/camera_poses_mm.txtREADME.markdown
   - 打开 [Transformed_to_pointclouds.py](Transformed_to_pointclouds.py)
   - 根据你的数据调整：
     - base_dir
     - fx, fy, cx, cy
     - voxel_size
     - start_frame, end_frame

3. 运行主脚本

```bash
python Transformed_to_pointclouds.py
```

运行结束后，结果会输出到：

- tunnel/output/merged.ply
- tunnel/output/merged.npy

## 说明

- `voxel_size` 越小，点云越密；越大，点云越稀疏。
- 脚本中的位姿变换矩阵是根据当前数据坐标系设定写死的，若换数据集，需要根据实际坐标系重新调整。
- 如果只想基于已有的 .npy 点云进行融合，可以直接运行 [TB_test_support_real_data.py](TB_test_support_real_data.py)。

## 目录结构

```text
.
├── Transformed_to_pointclouds.py
├── TB_test_support_real_data.py
├── code_byHWT/
├── tunnel/
│   ├── camera_poses_mm.txt
│   ├── camera_intrinsics.txt
│   ├── depth/
│   └── output/
└── README.markdown
```

