import os
import cv2
import numpy as np
from pathlib import Path
import OpenEXR
import Imath
import array
from tqdm import tqdm
import re


def read_exr(file_path):
    """读取EXR文件"""
    try:
        # 打开EXR文件
        exr_file = OpenEXR.InputFile(file_path)
        
        # 获取文件头信息
        header = exr_file.header() #NOTE
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


def process_folder(input_dir, output_base_dir, data_type='depth'):
    """
    处理整个文件夹中的EXR文件
    
    Args:
        input_dir: 输入目录路径
        output_base_dir: 输出基础目录
        data_type: 数据类型 ('depth' 或 'normal')
    """
    # 创建输出目录
    output_png_dir = os.path.join(output_base_dir, f"{data_type}_png")
    output_npy_dir = os.path.join(output_base_dir, f"{data_type}_npy")
    
    os.makedirs(output_png_dir, exist_ok=True)
    os.makedirs(output_npy_dir, exist_ok=True)
    
    # 获取所有EXR文件
    exr_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.exr')])
    
    
    print(f"\n处理 {data_type} 文件夹: {input_dir}")
    print(f"找到 {len(exr_files)} 个EXR文件")
    
    # 使用进度条
    success_count = 0
    fail_count = 0
    
    for filename in tqdm(exr_files, desc=f"转换{data_type}", unit="文件"):
        file_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        # 提取数字部分作为文件名
        numbers = re.findall(r'\d+', base_name)
        if numbers:
            save_name = numbers[-1]  # 使用最后一组数字
        else:
            save_name = base_name  # 如果没有数字，使用原名
        
        try:
            # 读取EXR文件
            data = read_exr(file_path)
            
            if data is not None:
                # 保存PNG到对应目录
                png_path = os.path.join(output_png_dir, f"{save_name}.png")
                npy_path = os.path.join(output_npy_dir, f"{save_name}.npy")
                
                # 保存NPY
                np.save(npy_path, data)
                
                # 保存PNG
                if data_type == 'depth':
                    # 深度图处理：使用彩色热图
                    if data.ndim == 3:
                        depth_data = data[:, :, 0]
                    else:
                        depth_data = data
                    
                    valid_mask = np.isfinite(depth_data)
                    if np.any(valid_mask):
                        min_val = np.min(depth_data[valid_mask])
                        max_val = np.max(depth_data[valid_mask])
                        normalized = np.zeros_like(depth_data)
                        if max_val > min_val:
                            normalized[valid_mask] = (depth_data[valid_mask] - min_val) / (max_val - min_val) * 255
                        depth_gray = normalized.astype(np.uint8)
                    else:
                        depth_gray = np.zeros_like(depth_data, dtype=np.uint8)
                    
                    # 应用彩色热图 (JET colormap)
                    depth_img = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)
                    cv2.imwrite(png_path, depth_img)
                    
                elif data_type == 'normal':
                    # 法向图处理：彩色RGB显示
                    if data.ndim == 3 and data.shape[2] >= 3:
                        # 法向量通常在[-1, 1]范围，转换到[0, 255]
                        normal_data = (data[:, :, :3] + 1.0) / 2.0 * 255.0
                        normal_data = np.clip(normal_data, 0, 255).astype(np.uint8)
                        # 已经是BGR顺序的彩色图
                        cv2.imwrite(png_path, normal_data)
                    else:
                        cv2.imwrite(png_path, data)
                
                success_count += 1
                
            else:
                fail_count += 1
                
        except Exception as e:
            fail_count += 1
    
    print(f"完成 {data_type} 文件夹处理! 成功: {success_count}, 失败: {fail_count}")


def main():
    # 设置路径
    base_dir = "./tunnel"
    depth_input = os.path.join(base_dir, "depth")
    normal_input = os.path.join(base_dir, "normal")
    output_dir = os.path.join(base_dir, "output")
    
    print("=" * 60)
    print("EXR转换程序")
    print("=" * 60)
    
    # 检查输入目录是否存在
    if not os.path.exists(depth_input):
        print(f"错误: Depth目录不存在: {depth_input}")
    else:
        # 处理Depth文件夹
        process_folder(depth_input, output_dir, data_type='depth')
    
    if not os.path.exists(normal_input):
        print(f"错误: Normal目录不存在: {normal_input}")
    else:
        # 处理Normal文件夹
        process_folder(normal_input, output_dir, data_type='normal')
    
    print("\n" + "=" * 60)
    print("所有处理完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
