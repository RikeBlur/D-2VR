import os
import cv2
import random
from pathlib import Path
from tqdm import tqdm

gt_root = "/mnt/sjtu/liangjianfeng/REDS/"
lq_root = "/mnt/sjtu/liangjianfeng/REDS-BI/"
output_dir = "/home/liangjianfeng/stablevsr-2026/dataset/REDS"
train_output_file = f"{output_dir}/train_metadata.txt"
valid_output_file = f"{output_dir}/valid_metadata.txt"

os.makedirs(output_dir, exist_ok=True)


def create_metadata():
    subsets = {
        'train': train_output_file,
        'val': valid_output_file
    }
    
    for subset_name, output_file in subsets.items():
        print(f"\n正在处理 {subset_name} 集...")
        subset_sequences = []
        
        lq_subset_path = Path(lq_root) / subset_name
        gt_subset_path = Path(gt_root) / subset_name
        
        if not lq_subset_path.exists() or not gt_subset_path.exists():
            print(f"警告: {lq_subset_path} 或 {gt_subset_path} 不存在，跳过该子集")
            continue
        
        seq_folders = sorted(os.listdir(lq_subset_path))
        
        for seq_folder in seq_folders:
            lq_seq_path = lq_subset_path / seq_folder
            gt_seq_path = gt_subset_path / seq_folder
            
            if not lq_seq_path.is_dir() or not gt_seq_path.is_dir():
                continue
            
            lq_png_files = [f for f in os.listdir(lq_seq_path) if f.lower().endswith('.png')]
            gt_png_files = [f for f in os.listdir(gt_seq_path) if f.lower().endswith('.png')]
            
            if not lq_png_files or not gt_png_files or len(lq_png_files) != len(gt_png_files):
                print(f"警告: {subset_name}/{seq_folder} 中GT和LQ的PNG文件数量不匹配，跳过")
                continue
            
            lq_png_files = sorted(lq_png_files)
            gt_png_files = sorted(gt_png_files)
            
            if lq_png_files != gt_png_files:
                print(f"警告: {subset_name}/{seq_folder} 中GT和LQ的PNG文件名不匹配，跳过")
                continue
            
            first_img_path = gt_seq_path / gt_png_files[0]
            img = cv2.imread(str(first_img_path))
            if img is None:
                print(f"警告: 无法读取图像 {first_img_path}，跳过")
                continue
                
            height, width, channels = img.shape
            
            seq_path = f"{seq_folder}"
            frame_count = len(lq_png_files)
            subset_sequences.append((seq_path, frame_count, height, width, channels))
            
            print(f"处理序列: {seq_path}, 帧数: {frame_count}, 分辨率: {height}x{width}")
        
        print(f"{subset_name}集: 共找到 {len(subset_sequences)} 个有效序列")
        
        with open(output_file, 'w') as f:
            for seq_path, frame_count, height, width, channels in subset_sequences:
                f.write(f"{seq_path} {frame_count} ({height},{width},{channels})\n")
        
        print(f"元数据已保存到 {output_file}")

if __name__ == "__main__":
    create_metadata()