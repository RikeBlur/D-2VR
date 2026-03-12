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
        print(f"\nProcessing {subset_name} split...")
        subset_sequences = []
        
        lq_subset_path = Path(lq_root) / subset_name
        gt_subset_path = Path(gt_root) / subset_name
        
        if not lq_subset_path.exists() or not gt_subset_path.exists():
            print(f"Warning: {lq_subset_path} or {gt_subset_path} does not exist, skipping")
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
                print(f"Warning: GT and LQ PNG file count mismatch in {subset_name}/{seq_folder}, skipping")
                continue
            
            lq_png_files = sorted(lq_png_files)
            gt_png_files = sorted(gt_png_files)
            
            if lq_png_files != gt_png_files:
                print(f"Warning: GT and LQ PNG filename mismatch in {subset_name}/{seq_folder}, skipping")
                continue
            
            first_img_path = gt_seq_path / gt_png_files[0]
            img = cv2.imread(str(first_img_path))
            if img is None:
                print(f"Warning: cannot read image {first_img_path}, skipping")
                continue
                
            height, width, channels = img.shape
            
            seq_path = f"{seq_folder}"
            frame_count = len(lq_png_files)
            subset_sequences.append((seq_path, frame_count, height, width, channels))
            
            print(f"Processing sequence: {seq_path}, frames: {frame_count}, resolution: {height}x{width}")
        
        print(f"{subset_name}: found {len(subset_sequences)} valid sequences")
        
        with open(output_file, 'w') as f:
            for seq_path, frame_count, height, width, channels in subset_sequences:
                f.write(f"{seq_path} {frame_count} ({height},{width},{channels})\n")
        
        print(f"Metadata saved to {output_file}")

if __name__ == "__main__":
    create_metadata()