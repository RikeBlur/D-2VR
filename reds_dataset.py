import os
import random
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils import data as data
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.data.transforms import augment, paired_random_crop

class REDSDataset(data.Dataset):
    """REDS dataset for training video super-resolution models.
    
    This dataset is designed to work with VFHQ dataset where frames are organized in
    group/sequence folders with 8-digit naming format (00000000.png, 00000001.png, etc).
    
    Args:
        opt (dict): Config for train dataset.
    """

    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        
        self.folders = []
        self.frame_counts = {}
        self.frame_names = {}
        
        if os.path.exists(opt['meta_info_file']) and os.path.getsize(opt['meta_info_file']) > 0:
            logger = get_root_logger()
            logger.info(f"从元数据文件加载: {opt['meta_info_file']}")
            
            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        folder = parts[0]
                        frame_count = int(parts[1])
                        self.folders.append(folder)
                        self.frame_counts[folder] = frame_count
                        
                        # group_name, seq_name = folder.split('/')
                        folder_path = self.gt_root / folder
                        
                        if not folder_path.exists():
                            print(f"警告: 文件夹不存在 {folder_path}，跳过")
                            continue
                            
                        try:
                            frame_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')],
                                            key=lambda x: int(x.split('.')[0]))
                        except Exception as e:
                            print(f"警告: 读取文件夹出错 {folder_path}: {e}，跳过")
                            continue

                        if not frame_files:
                            print(f"警告: 文件夹中没有PNG文件 {folder_path}，跳过")
                            continue

                        self.frame_names[folder] = frame_files
        else:
            logger = get_root_logger()
            logger.error(f"元数据文件不存在或为空: {opt['meta_info_file']}")
            raise FileNotFoundError(f"元数据文件不存在或为空: {opt['meta_info_file']}")
        
        self.keys = []
        for folder in self.folders:
            frame_count = self.frame_counts[folder]
            
            if frame_count >= self.num_frame:
                for i in range(frame_count - self.num_frame + 1):
                    self.keys.append((folder, i))
        
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
        
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'时间增强间隔列表: [{interval_str}]; '
                    f'随机反转: {self.random_reverse}.')
        logger.info(f'找到 {len(self.keys)} 个有效训练样本')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        folder, start_idx = self.keys[index]
        # group_name, seq_name = folder.split('/')
        
        interval = random.choice(self.interval_list)
        frame_files = self.frame_names[folder]
        
        neighbor_indices = [start_idx + i * interval for i in range(self.num_frame)]
        
        if max(neighbor_indices) >= len(frame_files):
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        neighbor_files = [frame_files[i] for i in neighbor_indices]
        
        if self.random_reverse and random.random() < 0.5:
            neighbor_files.reverse()
        
        img_lqs = []
        img_gts = []
        
        for frame_file in neighbor_files:
            img_lq_path = self.lq_root / folder / frame_file
            img_gt_path = self.gt_root / folder / frame_file
            
            try:
                img_lq = cv2.imread(str(img_lq_path))
                if img_lq is None:
                    raise FileNotFoundError(f"无法读取图像: {img_lq_path}")
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)
            except Exception as e:
                print(f"无法读取LQ图像 {img_lq_path}: {e}")
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            try:
                img_gt = cv2.imread(str(img_gt_path))
                if img_gt is None:
                    raise FileNotFoundError(f"无法读取图像: {img_gt_path}")
                img_gt = img_gt.astype(np.float32) / 255.
                img_gts.append(img_gt)
            except Exception as e:
                print(f"无法读取GT图像 {img_gt_path}: {e}")
                return self.__getitem__(random.randint(0, len(self) - 1))
        
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, str(img_gt_path))
        
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
        
        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)
        
        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        key = f"{folder}/{frame_files[neighbor_indices[0]]}"
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}
    
    def __len__(self):
        return len(self.keys)