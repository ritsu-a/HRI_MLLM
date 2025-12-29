#!/usr/bin/env python3
"""
动作分类数据集
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MotionClassificationDataset(Dataset):
    """动作分类数据集"""
    
    def __init__(self, data_list, dataset_info_path, num_frames=50, normalize=True):
        """
        Args:
            data_list: 数据列表（从JSON文件加载）
            dataset_info_path: 数据集信息文件路径
            num_frames: 使用的帧数（如果原始数据是100帧，可以下采样到50帧）
            normalize: 是否归一化
        """
        self.data_list = data_list
        self.num_frames = num_frames
        self.normalize = normalize
        
        # 加载数据集信息
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            self.dataset_info = json.load(f)
        
        self.class_to_id = self.dataset_info['class_to_id']
        self.id_to_class = {int(k): v for k, v in self.dataset_info['id_to_class'].items()}
        self.num_classes = len(self.class_to_id)
        
        # 计算均值和标准差（用于归一化）
        if normalize:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """计算数据集的均值和标准差"""
        print("正在计算数据集的均值和标准差...")
        all_data = []
        
        # 采样部分数据来计算统计信息
        sample_size = min(1000, len(self.data_list))
        indices = np.random.choice(len(self.data_list), sample_size, replace=False)
        
        for idx in indices:
            data = np.load(self.data_list[idx]['file_path'])
            data = self._preprocess_data(data)
            all_data.append(data)
        
        all_data = np.stack(all_data)  # (sample_size, num_frames, 491)
        # 计算所有样本、所有帧的均值和标准差
        self.mean = np.mean(all_data, axis=(0, 1))  # (491,)
        self.std = np.std(all_data, axis=(0, 1)) + 1e-8  # (491,)
        
        print(f"均值形状: {self.mean.shape}, 标准差形状: {self.std.shape}")
    
    def _preprocess_data(self, data):
        """
        预处理数据：下采样到指定帧数
        
        Args:
            data: (100, 491) 或 (T, 491) 的numpy数组
        
        Returns:
            (num_frames, 491) 的numpy数组
        """
        original_frames = data.shape[0]
        
        if original_frames == self.num_frames:
            return data
        elif original_frames > self.num_frames:
            # 下采样：均匀采样
            indices = np.linspace(0, original_frames - 1, self.num_frames, dtype=int)
            return data[indices]
        else:
            # 如果帧数不足，进行零填充
            padding = np.zeros((self.num_frames - original_frames, data.shape[1]), dtype=data.dtype)
            return np.vstack([data, padding])
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 加载数据
        data = np.load(item['file_path'])
        data = self._preprocess_data(data)
        
        # 归一化
        if self.normalize:
            # 确保mean和std的形状正确，以便正确广播
            # data: (num_frames, 491), mean/std: (491,)
            data = (data - self.mean) / self.std
        
        # 转换为tensor
        data = torch.FloatTensor(data)  # (num_frames, 491)
        label = torch.LongTensor([item['class_id']])
        
        return {
            'motion': data,
            'label': label.squeeze(),
            'motion_name': item['motion_name'],
            'motion_class': item['motion_class']
        }

