import random
import codecs as cs
import numpy as np
import torch
import os
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .T2M_dataset import Text2MotionDataset
from .DirectMotionDataset import DirectMotionDataset


class MixedMotionDatasetVQ(Text2MotionDataset):
    def __init__(
        self,
        data_root_list,  # 多个数据根目录的列表
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        dataset_weights=None,  # 每个数据集的权重列表
        **kwargs,
    ):
        # 初始化基础参数
        self.data_root_list = data_root_list
        self.dataset_weights = dataset_weights or [1.0] * len(data_root_list)
        self.window_size = win_size
        
        # 存储所有数据集的样本
        self.all_samples = []
        self.all_names = []
        self.all_data_dict = {}
        
        # 为每个数据集加载数据
        for dataset_idx, data_root in enumerate(data_root_list):
            # 临时设置数据根目录
            kwargs_temp = kwargs.copy()
            kwargs_temp['data_root'] = data_root
            
            # 🔧 检查split文件是否存在，如果不存在则使用DirectMotionDataset
            split_file = pjoin(data_root, split + '.txt')
            use_direct_dataset = not os.path.exists(split_file)
            
            if use_direct_dataset:
                # 使用DirectMotionDataset（直接从目录加载所有文件）
                print(f"  Dataset {dataset_idx}: split file not found, using DirectMotionDataset")
                temp_dataset = DirectMotionDataset(
                    split=split,
                    mean=mean,
                    std=std,
                    max_motion_length=max_motion_length,
                    min_motion_length=min_motion_length,
                    unit_length=unit_length,
                    fps=fps,
                    tmpFile=tmpFile,
                    tiny=tiny,
                    debug=debug,
                    **kwargs_temp
                )
            else:
                # 使用Text2MotionDataset（从split文件加载）
                temp_dataset = Text2MotionDataset(
                    split=split,
                    mean=mean,
                    std=std,
                    max_motion_length=max_motion_length,
                    min_motion_length=min_motion_length,
                    unit_length=unit_length,
                    fps=fps,
                    tmpFile=tmpFile,
                    tiny=tiny,
                    debug=debug,
                    **kwargs_temp
                )
            
            # 🔧 检查是否是seg_finger数据集
            is_seg_finger = 'seg_finger' in data_root
            
            # 过滤太短的运动
            valid_names = []
            for name in temp_dataset.name_list:
                motion = temp_dataset.data_dict[name]["motion"]
                
                # 🔧 对于seg_finger，保留所有动作（即使长度小于窗口大小）
                # 最小长度设为8帧，确保可以处理
                if is_seg_finger:
                    if motion.shape[0] >= 8:
                        valid_names.append(name)
                else:
                    # 对于其他数据集，使用原来的逻辑
                    if motion.shape[0] >= self.window_size:
                        valid_names.append(name)
                
                if name in valid_names:
                    # 添加数据集索引前缀以避免名称冲突
                    prefixed_name = f"dataset_{dataset_idx}_{name}"
                    self.all_names.append(prefixed_name)
                    self.all_data_dict[prefixed_name] = {
                        "motion": motion,
                        "length": temp_dataset.data_dict[name]["length"]
                    }
                    self.all_samples.append({
                        "name": prefixed_name,
                        "dataset_idx": dataset_idx,
                        "weight": self.dataset_weights[dataset_idx]
                    })
        
        # 设置基础参数
        self.mean = mean
        self.std = std
        self.name_list = self.all_names
        
        # 创建加权采样器
        self._create_weighted_sampler()
        
        print(f"Mixed dataset loaded:")
        for i, (data_root, weight) in enumerate(zip(data_root_list, self.dataset_weights)):
            dataset_count = sum(1 for s in self.all_samples if s["dataset_idx"] == i)
            print(f"  Dataset {i} ({data_root}): {dataset_count} samples, weight: {weight}")
        print(f"  Total samples: {len(self.all_samples)}")

    def _create_weighted_sampler(self):
        """创建加权采样器"""
        weights = []
        for sample in self.all_samples:
            weights.append(sample["weight"])
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 创建采样概率
        self.sample_probs = torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, item):
        # 使用加权采样选择样本
        if hasattr(self, 'sample_probs'):
            # 加权随机采样
            idx = torch.multinomial(self.sample_probs, 1).item()
        else:
            # 均匀采样作为后备
            idx = item % len(self.all_samples)
        
        sample_info = self.all_samples[idx]
        name = sample_info["name"]
        dataset_idx = sample_info["dataset_idx"]
        
        data = self.all_data_dict[name]
        motion, length = data["motion"], data["length"]
        
        # 🔧 检查是否是seg_finger数据集
        is_seg_finger = 'seg_finger' in self.data_root_list[dataset_idx] if dataset_idx < len(self.data_root_list) else False

        # 🔧 根据动作长度和数据集类型选择处理方式
        if motion.shape[0] < self.window_size:
            # 如果动作长度小于窗口大小，进行padding
            padding_length = self.window_size - motion.shape[0]
            padding = np.zeros((padding_length, motion.shape[1]), dtype=motion.dtype)
            motion = np.concatenate([motion, padding], axis=0)
            length = self.window_size  # 更新长度为窗口大小
        else:
            # 如果动作长度大于等于窗口大小，随机切分
            idx_start = random.randint(0, max(0, motion.shape[0] - self.window_size))
            motion = motion[idx_start:idx_start + self.window_size]
        
        motion = (motion - self.mean) / self.std

        # 返回dataset_idx用于后续的加权损失计算
        return None, motion, length, None, None, None, dataset_idx
