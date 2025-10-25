import random
import codecs as cs
import numpy as np
import torch
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .T2M_dataset import Text2MotionDataset


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
            
            # 创建临时数据集实例来加载数据
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
            
            # 过滤太短的运动
            valid_names = []
            for name in temp_dataset.name_list:
                motion = temp_dataset.data_dict[name]["motion"]
                if motion.shape[0] >= self.window_size:
                    valid_names.append(name)
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

        # 随机选择窗口
        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        motion = (motion - self.mean) / self.std

        # 返回dataset_idx用于后续的加权损失计算
        return None, motion, length, None, None, None, dataset_idx
