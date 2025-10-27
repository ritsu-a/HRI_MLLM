import random
import codecs as cs
import numpy as np
import torch
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .T2M_dataset import Text2MotionDataset


class MixedMotionDatasetVQGrouped(Text2MotionDataset):
    """
    按数据集分组的数据集类
    
    关键特性：
    1. 将不同数据集的样本分开存储
    2. 每个epoch采样时，确保每个batch只包含来自同一数据集的样本
    3. 这样可以避免BEAT（长序列）和SeG（短序列）在同一个batch中导致的padding浪费
    """
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
        dataset_window_sizes=None,  # 🔧 新增：每个数据集使用不同的窗口大小
        **kwargs,
    ):
        # 初始化基础参数
        self.data_root_list = data_root_list
        self.dataset_weights = dataset_weights or [1.0] * len(data_root_list)
        self.window_size = win_size  # 默认窗口大小
        self.dataset_window_sizes = dataset_window_sizes  # 每个数据集的特定窗口大小
        
        # 🔧 关键改进：为每个数据集单独存储样本
        self.dataset_samples = {}  # {dataset_idx: [sample1, sample2, ...]}
        self.dataset_names = {}    # {dataset_idx: [name1, name2, ...]}
        self.dataset_data_dict = {} # {dataset_idx: {name: data}}
        self.all_data_dict = {}
        self.all_names = []
        
        # 为每个数据集加载数据
        for dataset_idx, data_root in enumerate(data_root_list):
            # 初始化该数据集的数据结构
            self.dataset_samples[dataset_idx] = []
            self.dataset_names[dataset_idx] = []
            self.dataset_data_dict[dataset_idx] = {}
            
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
            # 🔧 获取该数据集的特定窗口大小
            dataset_win_size = self.dataset_window_sizes[dataset_idx] if self.dataset_window_sizes and len(self.dataset_window_sizes) > dataset_idx else self.window_size
            
            valid_names = []
            for name in temp_dataset.name_list:
                motion = temp_dataset.data_dict[name]["motion"]
                
                # 🔧 判断是否有效
                is_valid = False
                if dataset_win_size == 1:
                    # 任意长度模式：只要不是太短就可以
                    if motion.shape[0] >= min_motion_length:
                        is_valid = True
                else:
                    # 固定窗口模式：使用该数据集特定的窗口大小
                    if motion.shape[0] >= dataset_win_size:
                        is_valid = True
                
                # 如果有效，则存储样本
                if is_valid:
                    # 添加数据集索引前缀以避免名称冲突
                    prefixed_name = f"dataset_{dataset_idx}_{name}"
                    
                    # 存储到数据集特定的数据结构
                    self.dataset_names[dataset_idx].append(prefixed_name)
                    self.dataset_data_dict[dataset_idx][prefixed_name] = {
                        "motion": motion,
                        "length": temp_dataset.data_dict[name]["length"],
                        "window_size": dataset_win_size  # 🔧 存储该样本使用的窗口大小
                    }
                    self.dataset_samples[dataset_idx].append({
                        "name": prefixed_name,
                        "dataset_idx": dataset_idx,
                        "weight": self.dataset_weights[dataset_idx],
                        "window_size": dataset_win_size  # 🔧 存储窗口大小
                    })
                    
                    # 同时存储到全局数据结构（用于getitem）
                    self.all_names.append(prefixed_name)
                    self.all_data_dict[prefixed_name] = {
                        "motion": motion,
                        "length": temp_dataset.data_dict[name]["length"],
                        "window_size": dataset_win_size  # 🔧 存储窗口大小
                    }
        
        # 设置基础参数
        self.mean = mean
        self.std = std
        self.name_list = self.all_names
        
        print(f"Mixed dataset loaded (Grouped Mode):")
        for i, (data_root, weight) in enumerate(zip(data_root_list, self.dataset_weights)):
            dataset_count = len(self.dataset_samples[i])
            print(f"  Dataset {i} ({data_root}): {dataset_count} samples, weight: {weight}")
        print(f"  Total samples: {len(self.all_names)}")

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        """获取单个样本"""
        name = self.all_names[idx]
        data = self.all_data_dict[name]
        motion, length = data["motion"], data["length"]
        window_size = data.get("window_size", self.window_size)  # 🔧 获取该样本的窗口大小
        
        # 从name中提取dataset_idx
        dataset_idx = int(name.split('_')[1])

        # 🔧 根据该样本的窗口大小进行切分
        if window_size == 1:
            # 任意长度模式：直接使用完整序列
            motion = motion
        else:
            # 使用该样本特定的窗口大小进行切分
            idx_start = random.randint(0, max(0, motion.shape[0] - window_size))
            motion = motion[idx_start:idx_start + window_size]
        
        # 归一化
        motion = (motion - self.mean) / self.std

        # 返回dataset_idx用于后续的加权损失计算
        return None, motion, length, None, None, None, dataset_idx
    
    def get_dataset_samples(self, dataset_idx):
        """获取特定数据集的所有样本索引"""
        return [idx for idx, name in enumerate(self.all_names) if name.startswith(f"dataset_{dataset_idx}_")]


class GroupedBatchSampler:
    """
    按数据集分组的Batch采样器
    
    确保同一个batch中只包含来自同一个数据集的样本
    """
    def __init__(self, dataset: MixedMotionDatasetVQGrouped, batch_size, dataset_weights, num_batches_per_epoch=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_weights = dataset_weights
        
        # 计算每个数据集应该采样多少个batch
        num_datasets = len(dataset_weights)
        
        # 如果没有指定，根据数据集大小自动计算
        if num_batches_per_epoch is None:
            # 每个数据集至少有1个batch，总共至少num_datasets个batch
            num_batches_per_epoch = num_datasets
        else:
            # 确保至少有num_datasets个batch，以便每个数据集至少有1个batch
            num_batches_per_epoch = max(num_batches_per_epoch, num_datasets)
        
        self.num_batches_per_epoch = num_batches_per_epoch
        
        # 为每个数据集分配batch数量（根据权重）
        self.batches_per_dataset = {}
        total_weight = sum(dataset_weights)
        
        for dataset_idx in range(num_datasets):
            # 根据权重分配batch数量
            weight_ratio = dataset_weights[dataset_idx] / total_weight
            num_batches = max(1, int(num_batches_per_epoch * weight_ratio))
            self.batches_per_dataset[dataset_idx] = num_batches
        
        # 计算总的batch数量
        self.total_batches = sum(self.batches_per_dataset.values())
        
        print(f"GroupedBatchSampler initialized:")
        for dataset_idx, num_batches in self.batches_per_dataset.items():
            dataset_name = self.dataset.data_root_list[dataset_idx].split('/')[-1]
            print(f"  Dataset {dataset_idx} ({dataset_name}): {num_batches} batches/epoch")
        print(f"  Total: {self.total_batches} batches/epoch")
    
    def __len__(self):
        return self.total_batches
    
    def __iter__(self):
        """生成batch索引"""
        # 为每个数据集生成batch索引
        batches = []
        
        for dataset_idx in range(len(self.dataset_weights)):
            # 获取该数据集的所有样本索引
            dataset_samples = self.dataset.get_dataset_samples(dataset_idx)
            
            if len(dataset_samples) == 0:
                continue
            
            # 随机打乱
            random.shuffle(dataset_samples)
            
            # 生成batch
            num_batches = self.batches_per_dataset[dataset_idx]
            for i in range(num_batches):
                # 计算该batch的样本
                batch_indices = []
                for j in range(self.batch_size):
                    # 循环使用该数据集的样本
                    sample_idx = (i * self.batch_size + j) % len(dataset_samples)
                    batch_indices.append(dataset_samples[sample_idx])
                
                batches.append(batch_indices)
        
        # 随机打乱所有batch的顺序
        random.shuffle(batches)
        
        # 返回batch索引
        for batch_indices in batches:
            yield batch_indices
