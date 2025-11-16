import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, Sampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
import json
import random
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from HRI_mllm.datasets.BEATAudioMotionDataset import BEATAudioMotionDataset
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

class JSONLAudioMotionDataset(Dataset):
    """从JSONL文件加载音频-动作数据的Dataset，支持动作类别标记"""
    
    def __init__(self, jsonl_path, config, extract_action_class=False, dataset_name=""):
        self.config = config
        self.jsonl_path = jsonl_path
        self.dataset_name = dataset_name
        self.current_epoch = 0
        self.source_id = 0 if dataset_name == "BEAT" else 1
        self.samples = []
        self.action_classes = []  # 存储每个样本的动作类别
        self.seq_lengths = []  # 存储每个样本的序列长度
        self.stats = {'total_sequences': 0, 'generated_samples': 0, 'max_length': 0}
        self.interleave_audios, self.interleave_motions = config.interleave_ratio
        self.SEQ_PAD_TOKEN = config.pad_token_id
        self.extract_action_class = extract_action_class
        self.concat_samples = getattr(config, 'concat_samples', False)
        self.target_concat_length = getattr(config, 'target_concat_length', config.max_seq_length)
        self.sliding_window_step = getattr(config, 'sliding_window_step', config.max_seq_length)
        self.raw_samples = []
        
        print(f"Loading JSONL file: {jsonl_path}")
        
        # 读取JSONL文件
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = self._extract_sample_from_json(data)
                    if sample:
                        self.raw_samples.append(sample)
                except Exception as e:
                    print(f"Error processing line {line_num} in {jsonl_path}: {e}")
                    continue
        
        # 首次构建样本（不指定epoch）
        self._rebuild_samples(epoch=None)
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
        if self.extract_action_class:
            action_counts = defaultdict(int)
            for ac in self.action_classes:
                if ac:
                    action_counts[ac] += 1
            print(f"Action class distribution: {dict(action_counts)}")

    def _rebuild_samples(self, epoch: Optional[int]):
        """按当前配置重建样本；若启用拼接，则每次重建都会重新随机化顺序"""
        # 清空现有样本与统计
        self.samples = []
        self.action_classes = []
        self.seq_lengths = []
        self.stats = {'total_sequences': 0, 'generated_samples': 0, 'max_length': 0}
        
        if not self.concat_samples:
            for sample in self.raw_samples:
                self._add_sequence(sample['audio_tokens'], sample['motion_tokens'], sample['motion_labels'], sample['action_class'])
            return
        
        # 可重复的随机：若提供了epoch，则用其作为种子扰动
        rng = random.Random()
        if epoch is not None:
            rng.seed(epoch + 1337)
        shuffled = self.raw_samples[:]
        rng.shuffle(shuffled)
        
        buffer_audio, buffer_motion, buffer_labels = [], [], []
        for sample in shuffled:
            audio_tokens = sample['audio_tokens']
            motion_tokens = sample['motion_tokens']
            motion_labels = sample['motion_labels']
            
            motion_offset = len(buffer_motion)
            buffer_audio.extend(audio_tokens)
            buffer_motion.extend(motion_tokens)
            
            if motion_labels:
                for label in motion_labels:
                    new_label = dict(label)
                    if 'start_token_index' in new_label:
                        new_label['start_token_index'] += motion_offset
                    if 'end_token_index' in new_label:
                        new_label['end_token_index'] += motion_offset
                    buffer_labels.append(new_label)
            
            if len(buffer_audio) + len(buffer_motion) >= self.target_concat_length:
                self._add_sequence(buffer_audio, buffer_motion, buffer_labels, None)
                buffer_audio, buffer_motion, buffer_labels = [], [], []
        
        if buffer_audio and buffer_motion:
            self._add_sequence(buffer_audio, buffer_motion, buffer_labels, None)

    def set_epoch(self, epoch: int):
        """当启用拼接时，每个epoch重建一次拼接顺序以制造新样本排列"""
        self.current_epoch = epoch
        if self.concat_samples:
            self._rebuild_samples(epoch=epoch)

    def _extract_sample_from_json(self, data):
        audio_tokens = None
        motion_tokens = None
        motion_labels = data.get('motion_labels')
        action_class = None
        
        for msg in data.get('conversation', []):
            if msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                audio_tokens = msg['audio_tokens']
                if self.extract_action_class:
                    audio_path = msg.get('content', '')
                    if '/' in audio_path:
                        filename = audio_path.split('/')[-1]
                        if '_' in filename:
                            parts = filename.split('_')
                            if len(parts) >= 2:
                                action_class = f"{parts[0]}_{parts[1]}"
            elif msg.get('message_type') == 'audio_motion' and 'motion_tokens' in msg:
                motion_tokens = msg['motion_tokens']
        
        if audio_tokens is None or motion_tokens is None:
            return None
        
        audio_tokens = self._ensure_list(audio_tokens)
        motion_tokens = self._ensure_list(motion_tokens)
        
        return {
            'audio_tokens': audio_tokens,
            'motion_tokens': motion_tokens,
            'motion_labels': motion_labels,
            'action_class': action_class,
        }

    def _ensure_list(self, tokens):
        if isinstance(tokens, torch.Tensor):
            return tokens.clone().detach().tolist()
        if isinstance(tokens, np.ndarray):
            return tokens.tolist()
        return list(tokens)

    def _add_sequence(self, audio_tokens, motion_tokens, motion_labels, action_class):
        if not audio_tokens or not motion_tokens:
            return
        
        gesture_start_indices = set()
        gesture_end_indices = set()
        if motion_labels:
            for label in motion_labels:
                if 'start_token_index' in label:
                    gesture_start_indices.add(label['start_token_index'])
                if 'end_token_index' in label:
                    gesture_end_indices.add(label['end_token_index'])
        
        full_sequence = []
        token_types = []
        motion_token_idx = 0
        
        for i in range(len(audio_tokens)):
            full_sequence.append(int(audio_tokens[i]))
            token_types.append(0)
            
            if (i + 1) % self.interleave_audios == 0:
                motion_block_size = self.interleave_motions
                for _ in range(motion_block_size):
                    if motion_token_idx < len(motion_tokens):
                        if motion_token_idx in gesture_start_indices:
                            full_sequence.append(self.config.gesture_start_token_id)
                            token_types.append(1)
                            full_sequence.append(self.config.audio_gesture_start_token_id)
                            token_types.append(0)
                        
                        full_sequence.append(int(motion_tokens[motion_token_idx]))
                        token_types.append(1)
                        
                        if motion_token_idx in gesture_end_indices:
                            full_sequence.append(self.config.gesture_end_token_id)
                            token_types.append(1)
                            full_sequence.append(self.config.audio_gesture_end_token_id)
                            token_types.append(0)
                        
                        motion_token_idx += 1
        
        while motion_token_idx < len(motion_tokens):
            if motion_token_idx in gesture_start_indices:
                full_sequence.append(self.config.gesture_start_token_id)
                token_types.append(1)
                full_sequence.append(self.config.audio_gesture_start_token_id)
                token_types.append(0)
            
            full_sequence.append(int(motion_tokens[motion_token_idx]))
            token_types.append(1)
            
            if motion_token_idx in gesture_end_indices:
                full_sequence.append(self.config.gesture_end_token_id)
                token_types.append(1)
                full_sequence.append(self.config.audio_gesture_end_token_id)
                token_types.append(0)
            
            motion_token_idx += 1
        
        self.apply_sliding_window(full_sequence, token_types, action_class)
        self.stats['total_sequences'] += 1
    
    def apply_sliding_window(self, full_seq, token_types, action_class=None):
        """（已移除滑窗）仅生成单一样本：将序列截断到 max_seq_length"""
        max_len = self.config.max_seq_length
        sub_seq = full_seq[:max_len]
        sub_types = token_types[:max_len]
        mask = [1 if t == 1 else 0 for t in sub_types]
        
        padded_seq = sub_seq + [self.SEQ_PAD_TOKEN] * (max_len - len(sub_seq))
        padded_mask = mask + [0] * (max_len - len(mask))
        
        sample_idx = len(self.samples)
        seq_length = len(sub_seq)
        self.samples.append({
            'tokens': torch.tensor(padded_seq),
            'mask': torch.tensor(padded_mask),
            'seq_length': seq_length,
            'source_id': self.source_id
        })
        self.action_classes.append(action_class)
        self.seq_lengths.append(seq_length)
        
        self.stats['generated_samples'] += 1
        self.stats['max_length'] = max(self.stats['max_length'], seq_length)
        
        return [sample_idx]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_action_class(self, idx):
        """获取样本的动作类别"""
        return self.action_classes[idx] if idx < len(self.action_classes) else None
    
    def get_seq_length(self, idx):
        """获取样本的序列长度"""
        return self.seq_lengths[idx] if idx < len(self.seq_lengths) else 0


class WeightedDatasetSampler(Sampler):
    """根据数据集权重进行加权采样的Sampler"""
    
    def __init__(self, datasets: List[Dataset], dataset_weights: List[float], 
                 num_samples: Optional[int] = None, replacement: bool = True, shuffle: bool = True,
                 rank: int = 0, world_size: int = 1):
        """
        Args:
            datasets: 数据集列表
            dataset_weights: 每个数据集的权重列表
            num_samples: 每个epoch采样的总样本数（如果None，则使用所有数据集的总样本数）
            replacement: 是否允许有放回采样
            shuffle: 是否打乱顺序
            rank: 当前进程的rank（用于分布式训练）
            world_size: 总进程数（用于分布式训练）
        """
        self.datasets = datasets
        self.dataset_weights = dataset_weights
        self.replacement = replacement
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        
        # 计算每个数据集的累积大小（用于ConcatDataset）
        self.cumulative_sizes = [0]
        for dataset in datasets:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(dataset))
        
        # 为每个样本分配权重
        self.weights = []
        for dataset_idx, dataset in enumerate(datasets):
            weight = dataset_weights[dataset_idx]
            # 归一化权重：权重除以数据集大小，使得每个数据集的期望采样数 = 权重 * num_samples
            normalized_weight = weight / len(dataset) if len(dataset) > 0 else 0
            self.weights.extend([normalized_weight] * len(dataset))
        
        self.weights = torch.tensor(self.weights, dtype=torch.double)
        
        # 确定每个epoch的采样数量
        if num_samples is None:
            # 默认使用所有数据集的总样本数
            total_samples = sum(len(d) for d in datasets)
            # 在分布式训练中，每个进程只采样一部分
            self.num_samples = total_samples // world_size
        else:
            self.num_samples = num_samples // world_size
        
        if rank == 0:
            print(f"WeightedDatasetSampler initialized:")
            for idx, (dataset, weight) in enumerate(zip(datasets, dataset_weights)):
                print(f"  Dataset {idx}: {len(dataset)} samples, weight: {weight:.3f}")
            print(f"  Total samples per epoch (per process): {self.num_samples}")
            print(f"  Replacement: {self.replacement}")
    
    def __iter__(self):
        # 使用加权随机采样
        indices = torch.multinomial(self.weights, self.num_samples, replacement=self.replacement).tolist()
        
        if self.shuffle:
            random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """设置epoch，用于DistributedSampler兼容性"""
        # 设置随机种子以确保每个epoch的采样不同
        random.seed(epoch)
        torch.manual_seed(epoch)


class BalancedActionClassSampler(Sampler):
    """按动作类别均匀采样的Sampler，每个epoch重新采样"""
    
    def __init__(self, dataset: JSONLAudioMotionDataset, samples_per_class: Optional[int] = None, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        # 按动作类别分组样本索引
        self.action_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            action_class = dataset.get_action_class(idx)
            if action_class:
                self.action_to_indices[action_class].append(idx)
        
        # 确定每个类别的采样数量
        if samples_per_class is None:
            # 使用最小类别的样本数量
            self.samples_per_class = min(len(indices) for indices in self.action_to_indices.values() if len(indices) > 0)
        else:
            self.samples_per_class = samples_per_class
        
        # 计算总样本数
        num_classes = len(self.action_to_indices)
        self.num_samples = self.samples_per_class * num_classes
        
        print(f"BalancedActionClassSampler initialized:")
        print(f"  Original samples: {len(dataset)}")
        print(f"  Balanced samples per epoch: {self.num_samples}")
        for action_class, indices in self.action_to_indices.items():
            print(f"    {action_class}: {len(indices)} -> {self.samples_per_class} per epoch")
    
    def __iter__(self):
        # 每个epoch重新生成平衡后的样本索引
        balanced_indices = []
        for action_class, indices in self.action_to_indices.items():
            # 随机采样指定数量的样本
            if len(indices) >= self.samples_per_class:
                sampled = random.sample(indices, self.samples_per_class)
            else:
                # 如果样本不足，使用有放回采样
                sampled = random.choices(indices, k=self.samples_per_class)
            balanced_indices.extend(sampled)
        
        # 打乱顺序
        if self.shuffle:
            random.shuffle(balanced_indices)
        
        return iter(balanced_indices)
    
    def __len__(self):
        return self.num_samples


class LengthBasedBatchSampler(BatchSampler):
    """按序列长度分组的BatchSampler，提高GPU利用率，支持加权采样"""
    
    def __init__(self, dataset: Dataset, batch_size: int, length_bins: List[Tuple[int, int]],
                 shuffle: bool = True, datasets: Optional[List[Dataset]] = None, 
                 dataset_weights: Optional[List[float]] = None):
        """
        Args:
            dataset: 数据集（可能是ConcatDataset）
            batch_size: 批次大小
            length_bins: 长度分组，例如 [(0, 512), (512, 1024), (1024, 2048), (2048, 4096)]
            shuffle: 是否打乱
            datasets: 原始数据集列表（用于判断样本来源）
            dataset_weights: 数据集权重列表（用于加权采样）
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.length_bins = length_bins
        self.shuffle = shuffle
        self.datasets = datasets
        self.dataset_weights = dataset_weights
        
        # 如果是ConcatDataset，计算累积大小
        if isinstance(dataset, ConcatDataset) and datasets is not None:
            self.cumulative_sizes = [0]
            for d in datasets:
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(d))
        else:
            self.cumulative_sizes = None
        
        # 按长度分组样本索引，并记录每个样本的权重
        self.bin_to_indices = defaultdict(list)
        self.bin_to_weights = defaultdict(list)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            seq_length = sample['seq_length']
            bin_idx = self._find_bin(seq_length)
            self.bin_to_indices[bin_idx].append(idx)
            
            # 计算样本权重
            if self.cumulative_sizes and dataset_weights:
                # 找到样本来自哪个数据集
                dataset_idx = self._find_dataset_idx(idx)
                weight = dataset_weights[dataset_idx] if dataset_idx < len(dataset_weights) else 1.0
                self.bin_to_weights[bin_idx].append(weight)
            else:
                self.bin_to_weights[bin_idx].append(1.0)
        
        # 检查是否使用加权采样
        self.use_weighted = (dataset_weights is not None and 
                           len(datasets) > 1 and
                           len(set(dataset_weights)) > 1)
        
        if local_rank == 0:
            print(f"LengthBasedBatchSampler initialized:")
            if dataset_weights:
                print(f"  Dataset weights: {dataset_weights}")
                print(f"  Using weighted sampling: {self.use_weighted}")
            for bin_idx in range(len(length_bins)):
                bin_range = length_bins[bin_idx]
                indices = self.bin_to_indices[bin_idx]
                if len(indices) > 0:
                    print(f"  Bin {bin_idx} [{bin_range[0]}, {bin_range[1]}): {len(indices)} samples")
    
    def _find_dataset_idx(self, global_idx: int) -> int:
        """找到全局索引对应的数据集索引"""
        if self.cumulative_sizes is None:
            return 0
        for i in range(len(self.cumulative_sizes) - 1):
            if self.cumulative_sizes[i] <= global_idx < self.cumulative_sizes[i + 1]:
                return i
        return len(self.cumulative_sizes) - 2
    
    def _find_bin(self, length):
        """找到长度对应的分组索引"""
        for i, (min_len, max_len) in enumerate(self.length_bins):
            if min_len <= length < max_len:
                return i
        return len(self.length_bins) - 1
    
    def __iter__(self):
        """生成batch，每个epoch重新生成以支持加权采样"""
        batches = []
        
        for bin_idx in range(len(self.length_bins)):
            indices = self.bin_to_indices[bin_idx]
            weights = self.bin_to_weights[bin_idx]
            
            if len(indices) == 0:
                continue
            
            if self.use_weighted:
                # 加权采样：根据权重直接限制每个数据集的样本数
                # 首先按数据集分组
                dataset_to_indices = defaultdict(list)
                
                for idx in indices:
                    dataset_idx = self._find_dataset_idx(idx)
                    dataset_to_indices[dataset_idx].append(idx)
                
                # 根据权重计算每个数据集应该采样多少个样本
                # 使用最大权重作为基准，其他数据集按比例减少
                max_weight = max(self.dataset_weights)
                total_weight = sum(self.dataset_weights)
                sampled_indices_all = []
                
                # 统计信息（仅第一个bin打印）
                if bin_idx == 0 and local_rank == 0:
                    print(f"\n📊 Weighted sampling in bin {bin_idx}:")
                
                for dataset_idx, dataset_indices in dataset_to_indices.items():
                    weight = self.dataset_weights[dataset_idx]
                    # 计算采样比例：权重相对于最大权值的比例
                    # 例如：权重[1.0, 0.1]，max_weight=1.0
                    # BEAT: 1.0/1.0 = 1.0 (采样100%)
                    # single_motion: 0.1/1.0 = 0.1 (采样10%)
                    sample_ratio = weight / max_weight
                    num_samples = max(1, int(len(dataset_indices) * sample_ratio))
                    # 但不超过实际数据集大小
                    num_samples = min(num_samples, len(dataset_indices))
                    
                    # 随机采样（无放回，如果样本数足够）
                    if num_samples >= len(dataset_indices):
                        sampled = dataset_indices
                    else:
                        sampled = random.sample(dataset_indices, num_samples)
                    sampled_indices_all.extend(sampled)
                    
                    # 打印统计信息（仅第一个bin）
                    if bin_idx == 0 and local_rank == 0:
                        print(f"  Dataset {dataset_idx}: {len(dataset_indices)} -> {len(sampled)} samples (ratio={sample_ratio:.3f}, weight={weight})")
                
                # 打乱采样后的样本
                if self.shuffle:
                    random.shuffle(sampled_indices_all)
                
                # 生成batch
                for i in range(0, len(sampled_indices_all), self.batch_size):
                    batch_indices = sampled_indices_all[i:i + self.batch_size]
                    if len(batch_indices) > 0:
                        batches.append(batch_indices)
            else:
                # 标准采样
                indices_list = list(indices)
                if self.shuffle:
                    random.shuffle(indices_list)
                
                # 生成batch
                for i in range(0, len(indices_list), self.batch_size):
                    batch_indices = indices_list[i:i + self.batch_size]
                    if len(batch_indices) > 0:
                        batches.append(batch_indices)
        
        if self.shuffle:
            random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self):
        """返回估计的batch数量"""
        if self.use_weighted:
            # 加权采样时，根据权重计算估计的batch数量
            max_weight = max(self.dataset_weights) if self.dataset_weights else 1.0
            total_samples = 0
            
            # 估算每个bin的采样样本数
            for bin_idx, indices in self.bin_to_indices.items():
                if len(indices) == 0:
                    continue
                
                # 按数据集分组估算
                dataset_to_indices = defaultdict(list)
                for idx in indices:
                    dataset_idx = self._find_dataset_idx(idx)
                    dataset_to_indices[dataset_idx].append(idx)
                
                # 计算加权后的总样本数
                for dataset_idx, dataset_indices in dataset_to_indices.items():
                    weight = self.dataset_weights[dataset_idx] if dataset_idx < len(self.dataset_weights) else 1.0
                    sample_ratio = weight / max_weight
                    num_samples = int(len(dataset_indices) * sample_ratio)
                    total_samples += num_samples
            
            return max(1, total_samples // self.batch_size)
        else:
            # 标准采样时，计算实际的batch数量
            total_samples = sum(len(indices) for indices in self.bin_to_indices.values())
            return max(1, (total_samples + self.batch_size - 1) // self.batch_size)


class DynamicCollateFn:
    """动态collate函数，按batch内最长序列padding"""
    
    def __init__(self, pad_token_id: int, max_seq_length: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
    
    def __call__(self, batch):
        tokens = [item['tokens'] for item in batch]
        masks = [item['mask'] for item in batch]
        seq_lengths = [item['seq_length'] for item in batch]
        
        # 找到batch内的最大长度
        max_len = max(seq_lengths)
        if self.max_seq_length is not None:
            max_len = min(max_len, self.max_seq_length)
        
        # 对每个样本进行padding或截断
        batch_tokens = []
        batch_masks = []
        for t, m, l in zip(tokens, masks, seq_lengths):
            # 确保使用实际的有效长度（不超过tensor的实际长度）
            actual_len = min(l, len(t))
            
            if actual_len > max_len:
                # 截断到max_len
                t = t[:max_len]
                m = m[:max_len]
            elif actual_len < max_len:
                # 取有效部分并padding
                pad_len = max_len - actual_len
                pad_tokens = torch.full((pad_len,), self.pad_token_id, dtype=t.dtype)
                pad_mask = torch.zeros(pad_len, dtype=m.dtype)
                t = torch.cat([t[:actual_len], pad_tokens])
                m = torch.cat([m[:actual_len], pad_mask])
            else:
                # actual_len == max_len，直接取前max_len个元素
                t = t[:max_len]
                m = m[:max_len]
            
            # 确保所有tensor长度都是max_len
            assert len(t) == max_len, f"Token length mismatch: expected {max_len}, got {len(t)}"
            assert len(m) == max_len, f"Mask length mismatch: expected {max_len}, got {len(m)}"
            
            batch_tokens.append(t)
            batch_masks.append(m)
        
        return {
            'tokens': torch.stack(batch_tokens),
            'mask': torch.stack(batch_masks),
            'lengths': torch.tensor(seq_lengths)
        }


import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Finetune Motion Adaptor on BEAT and single_motion')
parser.add_argument('--resume_from', type=str, default=None,
                   help='Checkpoint path to resume from')
parser.add_argument('--epochs', type=int, default=300,
                   help='Number of epochs to train')
parser.add_argument('--use_dynamic_batching', action='store_true',
                   help='Use dynamic batching based on sequence length')
parser.add_argument('--use_balanced_sampling', action='store_true',
                   help='Use balanced sampling for single_motion action classes')
parser.add_argument('--use_weighted_datasets', action='store_true',
                   help='Use weighted dataset sampler to balance BEAT and single_motion per-epoch sampling')
parser.add_argument('--batch_size', type=int, default=64,
                   help='Batch size for BEAT dataset')
parser.add_argument('--max_seq_length', type=int, default=4096,
                   help='Maximum sequence length for BEAT dataset')
parser.add_argument('--dataset_weights', type=float, nargs='+', default=[1.0, 1.0],
                   help='Dataset weights for BEAT and single_motion')
# single_motion 专用参数
parser.add_argument('--single_motion_batch_size', type=int, default=256,
                   help='Batch size for single_motion dataset')
parser.add_argument('--single_motion_max_seq_length', type=int, default=4096,
                   help='Maximum sequence length for single_motion dataset')
args = parser.parse_args()

exp_name = "kimi_audio_motion_gpt2_brainco_finetune_beat_single_motion"
os.makedirs(os.path.join("output_disk0/motion_adaptor_v11", exp_name), exist_ok=True)
os.makedirs(os.path.join("output_disk0/motion_adaptor_v11", exp_name, "checkpoints"), exist_ok=True)

os.environ["WANDB_MODE"] = "offline"

# 初始化分布式训练环境
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

# 检查是否在分布式环境中运行
if 'RANK' in os.environ:
    local_rank, world_size = setup_distributed()
else:
    local_rank, world_size = 0, 1
    torch.cuda.set_device(0)

# JSONL文件路径
all_jsonl_files = {
    "BEAT": "/root/workspace/HRI_MLLM/data/BEAT_v2_1110_tokens.jsonl",
    "single_motion": "/root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens.train.jsonl",
}

jsonl_files = list(all_jsonl_files.values())

if local_rank == 0:
    print(f"Using datasets: BEAT, single_motion")
    print(f"JSONL files: {jsonl_files}")

# 检查文件是否存在
for jsonl_file in jsonl_files:
    if not os.path.exists(jsonl_file):
        if local_rank == 0:
            print(f"❌ JSONL file not found: {jsonl_file}")
        exit(1)

# 只在主进程初始化Weights & Biases
if local_rank == 0:
    wandb.init(
        project="audio-motion-BEAT-gpt2-adaptor-finetune",
        config={
            "jsonl_files": jsonl_files,
            "audio_vocab_size": 16384,
            "motion_vocab_size": 512*2,
            "total_vocab_size": 512*2 + 10,
            "max_seq_length": args.max_seq_length,
            "single_motion_max_seq_length": args.single_motion_max_seq_length,
            "min_seq_length": 128,
            "batch_size": args.batch_size,
            "single_motion_batch_size": args.single_motion_batch_size,
            "learning_rate": 1e-4,
            "epochs": args.epochs,
            "sliding_window_step": 32,
            "pad_token_id": 512*2 + 1,
            "gesture_start_token_id": 512*2 + 2,
            "audio_gesture_start_token_id": 512*2 + 3,
            "gesture_end_token_id": 512*2 + 4,
            "audio_gesture_end_token_id": 512*2 + 5,
            "interleave_ratio": [1, 1],
            "exp_name": exp_name,
            "use_dynamic_batching": args.use_dynamic_batching,
            "use_balanced_sampling": args.use_balanced_sampling,
            "dataset_weights": args.dataset_weights,
        }
    )
    config = wandb.config
else:
    config = type('Config', (), {
        "jsonl_files": jsonl_files,
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512*2,
        "total_vocab_size": 512*2 + 10,
        "max_seq_length": args.max_seq_length,
        "single_motion_max_seq_length": args.single_motion_max_seq_length,
        "min_seq_length": 128,
        "batch_size": args.batch_size,
        "single_motion_batch_size": args.single_motion_batch_size,
        "learning_rate": 1e-4,
        "epochs": args.epochs,
        "sliding_window_step": 32,
        "pad_token_id": 512*2 + 1,
        "gesture_start_token_id": 512*2 + 2,
        "audio_gesture_start_token_id": 512*2 + 3,
        "gesture_end_token_id": 512*2 + 4,
        "audio_gesture_end_token_id": 512*2 + 5,
        "interleave_ratio": [1, 1],
        "exp_name": exp_name,
        "use_dynamic_batching": args.use_dynamic_batching,
        "use_balanced_sampling": args.use_balanced_sampling,
        "dataset_weights": args.dataset_weights,
    })()

# 创建模型
# 注意：n_positions 需要与 checkpoint 保持一致（4096），否则无法加载权重
# 训练时使用较短序列（如 256/1024）不影响模型结构（n_positions 仍为 4096）
model_config = GPT2Config(
    vocab_size=config.total_vocab_size,
    n_positions=4096,  # 保持与原 checkpoint 一致
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
model_config.audio_gesture_start_token_id = config.audio_gesture_start_token_id
model_config.audio_gesture_end_token_id = config.audio_gesture_end_token_id
model_config.gesture_start_token_id = config.gesture_start_token_id
model_config.gesture_end_token_id = config.gesture_end_token_id
model = MixedInputGPT2(model_config)

device = torch.device(f'cuda:{local_rank}')
model.to(device)

# 如果提供了resume_from，加载checkpoint
start_epoch = 0
if args.resume_from and os.path.exists(args.resume_from):
    if local_rank == 0:
        print(f"\n{'='*80}")
        print(f"🔄 Resuming from checkpoint: {args.resume_from}")
        print(f"{'='*80}")
    
    checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    if local_rank == 0:
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"   Resuming from epoch {start_epoch}")
        print(f"{'='*80}\n")
else:
    if local_rank == 0:
        print("\nStarting finetuning from pretrained model\n")

# 只在分布式模式下使用DDP
if world_size > 1:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

if local_rank == 0:
    model_to_watch = model.module if world_size > 1 else model
    wandb.watch(model_to_watch, log="parameters", log_freq=100)

# 创建数据集配置：为不同数据集使用不同的max_seq_length
beat_config = type('Config', (), {
    "jsonl_files": config.jsonl_files,
    "audio_vocab_size": config.audio_vocab_size,
    "motion_vocab_size": config.motion_vocab_size,
    "total_vocab_size": config.total_vocab_size,
    "max_seq_length": args.max_seq_length,  # 统一使用运行参数
    "min_seq_length": config.min_seq_length,
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "epochs": config.epochs,
    "sliding_window_step": config.sliding_window_step,
    "pad_token_id": config.pad_token_id,
    "gesture_start_token_id": config.gesture_start_token_id,
    "audio_gesture_start_token_id": config.audio_gesture_start_token_id,
    "gesture_end_token_id": config.gesture_end_token_id,
    "audio_gesture_end_token_id": config.audio_gesture_end_token_id,
    "interleave_ratio": config.interleave_ratio,
    "exp_name": config.exp_name,
    "concat_samples": False,
    "target_concat_length": args.max_seq_length,
})()

single_motion_config = type('Config', (), {
    "jsonl_files": config.jsonl_files,
    "audio_vocab_size": config.audio_vocab_size,
    "motion_vocab_size": config.motion_vocab_size,
    "total_vocab_size": config.total_vocab_size,
    "max_seq_length": args.single_motion_max_seq_length,  # single_motion使用运行参数
    "min_seq_length": config.min_seq_length,
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "epochs": config.epochs,
    "sliding_window_step": config.sliding_window_step,
    "pad_token_id": config.pad_token_id,
    "gesture_start_token_id": config.gesture_start_token_id,
    "audio_gesture_start_token_id": config.audio_gesture_start_token_id,
    "gesture_end_token_id": config.gesture_end_token_id,
    "audio_gesture_end_token_id": config.audio_gesture_end_token_id,
    "interleave_ratio": config.interleave_ratio,
    "exp_name": config.exp_name,
    "concat_samples": True,
    "target_concat_length": 3000,
})()

# 创建独立的数据集（beat和single_motion）
beat_dataset = None
single_motion_dataset = None

for idx, jsonl_path in enumerate(config.jsonl_files):
    if os.path.exists(jsonl_path):
        dataset_name = list(all_jsonl_files.keys())[idx]
        extract_action_class = (dataset_name == "single_motion") and args.use_balanced_sampling
        
        # 根据数据集选择配置
        if dataset_name == "BEAT":
            dataset_config = beat_config
        else:
            dataset_config = single_motion_config
        
        dataset = JSONLAudioMotionDataset(jsonl_path, dataset_config, extract_action_class=extract_action_class, dataset_name=dataset_name)
        
        if dataset_name == "BEAT":
            beat_dataset = dataset
            if local_rank == 0:
                print(f"Loaded {len(dataset)} samples from {jsonl_path} ({dataset_name}) [max_seq_length={dataset_config.max_seq_length}]")
        elif dataset_name == "single_motion":
            single_motion_dataset = dataset
            if local_rank == 0:
                print(f"Loaded {len(dataset)} samples from {jsonl_path} ({dataset_name}) [max_seq_length={dataset_config.max_seq_length}]")
    else:
        if local_rank == 0:
            print(f"Warning: {jsonl_path} not found, skipping...")

if beat_dataset is None or single_motion_dataset is None:
    if local_rank == 0:
        print("❌ Both BEAT and single_motion datasets are required!")
    exit(1)

# 创建single_motion的子采样Dataset
class SubsampledDataset(Dataset):
    """每个epoch随机采样部分数据的Dataset"""
    def __init__(self, base_dataset, sample_ratio=0.05):
        self.base_dataset = base_dataset
        self.sample_ratio = sample_ratio
        self.epoch = 0
        self._resample()
    
    def _resample(self):
        """重新采样"""
        total_size = len(self.base_dataset)
        sample_size = int(total_size * self.sample_ratio)
        # 使用epoch作为随机种子，确保每个epoch的采样不同
        random.seed(self.epoch)
        self.indices = random.sample(range(total_size), sample_size)
        random.seed()  # 重置随机种子
    
    def set_epoch(self, epoch):
        """设置epoch并重新采样"""
        self.epoch = epoch
        self._resample()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

# 创建single_motion的子采样数据集（当未启用加权采样时才启用1/3子采样）
if args.use_weighted_datasets:
    single_motion_subsampled = single_motion_dataset
    if local_rank == 0:
        print(f"\nWeighted datasets enabled: disable single_motion subsampling. Using full single_motion: {len(single_motion_dataset)} samples")
else:
    single_motion_subsampled = SubsampledDataset(single_motion_dataset, sample_ratio=1/3)
    if local_rank == 0:
        print(f"\nSingle_motion subsampling: {len(single_motion_dataset)} -> {len(single_motion_subsampled)} samples (1/3)")

# 合并数据集
combined_dataset = ConcatDataset([beat_dataset, single_motion_subsampled])
if local_rank == 0:
    print(f"Combined dataset: {len(combined_dataset)} samples (BEAT: {len(beat_dataset)}, single_motion: {len(single_motion_subsampled)})")
    print(f"All sequences max_length: {beat_config.max_seq_length}")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

if args.resume_from and os.path.exists(args.resume_from):
    checkpoint = torch.load(args.resume_from, map_location='cpu')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

# 创建DataLoader
# 若启用数据集加权采样，则用WeightedDatasetSampler替代DistributedSampler，实现每epoch按权重均衡抽样
if args.use_weighted_datasets:
    datasets_list = [beat_dataset, single_motion_subsampled]
    dataset_weights = args.dataset_weights
    if local_rank == 0:
        print(f"\nUsing WeightedDatasetSampler with weights: {dataset_weights}")
    sampler = WeightedDatasetSampler(
        datasets=datasets_list,
        dataset_weights=dataset_weights,
        num_samples=None,  # 默认使用各数据集总样本数之和，再按world_size划分
        replacement=True,
        shuffle=True,
        rank=local_rank,
        world_size=world_size
    )
    # 统计一次本进程将要抽样的每个数据集样本占比（仅首次估计）
    if local_rank == 0:
        try:
            tmp_indices = list(iter(sampler))
            # 将global idx映射到dataset idx
            def find_ds_idx(cum_sizes, gidx):
                for i in range(len(cum_sizes)-1):
                    if cum_sizes[i] <= gidx < cum_sizes[i+1]:
                        return i
                return len(cum_sizes)-2
            beat_count = 0
            single_count = 0
            for gidx in tmp_indices[: min(10000, len(tmp_indices))]:
                ds_idx = find_ds_idx(sampler.cumulative_sizes, gidx)
                if ds_idx == 0:
                    beat_count += 1
                else:
                    single_count += 1
            total_tmp = beat_count + single_count
            if total_tmp > 0:
                print(f"Sampled preview (first {total_tmp}): BEAT {beat_count/total_tmp:.3f}, single_motion {single_count/total_tmp:.3f}")
        except Exception as _:
            pass
else:
    if world_size > 1:
        sampler = DistributedSampler(
            combined_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        )
    else:
        sampler = None

def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch]).long()
    masks = torch.stack([item['mask'] for item in batch]).long()
    lengths = torch.tensor([item['seq_length'] for item in batch])
    source_ids = torch.tensor([item.get('source_id', 0) for item in batch]).long()
    return {'tokens': tokens, 'mask': masks, 'lengths': lengths, 'source_ids': source_ids}

dataloader = DataLoader(
    combined_dataset,
    batch_size=config.batch_size // world_size,
    sampler=sampler,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4,
    shuffle=(sampler is None)
)

if local_rank == 0:
    print(f"Estimated steps per epoch: {len(dataloader)}")

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config.learning_rate,
    steps_per_epoch=len(dataloader),
    epochs=config.epochs
)

accum_steps = 1  # 当前不使用梯度累积（可按需要开启）

# 用于跟踪全局step数
global_step = 0

if local_rank == 0:
    print(f"\n🚀 Starting finetuning from epoch {start_epoch} to {config.epochs}")
    print(f"   Total epochs to train: {config.epochs - start_epoch}")
    print(f"   Training strategy: BEAT + single_motion (1/3 subsampled) combined")
    print(f"   Config: max_seq_length={beat_config.max_seq_length}, batch_size={config.batch_size}, accum_steps={accum_steps}")
    print(f"   BEAT samples: {len(beat_dataset)}")
    print(f"   single_motion samples per epoch: {len(single_motion_subsampled)} (1/3 of {len(single_motion_dataset)})")
    print(f"{'='*80}\n")

for epoch in range(start_epoch, config.epochs):
    # 每个epoch重新采样single_motion数据集（仅当包装为SubsampledDataset时）
    if hasattr(single_motion_subsampled, "set_epoch"):
        single_motion_subsampled.set_epoch(epoch)
    
    # 统计本epoch各数据集占batch数的占比（以多数样本来源为该batch归属）
    batch_majority_counts = {0: 0, 1: 0}  # 0: BEAT, 1: single_motion
    total_batches_counted = 0
    
    if local_rank == 0:
        print(f"\n📊 Epoch {epoch+1}/{config.epochs}")
        print(f"   Combined dataset size: {len(combined_dataset)} samples")
        print(f"   Batches: {len(dataloader)}")
    
    # 设置sampler的epoch（确保每个epoch的shuffle不同）
    if sampler is not None:
        sampler.set_epoch(epoch)
    # 若使用WeightedDatasetSampler，同样设置epoch以获得每epoch不同的重采样
    if args.use_weighted_datasets and sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    if local_rank == 0:
        dataloader_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    else:
        dataloader_iter = dataloader
        
    for step, batch in enumerate(dataloader_iter):
        inputs = batch['tokens'].to(device, non_blocking=True).long()
        masks = batch['mask'].to(device, non_blocking=True).long()
        lengths = batch['lengths']
        source_ids = batch.get('source_ids', None)
        
        # 统计当前batch的来源多数归属
        if source_ids is not None:
            num_beat = (source_ids == 0).sum().item()
            num_single = (source_ids == 1).sum().item()
            if num_beat >= num_single:
                batch_majority_counts[0] += 1
            else:
                batch_majority_counts[1] += 1
            total_batches_counted += 1
        
        attn_mask = (inputs != config.pad_token_id).float().to(device)
        
        labels = inputs.clone().long()
        labels[masks == 0] = -100
        
        outputs = model(inputs, labels=labels, attention_mask=attn_mask)
        loss = outputs.loss / accum_steps
        
        loss.backward()
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # 更新全局step（只在accum_steps的倍数时更新，因为optimizer.step()和scheduler.step()只在此时调用）
            global_step += 1
        
        total_loss += loss.item() * accum_steps
        
        if local_rank == 0 and step == 0:
            print(f"📊 Epoch {epoch+1}, Step {step+1}: Loss = {loss.item() * accum_steps:.4f}")
            print(f"   Input shape: {inputs.shape}, Labels shape: {labels.shape}")
            print(f"   Attention mask shape: {attn_mask.shape}")
            print(f"   Motion mask shape: {masks.shape}")
            print(f"   Valid motion tokens: {masks.sum().item()}/{masks.numel()}")
            print(f"   Average sequence length: {lengths.float().mean().item():.2f}")
            print(f"   Min/Max sequence length: {lengths.min().item()}/{lengths.max().item()}")
        
        if local_rank == 0 and (step % 500 == 0 or step == len(dataloader) - 1):
            log_data = {
                "train/loss": loss.item() * accum_steps,
                "train/lr": scheduler.get_last_lr()[0],
                "train/seq_length": lengths.float().mean().item(),
                "train/seq_length_min": lengths.min().item(),
                "train/seq_length_max": lengths.max().item(),
                "train/step": global_step,
                "train/epoch": epoch,
                "train/memory_allocated": torch.cuda.memory_allocated(device) / 1024**3,
                "train/memory_reserved": torch.cuda.memory_reserved(device) / 1024**3,
            }
            wandb.log(log_data)
            dataloader_iter.set_postfix(loss=loss.item() * accum_steps)

    if world_size > 1:
        dist.barrier()
    
    if local_rank == 0:
        avg_loss = total_loss / len(dataloader)
        # 输出本epoch的batch占比统计
        if total_batches_counted > 0:
            beat_ratio = batch_majority_counts[0] / total_batches_counted
            single_ratio = batch_majority_counts[1] / total_batches_counted
            print(f"   Batch majority ratio - BEAT: {beat_ratio:.3f}, single_motion: {single_ratio:.3f} "
                  f"({batch_majority_counts[0]}/{total_batches_counted} vs {batch_majority_counts[1]}/{total_batches_counted})")
            wandb.log({
                "epoch/batch_majority_ratio_beat": beat_ratio,
                "epoch/batch_majority_ratio_single_motion": single_ratio,
                "epoch/batches_counted": total_batches_counted,
            })
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch": epoch,
            "epoch/step": global_step,
        })
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f} | Steps: {len(dataloader)}")
        
        if (epoch + 1) % 50 == 0:
            ckpt_path = f"output_disk0/motion_adaptor_v11/{config.exp_name}/checkpoints/epoch_{epoch+1}.pt"
            model_to_save = model.module if world_size > 1 else model
            torch.save({
                'epoch': epoch,
                'model_state': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            wandb.save(ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

if local_rank == 0:
    model_to_save = model.module if world_size > 1 else model
    model_to_save.save_pretrained(f"output_disk0/motion_adaptor_v11/{config.exp_name}")

if world_size > 1:
    dist.destroy_process_group()
