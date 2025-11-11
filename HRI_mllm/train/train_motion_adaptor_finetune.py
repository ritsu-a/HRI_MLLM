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
        self.samples = []
        self.action_classes = []  # 存储每个样本的动作类别
        self.seq_lengths = []  # 存储每个样本的序列长度
        self.stats = {'total_sequences': 0, 'generated_samples': 0, 'max_length': 0}
        self.interleave_audios, self.interleave_motions = config.interleave_ratio
        self.SEQ_PAD_TOKEN = config.pad_token_id
        self.extract_action_class = extract_action_class
        
        print(f"Loading JSONL file: {jsonl_path}")
        
        # 读取JSONL文件
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # 从conversation中提取audio和motion tokens
                    audio_tokens = None
                    motion_tokens = None
                    motion_labels = None
                    action_class = None
                    
                    for msg in data['conversation']:
                        if msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                            audio_tokens = msg['audio_tokens']
                            # 从audio文件路径提取动作类别
                            if self.extract_action_class:
                                audio_path = msg.get('content', '')
                                if '/' in audio_path:
                                    filename = audio_path.split('/')[-1]
                                    if '_' in filename:
                                        # 提取动作类别（前两部分，如 FIST_BEAT）
                                        parts = filename.split('_')
                                        if len(parts) >= 2:
                                            action_class = f"{parts[0]}_{parts[1]}"
                        elif msg.get('message_type') == 'audio_motion' and 'motion_tokens' in msg:
                            motion_tokens = msg['motion_tokens']
                    
                    # 读取motion_labels（如果存在）
                    if 'motion_labels' in data:
                        motion_labels = data['motion_labels']
                    
                    if audio_tokens is None or motion_tokens is None:
                        continue
                    
                    # 转换为torch tensor
                    if not isinstance(audio_tokens, torch.Tensor):
                        audio_tokens = torch.tensor(audio_tokens)
                    if not isinstance(motion_tokens, torch.Tensor):
                        motion_tokens = torch.tensor(motion_tokens)
                    
                    # 构建motion_tokens中需要插入特殊token的位置集合
                    gesture_start_indices = set()
                    gesture_end_indices = set()
                    if motion_labels:
                        for label in motion_labels:
                            if 'start_token_index' in label:
                                gesture_start_indices.add(label['start_token_index'])
                            if 'end_token_index' in label:
                                gesture_end_indices.add(label['end_token_index'])
                    
                    # 构建完整序列（带特殊token）
                    full_sequence = []
                    token_types = []
                    
                    # 跟踪当前在motion_tokens中的索引
                    motion_token_idx = 0
                    
                    for i in range(len(audio_tokens)):
                        # 插入audio token
                        full_sequence.append(audio_tokens[i].item())
                        token_types.append(0)
                        
                        # 根据interleave_ratio插入motion tokens
                        if (i + 1) % self.interleave_audios == 0:
                            motion_block_size = self.interleave_motions
                            for j in range(motion_block_size):
                                if motion_token_idx < len(motion_tokens):
                                    # 检查是否是gesture_start
                                    if motion_token_idx in gesture_start_indices:
                                        full_sequence.append(self.config.gesture_start_token_id)
                                        token_types.append(1)
                                        full_sequence.append(self.config.audio_gesture_start_token_id)
                                        token_types.append(0)
                                    
                                    # 插入motion token
                                    full_sequence.append(motion_tokens[motion_token_idx].item())
                                    token_types.append(1)
                                    
                                    # 检查是否是gesture_end
                                    if motion_token_idx in gesture_end_indices:
                                        full_sequence.append(self.config.gesture_end_token_id)
                                        token_types.append(1)
                                        full_sequence.append(self.config.audio_gesture_end_token_id)
                                        token_types.append(0)
                                    
                                    motion_token_idx += 1
                    
                    # 处理剩余的motion tokens
                    while motion_token_idx < len(motion_tokens):
                        if motion_token_idx in gesture_start_indices:
                            full_sequence.append(self.config.gesture_start_token_id)
                            token_types.append(1)
                            full_sequence.append(self.config.audio_gesture_start_token_id)
                            token_types.append(0)
                        
                        full_sequence.append(motion_tokens[motion_token_idx].item())
                        token_types.append(1)
                        
                        if motion_token_idx in gesture_end_indices:
                            full_sequence.append(self.config.gesture_end_token_id)
                            token_types.append(1)
                            full_sequence.append(self.config.audio_gesture_end_token_id)
                            token_types.append(0)
                        
                        motion_token_idx += 1
                    
                    # 应用滑动窗口
                    sample_indices = self.apply_sliding_window(full_sequence, token_types, action_class)
                    self.stats['total_sequences'] += 1
                    
                except Exception as e:
                    print(f"Error processing line {line_num} in {jsonl_path}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
        if self.extract_action_class:
            action_counts = defaultdict(int)
            for ac in self.action_classes:
                if ac:
                    action_counts[ac] += 1
            print(f"Action class distribution: {dict(action_counts)}")
    
    def apply_sliding_window(self, full_seq, token_types, action_class=None):
        """应用滑动窗口，返回生成的样本索引"""
        seq_len = len(full_seq)
        sample_indices = []
        
        if seq_len > self.config.max_seq_length:
            return sample_indices
        
        sub_seq = full_seq
        sub_types = token_types
        
        mask = [1 if t == 1 else 0 for t in sub_types]
        
        padded_seq = sub_seq + [self.SEQ_PAD_TOKEN] * (self.config.max_seq_length - len(sub_seq))
        padded_mask = mask + [0] * (self.config.max_seq_length - len(mask))
        
        sample_idx = len(self.samples)
        seq_length = len(sub_seq)
        self.samples.append({
            'tokens': torch.tensor(padded_seq),
            'mask': torch.tensor(padded_mask),
            'seq_length': seq_length
        })
        self.action_classes.append(action_class)
        self.seq_lengths.append(seq_length)
        sample_indices.append(sample_idx)
        
        self.stats['generated_samples'] += 1
        self.stats['max_length'] = max(self.stats['max_length'], seq_length)
        
        return sample_indices
    
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
parser.add_argument('--batch_size', type=int, default=64,
                   help='Batch size for BEAT dataset')
parser.add_argument('--max_seq_length', type=int, default=4096,
                   help='Maximum sequence length for BEAT dataset')
parser.add_argument('--dataset_weights', type=float, nargs='+', default=[1.0, 1.0],
                   help='Dataset weights for BEAT and single_motion')
# single_motion 专用参数
parser.add_argument('--single_motion_batch_size', type=int, default=256,
                   help='Batch size for single_motion dataset')
parser.add_argument('--single_motion_max_seq_length', type=int, default=256,
                   help='Maximum sequence length for single_motion dataset')
args = parser.parse_args()

exp_name = "kimi_audio_motion_gpt2_brainco_finetune_beat_single_motion"
os.makedirs(os.path.join("output/motion_adaptor_v10", exp_name), exist_ok=True)
os.makedirs(os.path.join("output/motion_adaptor_v10", exp_name, "checkpoints"), exist_ok=True)

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
    "single_motion": "/root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens_with_labels.jsonl",
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
# 训练时使用较短序列（256）不影响模型结构
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
    "max_seq_length": 256,  # BEAT裁剪到256
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
})()

single_motion_config = type('Config', (), {
    "jsonl_files": config.jsonl_files,
    "audio_vocab_size": config.audio_vocab_size,
    "motion_vocab_size": config.motion_vocab_size,
    "total_vocab_size": config.total_vocab_size,
    "max_seq_length": args.single_motion_max_seq_length,  # single_motion使用256
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

# 创建single_motion的子采样数据集（1/3 ≈ 0.333）
single_motion_subsampled = SubsampledDataset(single_motion_dataset, sample_ratio=1/3)

if local_rank == 0:
    print(f"\nSingle_motion subsampling: {len(single_motion_dataset)} -> {len(single_motion_subsampled)} samples (1/3)")

# 合并数据集
combined_dataset = ConcatDataset([beat_dataset, single_motion_subsampled])
if local_rank == 0:
    print(f"Combined dataset: {len(combined_dataset)} samples (BEAT: {len(beat_dataset)}, single_motion: {len(single_motion_subsampled)})")
    print(f"All sequences max_length: 256")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

if args.resume_from and os.path.exists(args.resume_from):
    checkpoint = torch.load(args.resume_from, map_location='cpu')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

# 创建DataLoader
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
    return {'tokens': tokens, 'mask': masks, 'lengths': lengths}

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

accum_steps = 1  # 序列长度统一为256，不需要梯度累积

# 用于跟踪全局step数
global_step = 0

if local_rank == 0:
    print(f"\n🚀 Starting finetuning from epoch {start_epoch} to {config.epochs}")
    print(f"   Total epochs to train: {config.epochs - start_epoch}")
    print(f"   Training strategy: BEAT + single_motion (1/3 subsampled) combined")
    print(f"   Config: max_seq_length=256, batch_size={config.batch_size}, accum_steps={accum_steps}")
    print(f"   BEAT samples: {len(beat_dataset)}")
    print(f"   single_motion samples per epoch: {len(single_motion_subsampled)} (1/3 of {len(single_motion_dataset)})")
    print(f"{'='*80}\n")

for epoch in range(start_epoch, config.epochs):
    # 每个epoch重新采样single_motion数据集
    single_motion_subsampled.set_epoch(epoch)
    
    if local_rank == 0:
        print(f"\n📊 Epoch {epoch+1}/{config.epochs}")
        print(f"   Combined dataset size: {len(combined_dataset)} samples")
        print(f"   Batches: {len(dataloader)}")
    
    # 设置sampler的epoch（确保每个epoch的shuffle不同）
    if sampler is not None:
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
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch": epoch,
            "epoch/step": global_step,
        })
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f} | Steps: {len(dataloader)}")
        
        if (epoch + 1) % 50 == 0:
            ckpt_path = f"output/motion_adaptor_v10/{config.exp_name}/checkpoints/epoch_{epoch+1}.pt"
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
    model_to_save.save_pretrained(f"output/motion_adaptor_v10/{config.exp_name}")

if world_size > 1:
    dist.destroy_process_group()
