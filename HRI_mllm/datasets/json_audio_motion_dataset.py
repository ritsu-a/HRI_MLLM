#!/usr/bin/env python3
"""
JSON数据集加载器
支持从JSON文件加载音频和motion tokens数据
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path


class JSONAudioMotionDataset(Dataset):
    """
    从JSON文件加载音频-motion数据的Dataset
    支持多种JSON格式：
    1. 单条记录格式：{"audio_tokens": [...], "motion_tokens": [...]}
    2. 对话格式：{"conversation": [{"message_type": "audio", "audio_tokens": [...]}, ...]}
    3. 批量格式：{"data": [{"audio_tokens": [...], "motion_tokens": [...]}, ...]}
    """
    
    def __init__(self, 
                 json_path: str,
                 max_audio_length: int = 2048,
                 max_motion_length: int = 1024,
                 interleave_ratio: Tuple[int, int] = (1, 1),
                 min_audio_length: int = 32,
                 min_motion_length: int = 16,
                 token_offset: int = 0,
                 debug: bool = False):
        """
        Args:
            json_path: JSON文件路径
            max_audio_length: 最大音频token长度
            max_motion_length: 最大motion token长度
            interleave_ratio: 交错比例 (audio_tokens_per_motion, motion_tokens_per_audio)
            min_audio_length: 最小音频token长度
            min_motion_length: 最小motion token长度
            token_offset: token偏移量
            debug: 是否开启调试模式
        """
        self.json_path = json_path
        self.max_audio_length = max_audio_length
        self.max_motion_length = max_motion_length
        self.interleave_ratio = interleave_ratio
        self.min_audio_length = min_audio_length
        self.min_motion_length = min_motion_length
        self.token_offset = token_offset
        self.debug = debug
        
        self.samples = []
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'skipped_samples': 0,
            'max_audio_len': 0,
            'max_motion_len': 0,
            'avg_audio_len': 0,
            'avg_motion_len': 0
        }
        
        self._load_data()
        self._print_stats()
    
    def _load_data(self):
        """加载JSON数据"""
        print(f"🔄 Loading JSON data from: {self.json_path}")
        
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        
        # 尝试不同的JSON格式
        raw_samples = []
        
        try:
            # 首先尝试作为JSONL格式（每行一个JSON对象）
            with open(self.json_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            data = json.loads(line)
                            raw_samples.append(data)
                        except json.JSONDecodeError as e:
                            if self.debug:
                                print(f"⚠️  JSON decode error at line {line_num + 1}: {e}")
                            continue
            
            # 如果成功解析了JSONL格式
            if raw_samples:
                print(f"📊 Loaded as JSONL format: {len(raw_samples)} samples")
            else:
                raise ValueError("No valid JSONL samples found")
                
        except Exception as e:
            if self.debug:
                print(f"⚠️  JSONL parsing failed: {e}")
            
            # 回退到单个JSON文件格式
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理不同的JSON格式
                if isinstance(data, list):
                    # 列表格式：每个元素是一个样本
                    raw_samples = data
                elif isinstance(data, dict):
                    if 'data' in data:
                        # 批量格式：{"data": [...]}
                        raw_samples = data['data']
                    elif 'conversation' in data:
                        # 对话格式：{"conversation": [...]}
                        raw_samples = [data]  # 单个对话
                    else:
                        # 单条记录格式：{"audio_tokens": [...], "motion_tokens": [...]}
                        raw_samples = [data]
                else:
                    raise ValueError(f"Unsupported JSON format: {type(data)}")
                
                print(f"📊 Loaded as JSON format: {len(raw_samples)} samples")
                
            except Exception as e2:
                raise ValueError(f"Failed to parse JSON file: {e2}")
        
        print(f"📊 Found {len(raw_samples)} raw samples")
        
        # 处理每个样本
        for i, sample in enumerate(raw_samples):
            try:
                processed_sample = self._process_sample(sample, i)
                if processed_sample is not None:
                    self.samples.append(processed_sample)
                    self.stats['valid_samples'] += 1
                else:
                    self.stats['skipped_samples'] += 1
            except Exception as e:
                if self.debug:
                    print(f"⚠️  Error processing sample {i}: {e}")
                self.stats['skipped_samples'] += 1
        
        self.stats['total_samples'] = len(raw_samples)
        
        if self.stats['valid_samples'] > 0:
            self.stats['avg_audio_len'] = sum(s['user_audio_length'] for s in self.samples) / len(self.samples)
            self.stats['avg_motion_len'] = sum(s['motion_length'] for s in self.samples) / len(self.samples)
    
    def _process_sample(self, sample: Dict[str, Any], sample_idx: int) -> Optional[Dict[str, Any]]:
        """处理单个样本"""
        
        # 检查sample类型
        if not isinstance(sample, dict):
            if self.debug:
                print(f"⚠️  Sample {sample_idx}: Expected dict, got {type(sample)}")
            return None
        
        # 提取user text、user audio tokens、assistant audio tokens和motion tokens
        user_text = None
        user_audio_tokens = None
        assistant_audio_tokens = None
        motion_tokens = None
        
        if 'conversation' in sample:
            # 对话格式
            for msg in sample['conversation']:
                if msg.get('role') == 'user' and msg.get('message_type') == 'text':
                    user_text = msg.get('content')
                elif msg.get('role') == 'user' and msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                    user_audio_tokens = msg['audio_tokens']
                elif msg.get('role') == 'assistant' and msg.get('message_type') == 'audio_motion':
                    if 'audio_tokens' in msg:
                        assistant_audio_tokens = msg['audio_tokens']
                    if 'motion_tokens' in msg:
                        motion_tokens = msg['motion_tokens']
        else:
            # 直接格式（保持向后兼容）
            user_audio_tokens = sample.get('audio_tokens')
            motion_tokens = sample.get('motion_tokens')
        
        # 检查是否找到必要的tokens
        if user_audio_tokens is None or assistant_audio_tokens is None or motion_tokens is None:
            if self.debug:
                print(f"⚠️  Sample {sample_idx}: Missing required tokens (user_audio: {user_audio_tokens is not None}, assistant_audio: {assistant_audio_tokens is not None}, motion: {motion_tokens is not None})")
            return None
        
        # 转换为tensor
        if not isinstance(user_audio_tokens, torch.Tensor):
            user_audio_tokens = torch.tensor(user_audio_tokens, dtype=torch.long)
        if not isinstance(assistant_audio_tokens, torch.Tensor):
            assistant_audio_tokens = torch.tensor(assistant_audio_tokens, dtype=torch.long)
        if motion_tokens is not None and not isinstance(motion_tokens, torch.Tensor):
            motion_tokens = torch.tensor(motion_tokens, dtype=torch.long)
        
        # 检查长度
        user_audio_length = len(user_audio_tokens)
        assistant_audio_length = len(assistant_audio_tokens)
        motion_length = len(motion_tokens) if motion_tokens is not None else 0
        
        if user_audio_length < self.min_audio_length or assistant_audio_length < self.min_audio_length:
            if self.debug:
                print(f"⚠️  Sample {sample_idx}: Too short (user audio: {user_audio_length}, assistant audio: {assistant_audio_length})")
            return None
        
        # 截断过长的序列
        if user_audio_length > self.max_audio_length:
            user_audio_tokens = user_audio_tokens[:self.max_audio_length]
            user_audio_length = self.max_audio_length
        if assistant_audio_length > self.max_audio_length:
            assistant_audio_tokens = assistant_audio_tokens[:self.max_audio_length]
            assistant_audio_length = self.max_audio_length
        if motion_tokens is not None and motion_length > self.max_motion_length:
            motion_tokens = motion_tokens[:self.max_motion_length]
            motion_length = self.max_motion_length
        
        # 更新统计信息
        self.stats['max_audio_len'] = max(self.stats['max_audio_len'], user_audio_length, assistant_audio_length)
        self.stats['max_motion_len'] = max(self.stats['max_motion_len'], motion_length)
        
        # 构建交错序列（使用assistant audio tokens和motion tokens）
        interleaved_sequence, token_labels = self._build_interleaved_sequence(
            assistant_audio_tokens, motion_tokens
        )
        
        return {
            'user_text': user_text,
            'user_audio_tokens': user_audio_tokens,
            'assistant_audio_tokens': assistant_audio_tokens,
            'motion_tokens': motion_tokens,
            'interleaved_sequence': interleaved_sequence,
            'token_labels': token_labels,
            'user_audio_length': user_audio_length,
            'assistant_audio_length': assistant_audio_length,
            'motion_length': motion_length,
            'sequence_length': len(interleaved_sequence),
            'sample_idx': sample_idx
        }
    
    def _build_interleaved_sequence(self, 
                                  audio_tokens: torch.Tensor, 
                                  motion_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建交错的音频-motion序列
        参考train_motion_adaptor.py的实现
        """
        interleave_audios, interleave_motions = self.interleave_ratio
        
        sequence = []
        labels = []
        
        # 计算需要生成的motion token数量
        max_motion_tokens = len(audio_tokens) // interleave_audios * interleave_motions
        actual_motion_tokens = min(max_motion_tokens, len(motion_tokens))
        
        motion_idx = 0
        
        for i, audio_token in enumerate(audio_tokens):
            # 添加audio token
            sequence.append(audio_token.item())
            labels.append(-100)  # audio token对应的label为-100
            
            # 每interleave_audios个audio token后，添加interleave_motions个motion token
            if (i + 1) % interleave_audios == 0 and motion_idx < actual_motion_tokens:
                for j in range(interleave_motions):
                    if motion_idx < actual_motion_tokens:
                        motion_token = motion_tokens[motion_idx].item()
                        sequence.append(motion_token)
                        labels.append(motion_token)  # motion token对应的label为自身
                        motion_idx += 1
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def _print_stats(self):
        """打印统计信息"""
        print(f"\n📊 Dataset Statistics:")
        print(f"   - Total samples: {self.stats['total_samples']}")
        print(f"   - Valid samples: {self.stats['valid_samples']}")
        print(f"   - Skipped samples: {self.stats['skipped_samples']}")
        print(f"   - Max audio length: {self.stats['max_audio_len']}")
        print(f"   - Max motion length: {self.stats['max_motion_len']}")
        print(f"   - Avg audio length: {self.stats['avg_audio_len']:.1f}")
        print(f"   - Avg motion length: {self.stats['avg_motion_len']:.1f}")
        print(f"   - Interleave ratio: {self.interleave_ratio}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.samples[idx]
        
        # 验证sample数据
        if sample is None:
            raise ValueError(f"Sample at index {idx} is None")
        
        # 验证必要字段
        required_fields = ['user_audio_tokens', 'assistant_audio_tokens', 'motion_tokens', 'interleaved_sequence', 'token_labels', 'sequence_length']
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Sample at index {idx} missing field: {field}")
            if sample[field] is None:
                raise ValueError(f"Sample at index {idx} has None value for field: {field}")
        
        return {
            'user_text': sample['user_text'],
            'user_audio_tokens': sample['user_audio_tokens'],
            'assistant_audio_tokens': sample['assistant_audio_tokens'],
            'motion_tokens': sample['motion_tokens'],
            'interleaved_sequence': sample['interleaved_sequence'],
            'token_labels': sample['token_labels'],
            'user_audio_length': sample['user_audio_length'],
            'assistant_audio_length': sample['assistant_audio_length'],
            'motion_length': sample['motion_length'],
            'sequence_length': sample['sequence_length'],
            'sample_idx': sample['sample_idx']
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本信息"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.samples[idx]
        return {
            'sample_idx': sample['sample_idx'],
            'audio_length': sample['audio_length'],
            'motion_length': sample['motion_length'],
            'sequence_length': sample['sequence_length'],
            'audio_token_range': (sample['audio_tokens'].min().item(), sample['audio_tokens'].max().item()),
            'motion_token_range': (sample['motion_tokens'].min().item(), sample['motion_tokens'].max().item())
        }


def collate_fn(batch: List[Dict[str, Any]], debug: bool = False) -> Dict[str, torch.Tensor]:
    """
    批处理函数
    """
    if debug:
        print(f"🔍 Debug collate_fn: Processing batch with {len(batch)} items")
    
    # 过滤掉None值
    original_batch_size = len(batch)
    batch = [item for item in batch if item is not None]
    filtered_count = original_batch_size - len(batch)
    
    if filtered_count > 0:
        print(f"⚠️  Filtered out {filtered_count} None items from batch")
    
    if not batch:
        raise ValueError("Empty batch after filtering None values")
    
    if debug:
        print(f"✅ Batch after filtering: {len(batch)} items")
    
    # 验证每个item的必要字段
    for i, item in enumerate(batch):
        if debug:
            print(f"🔍 Debug item {i}:")
            print(f"   - Type: {type(item)}")
        
        if not isinstance(item, dict):
            if debug:
                print(f"   ❌ Item {i} is not a dict: {type(item)}")
            raise ValueError(f"Item {i} is not a dict: {type(item)}")
        
        if debug:
            print(f"   - Keys: {list(item.keys())}")
        
        required_keys = ['sequence_length', 'interleaved_sequence', 'token_labels']
        for key in required_keys:
            if key not in item:
                if debug:
                    print(f"   ❌ Item {i} missing required key: {key}")
                raise ValueError(f"Item {i} missing required key: {key}")
            if item[key] is None:
                if debug:
                    print(f"   ❌ Item {i} has None value for key: {key}")
                raise ValueError(f"Item {i} has None value for key: {key}")
            
            # 检查tensor属性
            tensor_value = item[key]
            if debug and hasattr(tensor_value, 'shape'):
                print(f"   - {key}: shape={tensor_value.shape}, dtype={tensor_value.dtype}")
            elif debug:
                print(f"   - {key}: type={type(tensor_value)}")
    
    # 获取最大序列长度
    max_seq_len = max(item['sequence_length'] for item in batch)
    batch_size = len(batch)
    
    # 初始化batch tensors
    interleaved_sequences = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    token_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    
    # 填充数据
    for i, item in enumerate(batch):
        seq_len = item['sequence_length']
        interleaved_sequences[i, :seq_len] = item['interleaved_sequence']
        token_labels[i, :seq_len] = item['token_labels']
        attention_masks[i, :seq_len] = 1
    
    # 提取用户文本
    user_texts = [item.get('user_text') for item in batch]
    
    # 提取用户音频tokens并处理不同长度
    user_audio_tokens_list = []
    assistant_audio_tokens_list = []
    motion_tokens_list = []
    
    for item in batch:
        user_audio_tokens_list.append(item['user_audio_tokens'])
        assistant_audio_tokens_list.append(item['assistant_audio_tokens'])
        if item.get('motion_tokens') is not None:
            motion_tokens_list.append(item['motion_tokens'])
        else:
            motion_tokens_list.append(torch.tensor([], dtype=torch.long))
    
    # 找到最大长度并padding
    max_user_audio_len = max(len(tokens) for tokens in user_audio_tokens_list)
    max_assistant_audio_len = max(len(tokens) for tokens in assistant_audio_tokens_list)
    max_motion_len = max(len(tokens) for tokens in motion_tokens_list)
    
    # Padding到相同长度
    user_audio_tokens = torch.zeros(batch_size, max_user_audio_len, dtype=torch.long)
    assistant_audio_tokens = torch.zeros(batch_size, max_assistant_audio_len, dtype=torch.long)
    motion_tokens = torch.zeros(batch_size, max_motion_len, dtype=torch.long)
    
    for i, (user_tokens, assistant_tokens, motion_tokens_item) in enumerate(zip(user_audio_tokens_list, assistant_audio_tokens_list, motion_tokens_list)):
        user_audio_tokens[i, :len(user_tokens)] = user_tokens
        assistant_audio_tokens[i, :len(assistant_tokens)] = assistant_tokens
        if len(motion_tokens_item) > 0:
            motion_tokens[i, :len(motion_tokens_item)] = motion_tokens_item
    
    return {
        'user_text': user_texts,
        'user_audio_tokens': user_audio_tokens,
        'assistant_audio_tokens': assistant_audio_tokens,
        'motion_tokens': motion_tokens,
        'interleaved_sequences': interleaved_sequences,
        'token_labels': token_labels,
        'attention_masks': attention_masks,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len
    }


class MultiJSONDataset(Dataset):
    """
    多个JSON文件的组合数据集
    """
    
    def __init__(self, 
                 json_paths: List[str],
                 dataset_weights: Optional[List[float]] = None,
                 **kwargs):
        """
        Args:
            json_paths: JSON文件路径列表
            dataset_weights: 数据集权重列表
            **kwargs: 传递给JSONAudioMotionDataset的参数
        """
        self.json_paths = json_paths
        self.dataset_weights = dataset_weights or [1.0] * len(json_paths)
        
        # 加载所有数据集
        self.datasets = []
        for json_path in json_paths:
            dataset = JSONAudioMotionDataset(json_path, **kwargs)
            self.datasets.append(dataset)
        
        # 计算累积权重
        self.cumulative_weights = []
        total_weight = sum(self.dataset_weights)
        cumsum = 0
        for weight in self.dataset_weights:
            cumsum += weight / total_weight
            self.cumulative_weights.append(cumsum)
        
        # 计算总样本数
        self.total_samples = sum(len(dataset) for dataset in self.datasets)
        
        print(f"📊 MultiJSONDataset initialized:")
        print(f"   - {len(self.datasets)} datasets")
        print(f"   - Total samples: {self.total_samples}")
        print(f"   - Weights: {self.dataset_weights}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # 根据权重选择数据集
        rand = torch.rand(1).item()
        dataset_idx = 0
        for i, cum_weight in enumerate(self.cumulative_weights):
            if rand <= cum_weight:
                dataset_idx = i
                break
        
        # 从选定的数据集中获取样本
        dataset = self.datasets[dataset_idx]
        sample_idx = torch.randint(0, len(dataset), (1,)).item()
        
        return dataset[sample_idx]


def create_dataloader(json_paths: List[str],
                     batch_size: int = 4,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     dataset_weights: Optional[List[float]] = None,
                     debug: bool = False,
                     **dataset_kwargs) -> torch.utils.data.DataLoader:
    """
    创建数据加载器
    
    Args:
        json_paths: JSON文件路径列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        dataset_weights: 数据集权重
        debug: 是否开启debug模式
        **dataset_kwargs: 传递给数据集的其他参数
    
    Returns:
        DataLoader: 数据加载器
    """
    
    if len(json_paths) == 1:
        dataset = JSONAudioMotionDataset(json_paths[0], **dataset_kwargs)
    else:
        dataset = MultiJSONDataset(json_paths, dataset_weights, **dataset_kwargs)
    
    # 创建带debug参数的collate_fn
    def debug_collate_fn(batch):
        return collate_fn(batch, debug=debug)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=debug_collate_fn,
        pin_memory=False  # 禁用pin_memory避免CUDA错误
    )
    
    return dataloader


# 示例用法
if __name__ == "__main__":
    # 测试数据集加载
    json_path = "/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl"
    
    if os.path.exists(json_path):
        dataset = JSONAudioMotionDataset(
            json_path=json_path,
            max_audio_length=1024,
            max_motion_length=512,
            interleave_ratio=(1, 1),
            debug=True
        )
        
        print(f"\n🧪 Testing dataset:")
        print(f"   - Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   - Sample keys: {list(sample.keys())}")
            print(f"   - Audio tokens shape: {sample['audio_tokens'].shape}")
            print(f"   - Motion tokens shape: {sample['motion_tokens'].shape}")
            print(f"   - Interleaved sequence shape: {sample['interleaved_sequence'].shape}")
            print(f"   - Token labels shape: {sample['token_labels'].shape}")
            
            # 测试批处理
            dataloader = create_dataloader([json_path], batch_size=2, num_workers=0)
            batch = next(iter(dataloader))
            print(f"   - Batch keys: {list(batch.keys())}")
            print(f"   - Batch interleaved_sequences shape: {batch['interleaved_sequences'].shape}")
            print(f"   - Batch token_labels shape: {batch['token_labels'].shape}")
    else:
        print(f"❌ Test JSON file not found: {json_path}")
