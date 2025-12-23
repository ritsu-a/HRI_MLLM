"""
音频-动作未来预测数据集
任务：根据25帧audio token + 25帧motion token历史，预测14帧future motion token
"""

import torch
from torch.utils.data import Dataset
import json
import os


class AudioMotionFuturePredictionDataset(Dataset):
    """
    音频-动作未来预测数据集
    
    输入格式：
    - 25帧audio token（历史）
    - 25帧motion token（历史）
    预测目标：
    - 14帧future motion token
    
    总序列长度：64 (25 + 25 + 14)
    """
    
    def __init__(self, jsonl_path, config, max_samples=None):
        """
        Args:
            jsonl_path: JSONL文件路径
            config: 配置对象，需要包含以下属性：
                - pad_token_id: padding token ID
                - audio_empty_token_id: audio empty token ID
                - motion_empty_token_id: motion empty token ID
                - history_audio_frames: 历史audio帧数（默认25）
                - history_motion_frames: 历史motion帧数（默认25）
                - future_motion_frames: 未来motion帧数（默认14）
            max_samples: 最大样本数量（用于调试，None表示加载全部）
        """
        self.config = config
        self.jsonl_path = jsonl_path
        self.samples = []
        self.max_samples = max_samples
        
        # 任务参数
        self.history_audio_frames = getattr(config, 'history_audio_frames', 25)
        self.history_motion_frames = getattr(config, 'history_motion_frames', 25)
        self.future_motion_frames = getattr(config, 'future_motion_frames', 14)
        self.max_seq_length = self.history_audio_frames + self.history_motion_frames + self.future_motion_frames  # 64
        
        self.pad_token_id = config.pad_token_id
        self.audio_empty_token_id = config.audio_empty_token_id
        self.motion_empty_token_id = config.motion_empty_token_id
        
        print(f"Loading JSONL file: {jsonl_path}")
        print(f"Task configuration:")
        print(f"  - History audio frames: {self.history_audio_frames}")
        print(f"  - History motion frames: {self.history_motion_frames}")
        print(f"  - Future motion frames: {self.future_motion_frames}")
        print(f"  - Total sequence length: {self.max_seq_length}")
        
        # 读取JSONL文件
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # 如果设置了max_samples，检查是否已达到限制
                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    
                    # 从conversation中提取audio和motion tokens
                    audio_tokens = None
                    motion_tokens = None
                    
                    for msg in data['conversation']:
                        if msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                            audio_tokens = msg['audio_tokens']
                        elif msg.get('message_type') == 'audio_motion' and 'motion_tokens' in msg:
                            motion_tokens = msg['motion_tokens']
                    
                    if audio_tokens is None or motion_tokens is None:
                        continue
                    
                    # 转换为torch tensor
                    if not isinstance(audio_tokens, torch.Tensor):
                        audio_tokens = torch.tensor(audio_tokens)
                    if not isinstance(motion_tokens, torch.Tensor):
                        motion_tokens = torch.tensor(motion_tokens)
                    
                    # 记录添加前的样本数量
                    samples_before = len(self.samples)
                    
                    # 应用滑动窗口生成训练样本
                    self._create_samples_from_sequence(audio_tokens, motion_tokens)
                    
                    # 如果设置了max_samples，截断到指定数量
                    if self.max_samples is not None and len(self.samples) > self.max_samples:
                        self.samples = self.samples[:self.max_samples]
                        break
                    
                except Exception as e:
                    print(f"Error processing line {line_num} in {jsonl_path}: {e}")
                    continue
        
        if self.max_samples is not None:
            print(f"Loaded {len(self.samples)} samples from {jsonl_path} (limited to {self.max_samples} for debugging)")
        else:
            print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def _create_samples_from_sequence(self, audio_tokens, motion_tokens):
        """
        从完整的audio和motion序列创建训练样本
        
        使用滑动窗口策略：
        - 每次取history_audio_frames个audio token和history_motion_frames个motion token作为历史
        - 取接下来的future_motion_frames个motion token作为预测目标
        """
        audio_len = len(audio_tokens)
        motion_len = len(motion_tokens)
        
        # 计算可以创建多少个样本
        # 需要确保有足够的历史和未来数据
        min_required_motion = self.history_motion_frames + self.future_motion_frames
        
        if motion_len < min_required_motion:
            # 如果motion序列太短，跳过
            return
        
        # 使用滑动窗口，步长为1（可以重叠）
        # 或者使用更大的步长以减少重叠
        window_step = getattr(self.config, 'window_step', 1)
        
        # 对于每个可能的起始位置
        max_start_idx = motion_len - min_required_motion
        
        for start_motion_idx in range(0, max_start_idx + 1, window_step):
            # 提取motion历史
            history_motion = motion_tokens[start_motion_idx:start_motion_idx + self.history_motion_frames]
            
            # 提取future motion（预测目标）
            future_start_idx = start_motion_idx + self.history_motion_frames
            future_end_idx = future_start_idx + self.future_motion_frames
            
            if future_end_idx > motion_len:
                # 如果future motion不足，跳过
                continue
            
            future_motion = motion_tokens[future_start_idx:future_end_idx]
            
            # 提取对应的audio历史
            # audio和motion可能不是一一对应的，需要找到对应的audio范围
            # 这里假设audio和motion是时间对齐的，或者使用简单的映射
            # 如果audio长度不够，使用padding
            
            # 计算对应的audio起始位置（简单映射：假设audio和motion大致对齐）
            audio_start_idx = min(start_motion_idx, audio_len - self.history_audio_frames)
            audio_start_idx = max(0, audio_start_idx)
            audio_end_idx = audio_start_idx + self.history_audio_frames
            
            if audio_end_idx > audio_len:
                # 如果audio不足，在前面padding
                history_audio = torch.full((self.history_audio_frames,), self.audio_empty_token_id, dtype=audio_tokens.dtype)
                available_audio = audio_tokens[audio_start_idx:]
                history_audio[:len(available_audio)] = available_audio
            else:
                history_audio = audio_tokens[audio_start_idx:audio_end_idx]
            
            # 【修复】构建完整序列：[25 audio] + [25 motion] + [14 padding]
            # 重要：future_motion位置应该用padding，而不是真实token！
            # 这样才能让模型学习从历史预测未来，而不是"看到答案"再预测
            future_motion_padding = torch.full(
                (self.future_motion_frames,), 
                self.motion_empty_token_id, 
                dtype=motion_tokens.dtype
            )
            sequence = torch.cat([
                history_audio,
                history_motion,
                future_motion_padding  # 使用padding而不是真实token
            ])
            
            # 创建token类型mask：0=audio, 1=motion, 2=future_motion（用于loss计算）
            token_types = torch.cat([
                torch.zeros(self.history_audio_frames, dtype=torch.long),  # audio
                torch.ones(self.history_motion_frames, dtype=torch.long),  # motion history
                torch.full((self.future_motion_frames,), 2, dtype=torch.long)  # future motion (target)
            ])
            
            # 创建labels：
            # - audio位置：-100（不计算loss，模型会识别为audio token）
            # - motion history位置：-100（不计算loss，但模型会识别为motion token，因为token ID不在audio vocab范围内）
            # - future motion位置：实际的token ID（计算loss，但输入序列中是padding）
            labels = torch.full((self.max_seq_length,), -100, dtype=torch.long)
            # 只有future motion位置有label（真实token用于计算loss）
            labels[self.history_audio_frames + self.history_motion_frames:] = future_motion
            
            # 创建attention mask：所有位置都是1（没有padding）
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
            
            # 如果设置了max_samples，检查是否已达到限制
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                return
            
            self.samples.append({
                'tokens': sequence,
                'labels': labels,
                'attention_mask': attention_mask,
                'token_types': token_types,  # 0=audio, 1=motion_history, 2=future_motion
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

