"""
音频-动作未来预测数据集
任务：输入 = 3*padding + 25*past_motion + 50*future_audio + 50*future_motion_padding = 128
      输出 = 50*future_motion（作为监督）
"""

import torch
from torch.utils.data import Dataset
import json
import os


class AudioMotionFuturePredictionDataset(Dataset):
    """
    音频-动作未来预测数据集
    
    输入格式：
    - 3帧padding
    - 25帧past motion token
    - 50帧future audio token
    - 50帧future motion padding
    预测目标：
    - 50帧future motion token（作为监督）
    
    输入序列长度：128 (3 + 25 + 50 + 50)
    总序列长度：178 (包含输出监督的50帧)
    """
    
    def __init__(self, jsonl_path, config, max_samples=None):
        """
        Args:
            jsonl_path: JSONL文件路径
            config: 配置对象，需要包含以下属性：
                - pad_token_id: padding token ID
                - audio_empty_token_id: audio empty token ID
                - motion_empty_token_id: motion empty token ID
                - padding_frames: 初始padding帧数（默认3）
                - past_motion_frames: 历史motion帧数（默认25）
                - future_audio_frames: 未来audio帧数（默认50）
                - future_motion_frames: 未来motion帧数（默认50，输入序列中对应位置是padding，labels中是真实token）
            max_samples: 最大样本数量（用于调试，None表示加载全部）
        """
        self.config = config
        self.jsonl_path = jsonl_path
        self.samples = []
        self.max_samples = max_samples
        
        # 任务参数
        self.padding_frames = getattr(config, 'padding_frames', 3)
        self.past_motion_frames = getattr(config, 'past_motion_frames', 25)
        self.future_audio_frames = getattr(config, 'future_audio_frames', 50)
        self.future_motion_frames = getattr(config, 'future_motion_frames', 50)
        # 输入序列长度：3 + 25 + 50 + 50 = 128
        # 总序列长度：3 + 25 + 50 + 50 + 50 = 178（包含输出监督的50帧）
        self.max_seq_length = self.padding_frames + self.past_motion_frames + self.future_audio_frames + self.future_motion_frames + self.future_motion_frames
        
        self.pad_token_id = config.pad_token_id
        self.audio_empty_token_id = config.audio_empty_token_id
        self.motion_empty_token_id = config.motion_empty_token_id
        
        print(f"Loading JSONL file: {jsonl_path}")
        print(f"Task configuration:")
        print(f"  - Padding frames: {self.padding_frames}")
        print(f"  - Past motion frames: {self.past_motion_frames}")
        print(f"  - Future audio frames: {self.future_audio_frames}")
        print(f"  - Future motion padding frames: {self.future_motion_frames}")
        print(f"  - Future motion frames (supervision): {self.future_motion_frames}")
        print(f"  - Input sequence length: {self.padding_frames + self.past_motion_frames + self.future_audio_frames + self.future_motion_frames}")
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
        - 输入序列：[3*padding] + [25*past_motion] + [50*future_audio] + [50*motion_padding] + [50*future_motion_padding]
        - 输出监督：只有最后50个future_motion位置有真实token，其他位置都是-100
        """
        audio_len = len(audio_tokens)
        motion_len = len(motion_tokens)
        
        # 计算可以创建多少个样本
        # 需要确保有足够的past motion和future motion数据
        min_required_motion = self.past_motion_frames + self.future_motion_frames
        
        if motion_len < min_required_motion:
            # 如果motion序列太短，跳过
            return
        
        # 使用滑动窗口，步长为1（可以重叠）
        window_step = getattr(self.config, 'window_step', 1)
        
        # 对于每个可能的起始位置
        max_start_idx = motion_len - min_required_motion
        
        for start_motion_idx in range(0, max_start_idx + 1, window_step):
            # 提取past motion（历史motion）
            past_motion = motion_tokens[start_motion_idx:start_motion_idx + self.past_motion_frames]
            
            # 提取future motion（预测目标）
            future_start_idx = start_motion_idx + self.past_motion_frames
            future_end_idx = future_start_idx + self.future_motion_frames
            
            if future_end_idx > motion_len:
                # 如果future motion不足，跳过
                continue
            
            future_motion = motion_tokens[future_start_idx:future_end_idx]
            
            # 提取future audio（未来audio，用于预测）
            # 假设audio和motion大致对齐，future_audio应该对应future_motion的时间范围
            # 计算对应的audio起始位置
            audio_start_idx = max(0, future_start_idx)
            audio_end_idx = min(audio_start_idx + self.future_audio_frames, audio_len)
            
            if audio_end_idx - audio_start_idx < self.future_audio_frames:
                # 如果audio不足，在后面padding
                future_audio = torch.full((self.future_audio_frames,), self.audio_empty_token_id, dtype=audio_tokens.dtype)
                available_audio = audio_tokens[audio_start_idx:audio_end_idx]
                future_audio[:len(available_audio)] = available_audio
            else:
                future_audio = audio_tokens[audio_start_idx:audio_end_idx]
            
            # 构建完整序列：
            # [3*padding] + [25*past_motion] + [50*future_audio] + [50*future_motion_padding] + [50*future_motion_padding（输出监督位置，输入中是padding）]
            # 重要：future_motion位置应该用padding，而不是真实token！
            initial_padding = torch.full((self.padding_frames,), self.pad_token_id, dtype=motion_tokens.dtype)
            future_motion_padding = torch.full((self.future_motion_frames,), self.motion_empty_token_id, dtype=motion_tokens.dtype)
            # 输出监督位置的padding（输入序列中这部分也是padding）
            future_motion_supervision_padding = torch.full((self.future_motion_frames,), self.motion_empty_token_id, dtype=motion_tokens.dtype)
            
            sequence = torch.cat([
                initial_padding,      # 3帧padding
                past_motion,           # 25帧past motion
                future_audio,         # 50帧future audio
                future_motion_padding,  # 50帧future motion padding（输入序列的一部分）
                future_motion_supervision_padding  # 50帧future motion padding（输出监督位置，输入中也是padding）
            ])
            
            # 创建labels：
            # - 只有最后50个future_motion位置有真实token（用于计算loss）
            # - 其他位置都是-100（不计算loss）
            labels = torch.full((self.max_seq_length,), -100, dtype=torch.long)
            # 计算future_motion在序列中的起始位置（在最后50个位置）
            future_motion_start_idx = self.padding_frames + self.past_motion_frames + self.future_audio_frames + self.future_motion_frames
            labels[future_motion_start_idx:] = future_motion
            
            # 创建attention mask：所有位置都是1（没有padding）
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
            
            # 如果设置了max_samples，检查是否已达到限制
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                return
            
            self.samples.append({
                'tokens': sequence,
                'labels': labels,
                'attention_mask': attention_mask,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

