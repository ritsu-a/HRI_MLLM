"""
JSONL格式的音频-动作数据集加载器
用于训练motion adaptor模型
"""

import torch
from torch.utils.data import Dataset
import json


class JSONLAudioMotionDataset(Dataset):
    """从JSONL文件加载音频-动作数据的Dataset"""
    
    def __init__(self, jsonl_path, config):
        """
        Args:
            jsonl_path: JSONL文件路径
            config: 配置对象，需要包含以下属性：
                - interleave_ratio: [audio_ratio, motion_ratio] 交错比例
                - pad_token_id: padding token ID
                - max_seq_length: 最大序列长度
                - audio_empty_token_id: audio empty token ID (glm-voice-4的padding token)
                - motion_empty_token_id: motion empty token ID
        """
        self.config = config
        self.jsonl_path = jsonl_path
        self.samples = []
        self.stats = {'total_sequences': 0, 'generated_samples': 0, 'max_length': 0}
        self.interleave_audios, self.interleave_motions = config.interleave_ratio
        self.SEQ_PAD_TOKEN = config.pad_token_id
        
        print(f"Loading JSONL file: {jsonl_path}")
        
        # 读取JSONL文件
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
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
                    
                    # 在gt的audio token序列后添加10个audio_empty_token
                    audio_empty_tokens = torch.full((10,), self.config.audio_empty_token_id, dtype=audio_tokens.dtype)
                    audio_tokens = torch.cat([audio_tokens, audio_empty_tokens])
                    
                    # 在gt的motion token序列前添加10个motion_empty_token
                    motion_empty_tokens = torch.full((10,), self.config.motion_empty_token_id, dtype=motion_tokens.dtype)
                    motion_tokens = torch.cat([motion_empty_tokens, motion_tokens])
                    
                    # 构建完整序列
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
                                    # 插入motion token
                                    full_sequence.append(motion_tokens[motion_token_idx].item())
                                    token_types.append(1)
                                    motion_token_idx += 1
                    
                    # 处理剩余的motion tokens（如果有的话）
                    while motion_token_idx < len(motion_tokens):
                        # 插入motion token
                        full_sequence.append(motion_tokens[motion_token_idx].item())
                        token_types.append(1)
                        motion_token_idx += 1
                    
                    # 应用滑动窗口
                    self.apply_sliding_window(full_sequence, token_types)
                    self.stats['total_sequences'] += 1
                    
                except Exception as e:
                    print(f"Error processing line {line_num} in {jsonl_path}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def apply_sliding_window(self, full_seq, token_types):
        """
        应用滑动窗口策略：
        - 序列长度 < 64: 在序列前补充padding到64（保持history token位置固定）
        - 序列长度 >= 64: 使用滑动窗口切分成多个64长度的窗口
        """
        seq_len = len(full_seq)
        max_seq_length = self.config.max_seq_length  # 64
        
        if seq_len < max_seq_length:
            # 序列长度小于64，在序列前补充padding
            pad_len = max_seq_length - seq_len
            padded_seq = [self.SEQ_PAD_TOKEN] * pad_len + full_seq
            padded_types = [0] * pad_len + token_types  # padding部分token_type为0
            mask = [1 if t == 1 else 0 for t in padded_types]
            
            self.samples.append({
                'tokens': torch.tensor(padded_seq),
                'mask': torch.tensor(mask),
                'seq_length': seq_len  # 实际有效序列长度
            })
            
            self.stats['generated_samples'] += 1
            self.stats['max_length'] = max(self.stats['max_length'], seq_len)
        else:
            # 序列长度 >= 64，使用滑动窗口切分
            # 使用滑动窗口步长（如果配置了），否则使用窗口大小（不重叠）
            window_step = getattr(self.config, 'sliding_window_step', max_seq_length)
            
            start_idx = 0
            while start_idx < seq_len:
                end_idx = min(start_idx + max_seq_length, seq_len)
                
                # 提取窗口
                window_seq = full_seq[start_idx:end_idx]
                window_types = token_types[start_idx:end_idx]
                
                # 如果窗口长度小于max_seq_length，在前面补充padding
                if len(window_seq) < max_seq_length:
                    pad_len = max_seq_length - len(window_seq)
                    window_seq = [self.SEQ_PAD_TOKEN] * pad_len + window_seq
                    window_types = [0] * pad_len + window_types
                
                mask = [1 if t == 1 else 0 for t in window_types]
                
                self.samples.append({
                    'tokens': torch.tensor(window_seq),
                    'mask': torch.tensor(mask),
                    'seq_length': end_idx - start_idx  # 实际有效窗口长度
                })
                
                self.stats['generated_samples'] += 1
                self.stats['max_length'] = max(self.stats['max_length'], end_idx - start_idx)
                
                # 移动到下一个窗口
                start_idx += window_step
                
                # 如果剩余长度小于窗口步长，且已经处理了至少一个窗口，可以提前结束
                if start_idx >= seq_len:
                    break
    
    def pool_and_concat_samples(self, sep_token):
        """将多个样本合并成一个序列（可选功能）"""
        max_seq = self.config.max_seq_length
        pad_token = self.SEQ_PAD_TOKEN
        processed_samples = []
        cur_seq, cur_mask = [], []
        for idx, item in enumerate(self.samples):
            seq = item['tokens'].tolist()
            mask = item['mask'].tolist()
            valid_len = item['seq_length']
            data = seq[:valid_len]
            mask_data = mask[:valid_len]
            # 若加本样本+1分隔后超max，先flush已有
            if cur_seq and len(cur_seq) + 1 + len(data) > max_seq:
                pad_needed = max_seq - len(cur_seq)
                padded = cur_seq + [pad_token]*pad_needed
                padded_mask = cur_mask + [0]*pad_needed
                processed_samples.append({'tokens': torch.tensor(padded), 'mask': torch.tensor(padded_mask), 'seq_length': len(cur_seq)})
                cur_seq, cur_mask = [], []
            # 每个样本段前加分割符
            if cur_seq:  # 非开头才加
                cur_seq.append(sep_token)
                cur_mask.append(0)
            cur_seq.extend(data)
            cur_mask.extend(mask_data)
        # flush最后一批
        if cur_seq:
            pad_needed = max_seq - len(cur_seq)
            padded = cur_seq + [pad_token]*pad_needed
            padded_mask = cur_mask + [0]*pad_needed
            processed_samples.append({'tokens': torch.tensor(padded), 'mask': torch.tensor(padded_mask), 'seq_length': len(cur_seq)})
        self.samples = processed_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


