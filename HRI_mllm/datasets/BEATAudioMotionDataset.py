from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch 
import numpy as np
import os

class BEATAudioMotionDataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.samples = []
        self.stats = defaultdict(int)  # 统计信息
        self.data_root = config.beat_tts_root
        self.interleave_audios, self.interleave_motions = config.interleave_ratio  # 每5个音频token插入2个动作token


        self.SEQ_PAD_TOKEN = config.pad_token_id             # 序列填充token

        with open(f"{self.data_root}/all.txt", "r") as f:
            audio_files = f.readlines()
        
        for audio_file in audio_files:

            audio_file = audio_file.strip()
            if not audio_file:
                continue
            
            audio_token_save_path = audio_file
            motion_token_save_path = audio_file.replace('audio_tokens.pt', 'motion_tokens.pt')
            
            if not (os.path.exists(audio_token_save_path) and os.path.exists(motion_token_save_path)):
                continue
            
            # 读取音频和动作token
            audio_tokens = torch.load(audio_token_save_path).squeeze(0)
            motion_tokens = torch.load(motion_token_save_path).squeeze(0)    



            # 构建完整序列和掩码
            full_sequence = []
            token_types = []  # 0=audio, 1=motion
            for i in range(len(audio_tokens)):
                full_sequence.append(audio_tokens[i])
                token_types.append(0)
                
                if (i + 1) % self.interleave_audios == 0 and i // self.interleave_audios * self.interleave_motions + self.interleave_motions - 1  < len(motion_tokens):
                    for _ in range(self.interleave_motions):
                        full_sequence.append(motion_tokens[i//self.interleave_audios * self.interleave_motions + _])
                        token_types.append(1)
            
            # 存储原始长序列
            self.stats['total_sequences'] += 1
            self.stats['max_length'] = max(self.stats['max_length'], len(full_sequence))


            
            # 应用滑动窗口裁剪
            self.apply_sliding_window(full_sequence, token_types)
    
    def apply_sliding_window(self, full_seq, token_types):

        seq_len = len(full_seq)

        if seq_len > self.config.max_seq_length:
            return  # 跳过过长序列

       
        sub_seq = full_seq
        sub_types = token_types
        
        # 创建掩码（只计算motion位置的损失）
        mask = [1 if t == 1 else 0 for t in sub_types]
        
        # 填充到统一长度
        padded_seq = sub_seq + [self.SEQ_PAD_TOKEN] * (self.config.max_seq_length - len(sub_seq))
        padded_mask = mask + [0] * (self.config.max_seq_length - len(mask))
        
        self.samples.append({
            'tokens': torch.tensor(padded_seq),
            'mask': torch.tensor(padded_mask),
            'seq_length': len(sub_seq)  # 实际有效长度
        })
        
        self.stats['generated_samples'] += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def print_stats(self):
        print(f"数据集统计:")
        print(f"- 原始长序列数: {self.stats['total_sequences']}")
        print(f"- 生成样本数: {self.stats['generated_samples']}")
        print(f"- 最长原始序列: {self.stats['max_length']} tokens")
        print(f"- 平均样本长度: {sum(s['seq_length'] for s in self.samples)/len(self.samples):.1f} tokens")