import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
from collections import defaultdict
from tqdm import tqdm

from HRI_mllm.datasets.BEATAudioMotionDataset import BEATAudioMotionDataset
from types import SimpleNamespace

# 初始化

config={
        "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS",
        "audio_vocab_size": 8194,
        "motion_vocab_size": 512,
        "total_vocab_size": 8194 + 512,
        "max_seq_length": 1024,    # 模型支持的最大长度
        "min_seq_length": 128,     # 最小序列长度
        "batch_size": 8,
        "learning_rate": 5e-5,
        "epochs": 100,
        "sliding_window_step": 32,  # 滑动窗口步长（单元数）
        "pad_token_id": 8194 + 512, # 新增的填充token
        "interleave_ratio": 10,      # 每10个音频token插入1个动作token
    }
config = SimpleNamespace(**config)


model = GPT2LMHeadModel.from_pretrained("output/motion_adaptor/audio_motion_gpt2")
model.to("cuda" if torch.cuda.is_available() else "cpu")


# 创建数据集
dataset = BEATAudioMotionDataset(config)
dataset.print_stats()


# # 长序列推理函数
def generate_for_long_audio(audio_tokens, model, device, max_length=3000):
    model.eval()
    generated = []
    current_seq = []
    motion_count = 0
    max_context = max_length - 50  # 保留空间生成新token
    
    with torch.no_grad():
        for i, token in enumerate(audio_tokens):
            current_seq.append(token)
            
            # 每5个audio token尝试生成motion
            if (i + 1) % config.interleave_ratio == 0:
                # 当序列过长时使用滑动窗口
                if len(current_seq) > max_context:
                    # 保留最近的完整上下文
                    keep_from = max(0, len(current_seq) - max_context)
                    # 确保从完整单元开始
                    while keep_from < len(current_seq) and (keep_from % (config.interleave_ratio+1) != 0):
                        keep_from += 1
                    current_seq = current_seq[keep_from:]
                
                inputs = torch.tensor([current_seq]).to(device)
                attn_mask = torch.ones_like(inputs).float().to(device)
                
                # 预测下一个motion token
                output = model(inputs, attention_mask=attn_mask)
                next_token_logits = output.logits[0, -1, :]
                
                # 限制在motion词表范围内
                motion_logits = next_token_logits[config.audio_vocab_size:]
                next_token = torch.argmax(motion_logits).item() + config.audio_vocab_size
                
                generated.append(next_token)
                current_seq.append(next_token)  # 添加到上下文
                motion_count += 1
            
            if len(current_seq) >= max_length:
                break
    
    # 提取生成的motion tokens
    motion_tokens = [t - config.audio_vocab_size for t in generated]
    return motion_tokens

# # 示例使用
# long_audio = np.random.randint(0, config.audio_vocab_size-2, 2500)  # 2500个audio token
data_idx = 0
audio_tokens = dataset[data_idx]['tokens'][dataset[data_idx]['mask']==0]

motion_output = generate_for_long_audio(audio_tokens, model, device=model.device)
import ipdb;ipdb.set_trace()


# print(f"Generated {len(motion_output)} motion tokens")




