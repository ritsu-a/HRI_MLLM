import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
from collections import defaultdict
from tqdm import tqdm

# 初始化Weights & Biases
wandb.init(
    project="audio-motion-BEAT-gpt2",
    config={
        "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS",
        "audio_vocab_size": 8194,
        "motion_vocab_size": 512,
        "total_vocab_size": 8194 + 512,
        "max_seq_length": 1024,    # 模型支持的最大长度
        "min_seq_length": 128,     # 最小序列长度
        "batch_size": 8,
        "learning_rate": 5e-5,
        "epochs": 10,
        "sliding_window_step": 32,  # 滑动窗口步长（单元数）
        "pad_token_id": 8194 + 512, # 新增的填充token
    }
)
config = wandb.config



# 特殊token定义
SEQ_PAD_TOKEN = config.pad_token_id             # 序列填充token

class BEATAudioMotionDataset(Dataset):
    def __init__(self, data_root=config.beat_tts_root):
        self.samples = []
        self.stats = defaultdict(int)  # 统计信息
        self.data_root = data_root
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
                
                if (i + 1) % 5 == 0 and i // 5 < len(motion_tokens):
                    full_sequence.append(motion_tokens[i//5] + config.audio_vocab_size)
                    token_types.append(1)
            
            # 存储原始长序列
            self.stats['total_sequences'] += 1
            self.stats['max_length'] = max(self.stats['max_length'], len(full_sequence))
            
            # 应用滑动窗口裁剪
            self.apply_sliding_window(full_sequence, token_types)
    
    def apply_sliding_window(self, full_seq, token_types):
        """将长序列分割为多个子序列"""
        seq_len = len(full_seq)
        unit_size = 6  # 5 audio + 1 motion
        
        # 计算最大单元数（基于模型最大长度）
        max_units = config.max_seq_length // unit_size
        step_units = config.sliding_window_step
        
        # 随机起始偏移（增加数据多样性）
        start_offset = np.random.randint(0, step_units) if seq_len > config.max_seq_length else 0
        
        # 滑动窗口裁剪
        for start_idx in range(start_offset, seq_len, step_units * unit_size):
            end_idx = min(start_idx + max_units * unit_size, seq_len)
            
            # 确保窗口以motion token结束（保持完整单元）
            while end_idx > start_idx and token_types[end_idx-1] != 1:
                end_idx -= 1
                
            if end_idx - start_idx < config.min_seq_length:
                continue  # 跳过太短的序列
                
            # 截取子序列
            sub_seq = full_seq[start_idx:end_idx]
            sub_types = token_types[start_idx:end_idx]
            
            # 创建掩码（只计算motion位置的损失）
            mask = [1 if t == 1 else 0 for t in sub_types]
            
            # 填充到统一长度
            padded_seq = sub_seq + [SEQ_PAD_TOKEN] * (config.max_seq_length - len(sub_seq))
            padded_mask = mask + [0] * (config.max_seq_length - len(mask))
            
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

# 创建模型（扩展词表）
model_config = GPT2Config(
    vocab_size=config.total_vocab_size + 1,  # +1 for pad token
    n_positions=config.max_seq_length,
    n_embd=768,
    n_layer=12,          # 增加层数处理长序列
    n_head=12,
    n_inner=3072,        # 增加FFN维度
    resid_pdrop=0.1,      # 正则化
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
model = GPT2LMHeadModel(model_config)

# 扩展位置编码（处理长序列的关键）
if config.max_seq_length > 1024:
    print("扩展位置编码...")
    model.resize_position_embeddings(config.max_seq_length)

# 记录模型
wandb.watch(model, log="parameters", log_freq=100)

# 创建数据集
dataset = BEATAudioMotionDataset()
dataset.print_stats()

# 数据加载器（带填充处理）
def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    lengths = torch.tensor([item['seq_length'] for item in batch])
    return {'tokens': tokens, 'mask': masks, 'lengths': lengths}

dataloader = DataLoader(dataset, batch_size=config.batch_size, 
                        shuffle=True, collate_fn=collate_fn)

# 优化器和学习率调度
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=config.learning_rate,
    steps_per_epoch=len(dataloader),
    epochs=config.epochs
)

# 训练循环（带梯度累积）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 梯度累积步骤（处理长序列）
accum_steps = 4 if config.max_seq_length > 1024 else 1

for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for step, batch in tqdm(enumerate(dataloader)):
        inputs = batch['tokens'].to(device)
        masks = batch['mask'].to(device)
        lengths = batch['lengths']
        
        # 创建注意力掩码（忽略填充位置）
        attn_mask = (inputs != SEQ_PAD_TOKEN).float().to(device)
        
        # 创建标签
        labels = inputs.clone()
        labels[masks == 0] = -100  # 只计算motion位置的损失
        
        # 模型前向
        outputs = model(
            inputs, 
            labels=labels,
            attention_mask=attn_mask
        )
        loss = outputs.loss / accum_steps  # 梯度累积
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (step + 1) % accum_steps == 0:
            # 梯度裁剪（防止长序列梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        
        # 记录指标
        if step % 10 == 0:
            wandb.log({
                "train/loss": loss.item() * accum_steps,
                "train/lr": scheduler.get_last_lr()[0],
                "train/seq_length": lengths.float().mean().item(),
                "train/step": epoch * len(dataloader) + step
            })
    
    # 每个epoch结束
    avg_loss = total_loss / len(dataloader)
    wandb.log({
        "epoch/loss": avg_loss,
        "epoch": epoch
    })
    print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")
    
    # 保存检查点
    if (epoch + 1) % 2 == 0:
        ckpt_path = f"checkpoints/epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        wandb.save(ckpt_path)

# 保存最终模型
model.save_pretrained("audio_motion_gpt2_long")
wandb.save("audio_motion_gpt2_long/*")

# # 长序列推理函数
# def generate_for_long_audio(audio_tokens, model, device, max_length=3000):
#     model.eval()
#     generated = []
#     current_seq = []
#     motion_count = 0
#     max_context = config.max_seq_length - 50  # 保留空间生成新token
    
#     with torch.no_grad():
#         for i, token in enumerate(audio_tokens):
#             current_seq.append(token)
            
#             # 每5个audio token尝试生成motion
#             if (i + 1) % 5 == 0:
#                 # 当序列过长时使用滑动窗口
#                 if len(current_seq) > max_context:
#                     # 保留最近的完整上下文
#                     keep_from = max(0, len(current_seq) - max_context)
#                     # 确保从完整单元开始
#                     while keep_from < len(current_seq) and (keep_from % 6 != 0):
#                         keep_from += 1
#                     current_seq = current_seq[keep_from:]
                
#                 inputs = torch.tensor([current_seq]).to(device)
#                 attn_mask = torch.ones_like(inputs).float().to(device)
                
#                 # 预测下一个motion token
#                 output = model(inputs, attention_mask=attn_mask)
#                 next_token_logits = output.logits[0, -1, :]
                
#                 # 限制在motion词表范围内
#                 motion_logits = next_token_logits[config.audio_vocab_size:]
#                 next_token = torch.argmax(motion_logits).item() + config.audio_vocab_size
                
#                 generated.append(next_token)
#                 current_seq.append(next_token)  # 添加到上下文
#                 motion_count += 1
            
#             if len(current_seq) >= max_length:
#                 break
    
#     # 提取生成的motion tokens
#     motion_tokens = [t - config.audio_vocab_size for t in generated]
#     return motion_tokens

# # 示例使用
# long_audio = np.random.randint(0, config.audio_vocab_size-2, 2500)  # 2500个audio token
# motion_output = generate_for_long_audio(long_audio, model, device)

# print(f"Generated {len(motion_output)} motion tokens")
# wandb.log({
#     "generated_motion": wandb.Histogram(motion_output),
#     "input_audio": wandb.Histogram(long_audio)
# })

# wandb.finish()