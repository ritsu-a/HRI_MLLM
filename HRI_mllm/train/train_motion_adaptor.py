import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
from tqdm import tqdm
from HRI_mllm.datasets.BEATAudioMotionDataset import BEATAudioMotionDataset

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
        "epochs": 100,
        "sliding_window_step": 32,  # 滑动窗口步长（单元数）
        "pad_token_id": 8194 + 512, # 新增的填充token
        "interleave_ratio": 10,      # 每10个音频token插入1个动作token
    }
)


config = wandb.config



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
dataset = BEATAudioMotionDataset(wandb.config)
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
        attn_mask = (inputs != dataset.SEQ_PAD_TOKEN).float().to(device)
        
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
        ckpt_path = f"output/motion_adaptor/checkpoints/epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        wandb.save(ckpt_path)

# 保存最终模型
model.save_pretrained("output/motion_adaptor/audio_motion_gpt2_v1")
wandb.save("output/motion_adaptor/audio_motion_gpt2_v1/*")
