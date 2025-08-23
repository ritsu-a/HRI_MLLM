import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
from tqdm import tqdm
from HRI_mllm.datasets.BEATAudioMotionDataset import BEATAudioMotionDataset

# 初始化分布式训练环境
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

# 获取全局rank
local_rank, world_size = setup_distributed()

# 只在主进程初始化Weights & Biases
if local_rank == 0:
    wandb.init(
        project="audio-motion-BEAT-gpt2",
        config={
            "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS_kimi",
            "audio_vocab_size": 16384,
            "motion_vocab_size": 512,
            "total_vocab_size": 16384 + 512,
            "max_seq_length": 1024,
            "min_seq_length": 128,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "epochs": 100,
            "sliding_window_step": 32,
            "pad_token_id": 16384 + 512,
            "interleave_ratio": [5, 2],
        }
    )
    config = wandb.config
else:
    # 非主进程使用相同的配置
    config = type('Config', (), {
        "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS_kimi",
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512,
        "total_vocab_size": 16384 + 512,
        "max_seq_length": 1024,
        "min_seq_length": 128,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "epochs": 100,
        "sliding_window_step": 32,
        "pad_token_id": 16384 + 512,
        "interleave_ratio": [5, 2],
    })()

# 创建模型
model_config = GPT2Config(
    vocab_size=config.total_vocab_size + 1,
    n_positions=config.max_seq_length,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
model = GPT2LMHeadModel(model_config)

# 扩展位置编码
if config.max_seq_length > 1024:
    if local_rank == 0:
        print("扩展位置编码...")
    model.resize_position_embeddings(config.max_seq_length)

# 将模型移到当前GPU
device = torch.device(f'cuda:{local_rank}')
model.to(device)

# 使用DistributedDataParallel包装模型
model = torch.nn.parallel.DistributedDataParallel(
    model, 
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False  # 改为False以消除警告
)

# 只在主进程记录模型
if local_rank == 0:
    wandb.watch(model.module, log="parameters", log_freq=100)

# 创建数据集
dataset = BEATAudioMotionDataset(config)
if local_rank == 0:
    dataset.print_stats()

# 数据加载器
def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    lengths = torch.tensor([item['seq_length'] for item in batch])
    return {'tokens': tokens, 'mask': masks, 'lengths': lengths}

# 使用DistributedSampler
sampler = DistributedSampler(
    dataset, 
    num_replicas=world_size, 
    rank=local_rank,
    shuffle=True
)

dataloader = DataLoader(
    dataset, 
    batch_size=config.batch_size // world_size,
    sampler=sampler,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4  # 可以适当增加
)

# 优化器和学习率调度
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=config.learning_rate,
    steps_per_epoch=len(dataloader),
    epochs=config.epochs
)

# 训练循环（带梯度累积）
accum_steps = 4 if config.max_seq_length > 1024 else 1

for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    # 设置epoch对于DistributedSampler很重要
    sampler.set_epoch(epoch)
    
    # 只在主进程显示进度条
    if local_rank == 0:
        dataloader_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    else:
        dataloader_iter = dataloader
        
    for step, batch in enumerate(dataloader_iter):
        inputs = batch['tokens'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
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
        
        # 只在主进程记录指标
        if local_rank == 0 and step % 1000 == 0:
            log_data = {
                "train/loss": loss.item() * accum_steps,
                "train/lr": scheduler.get_last_lr()[0],
                "train/seq_length": lengths.float().mean().item(),
                "train/step": epoch * len(dataloader) + step,
                "train/memory_allocated": torch.cuda.memory_allocated(device) / 1024**3,
                "train/memory_reserved": torch.cuda.memory_reserved(device) / 1024**3,
            }
            wandb.log(log_data)
            dataloader_iter.set_postfix(loss=loss.item() * accum_steps)


    print(f"Epoch {epoch+1}/{config.epochs} | Loss: {total_loss / len(dataloader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 每个epoch结束，同步所有进程
    dist.barrier()
    
    # 只在主进程记录和保存
    if local_rank == 0:
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
                'model_state': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            wandb.save(ckpt_path)

# 保存最终模型
if local_rank == 0:
    model.module.save_pretrained("output/motion_adaptor/kimi_audio_motion_gpt2_v1")
    wandb.save("output/motion_adaptor/kimi_audio_motion_gpt2_v1/*")

# 清理分布式进程
dist.destroy_process_group()