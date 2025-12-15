import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
from tqdm import tqdm
from HRI_mllm.datasets.jsonl_audio_motion_dataset import JSONLAudioMotionDataset
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train Motion Adaptor')
parser.add_argument('--resume_from', type=str, default=None,
                   help='Checkpoint path to resume from (e.g., output/motion_adaptor_v2/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_300.pt)')
parser.add_argument('--datasets', type=str, nargs='+', default=None,
                   help='Specific datasets to use (e.g., BEAT or internet) - default: all')
parser.add_argument('--jsonl_files', type=str, nargs='+', default=None,
                   help='Direct JSONL file paths to use for training')
parser.add_argument('--epochs', type=int, default=300,
                   help='Number of epochs to train')
args = parser.parse_args()

exp_name = "kimi_audio_motion_gpt2_brainco_synthetic_en"
os.makedirs(os.path.join("output_disk0/motion_adaptor_v18", exp_name), exist_ok=True)
os.makedirs(os.path.join("output_disk0/motion_adaptor_v18", exp_name, "checkpoints"), exist_ok=True)

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
    # 单机模式
    local_rank, world_size = 0, 1
    torch.cuda.set_device(0)

# JSONL文件路径 - 每个数据集独立的jsonl文件
all_jsonl_files = {
    "BEAT": "/root/workspace/HRI_MLLM/data/BEAT_v2_1110_tokens.jsonl",
}

# 根据命令行参数选择数据集
if args.jsonl_files:
    # 如果直接指定了jsonl_files，优先使用
    jsonl_files = args.jsonl_files
elif args.datasets:
    jsonl_files = [all_jsonl_files[ds] for ds in args.datasets if ds in all_jsonl_files]
    if not jsonl_files:
        if local_rank == 0:
            print(f"❌ No valid datasets found from: {args.datasets}")
            print(f"Available datasets: {list(all_jsonl_files.keys())}")
        exit(1)
else:
    jsonl_files = list(all_jsonl_files.values())

if local_rank == 0:
    print(f"Using datasets: {args.datasets if args.datasets else 'all'}")
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
        project="audio-motion-BEAT-gpt2-adaptor",
        config={
            "jsonl_files": jsonl_files,
            "audio_vocab_size": 16384,
            "motion_vocab_size": 512*2,
            "total_vocab_size": 512*2 + 10,
            "max_seq_length": 256,
            "min_seq_length": 32,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "epochs": args.epochs,  # 使用命令行参数
            "sliding_window_step": 32,
            "pad_token_id": 512*2 + 1,
            "gesture_start_token_id": 512*2 + 2,
            "audio_gesture_start_token_id": 512*2 + 3,
            "gesture_end_token_id": 512*2 + 4,
            "audio_gesture_end_token_id": 512*2 + 5,
            "audio_empty_token_id": 152063,  # glm-voice-4 audio tokenizer的padding token
            "motion_empty_token_id": 512*2 + 7,
            "interleave_ratio": [1, 1],
            "exp_name": exp_name,
        }
    )
    config = wandb.config
else:
    config = type('Config', (), {
        "jsonl_files": jsonl_files,
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512*2,
        "total_vocab_size": 512*2 + 10,
        "max_seq_length": 256,
        "min_seq_length": 32,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "epochs": args.epochs,  # 使用命令行参数
        "sliding_window_step": 32,
        "pad_token_id": 512*2 + 1,
        "gesture_start_token_id": 512*2 + 2,
        "audio_gesture_start_token_id": 512*2 + 3,
        "gesture_end_token_id": 512*2 + 4,
        "audio_gesture_end_token_id": 512*2 + 5,
        "audio_empty_token_id": 152063,  # glm-voice-4 audio tokenizer的padding token
        "motion_empty_token_id": 512*2 + 7,
        "interleave_ratio": [1, 1],
        "exp_name": exp_name,
    })()

# 创建模型
model_config = GPT2Config(
    vocab_size=config.total_vocab_size,
    n_positions=config.max_seq_length,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)
# 添加特殊token ID到模型配置
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
    
    # 检查是否需要调整epochs
    checkpoint_epoch = checkpoint.get('epoch', 0)
    if local_rank == 0:
        print(f"✅ Loaded checkpoint from epoch {checkpoint_epoch}")
        print(f"   Resuming from epoch {start_epoch}")
        print(f"   Total epochs requested: {args.epochs}")
        if start_epoch >= args.epochs:
            print(f"⚠️  WARNING: Checkpoint epoch ({checkpoint_epoch}) >= total epochs ({args.epochs})")
            print(f"   Training will start but may complete immediately")
        print(f"{'='*80}\n")
else:
    if local_rank == 0:
        print("\nStarting training from scratch\n")

# 只在分布式模式下使用DDP
if world_size > 1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

if local_rank == 0:
    # 在DDP模式下使用model.module，否则直接使用model
    model_to_watch = model.module if world_size > 1 else model
    wandb.watch(model_to_watch, log="parameters", log_freq=100)

# 创建多个数据集并平衡采样
datasets_info = []
for jsonl_path in config.jsonl_files:
    if os.path.exists(jsonl_path):
        dataset = JSONLAudioMotionDataset(jsonl_path, config)
        datasets_info.append((dataset, jsonl_path))
        if local_rank == 0:
            print(f"Loaded {len(dataset)} samples from {jsonl_path}")
    else:
        print(f"Warning: {jsonl_path} not found, skipping...")

if len(datasets_info) == 0:
    if local_rank == 0:
        print("❌ No valid JSONL files found!")
        print(f"Searched files: {jsonl_files}")
    exit(1)

# 如果只有1个数据集，直接使用
if len(datasets_info) == 1:
    dataset = datasets_info[0][0]
else:
    # 对于多个数据集，找到最小的数据集大小
    min_size = min(len(d) for d, _ in datasets_info)
    
    if local_rank == 0:
        print(f"Balancing datasets to {min_size} samples each")
    
    # 为每个数据集创建加权随机采样子集
    balanced_datasets = []
    for dataset, jsonl_path in datasets_info:
        if len(dataset) > min_size:
            # 创建加权采样索引
            indices = list(range(len(dataset)))
            # 使用加权随机采样，保持原始分布
            selected_indices = torch.multinomial(
                torch.ones(len(dataset)), 
                min_size, 
                replacement=True
            ).tolist()
            
            # 创建子集
            subset = Subset(dataset, selected_indices)
            balanced_datasets.append(subset)
            
            if local_rank == 0:
                print(f"  {jsonl_path}: {len(dataset)} -> {len(subset)} samples")
        else:
            balanced_datasets.append(dataset)
            if local_rank == 0:
                print(f"  {jsonl_path}: {len(dataset)} samples (no sampling needed)")
    
    # 合并平衡后的数据集
    dataset = ConcatDataset(balanced_datasets)

if local_rank == 0:
    if isinstance(dataset, ConcatDataset):
        total_samples = len(dataset)
        print(f"Total samples: {total_samples} (balanced from {len(datasets_info)} files)")
    else:
        total_samples = len(dataset)
        print(f"Total samples: {total_samples}")

def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch]).long()
    masks = torch.stack([item['mask'] for item in batch]).long()
    lengths = torch.tensor([item['seq_length'] for item in batch])
    return {'tokens': tokens, 'mask': masks, 'lengths': lengths}

def dynamic_collate_fn(batch, max_seq_length=None):
    """每批自动找最长序列padding，支持max_seq_length截断"""
    tokens = [item['tokens'] for item in batch]
    mask = [item['mask'] for item in batch]
    seq_lengths = [item['seq_length'] for item in batch]
    L_max = max(seq_lengths)
    if max_seq_length is not None:
        L_max = min(L_max, max_seq_length)
    batch_tokens = []
    batch_mask = []
    for t,m,l in zip(tokens,mask,seq_lengths):
        t = t[:L_max]
        m = m[:L_max]
        # padding
        if t.shape[0] < L_max:
            pad_num = L_max-t.shape[0]
            t = torch.cat([t, torch.full((pad_num,), t[-1].item() if t.shape[0]>0 else 0, dtype=t.dtype)])
            m = torch.cat([m, torch.zeros(pad_num, dtype=m.dtype)])
        batch_tokens.append(t)
        batch_mask.append(m)
    batch_tokens = torch.stack(batch_tokens)
    batch_mask = torch.stack(batch_mask)
    return {'tokens': batch_tokens, 'mask': batch_mask, 'seq_length': torch.tensor(seq_lengths)}

# 只在分布式模式下使用DistributedSampler
if world_size > 1:
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=True
    )
else:
    sampler = None

dataloader = DataLoader(
    dataset, 
    batch_size=config.batch_size // world_size,
    sampler=sampler,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4
)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# 如果resume from checkpoint，也恢复optimizer状态
if args.resume_from and os.path.exists(args.resume_from):
    checkpoint = torch.load(args.resume_from, map_location='cpu')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=config.learning_rate,
    steps_per_epoch=len(dataloader),
    epochs=config.epochs
)

accum_steps = 4 if config.max_seq_length > 1024 else 1

# 从start_epoch开始训练
if local_rank == 0:
    print(f"\n🚀 Starting training from epoch {start_epoch} to {config.epochs}")
    print(f"   Total epochs to train: {config.epochs - start_epoch}")
    print(f"{'='*80}\n")

for epoch in range(start_epoch, config.epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    if sampler is not None:
        sampler.set_epoch(epoch)
    
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
        
        total_loss += loss.item() * accum_steps
        
        # 添加调试信息
        if local_rank == 0 and step == 0:
            print(f"📊 Epoch {epoch+1}, Step {step+1}: Loss = {loss.item() * accum_steps:.4f}")
            print(f"   Input shape: {inputs.shape}, Labels shape: {labels.shape}")
            print(f"   Attention mask shape: {attn_mask.shape}")
            print(f"   Motion mask shape: {masks.shape}")
            print(f"   Valid motion tokens: {masks.sum().item()}/{masks.numel()}")
        
        if local_rank == 0 and (step % 500 == 0 or step == len(dataloader) - 1):
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
    
    if world_size > 1:
        dist.barrier()
    
    if local_rank == 0:
        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch/loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 50 == 0:
            ckpt_path = f"output_disk0/motion_adaptor_v18/{config.exp_name}/checkpoints/epoch_{epoch+1}.pt"
            # 在DDP模式下使用model.module，否则直接使用model
            model_to_save = model.module if world_size > 1 else model
            torch.save({
                'epoch': epoch,
                'model_state': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            wandb.save(ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

if local_rank == 0:
    # 在DDP模式下使用model.module，否则直接使用model
    model_to_save = model.module if world_size > 1 else model
    model_to_save.save_pretrained(f"output_disk0/motion_adaptor_v18/{config.exp_name}")

if world_size > 1:
    dist.destroy_process_group()