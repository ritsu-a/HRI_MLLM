import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
import json
from tqdm import tqdm
from HRI_mllm.datasets.BEATAudioMotionDataset import BEATAudioMotionDataset
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

class JSONLAudioMotionDataset(Dataset):
    """从JSONL文件加载音频-动作数据的Dataset"""
    
    def __init__(self, jsonl_path, config):
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
                    
                    # 构建完整序列
                    full_sequence = []
                    token_types = []
                    
                    for i in range(len(audio_tokens)):
                        full_sequence.append(audio_tokens[i].item())
                        token_types.append(0)
                        
                        if (i + 1) % self.interleave_audios == 0:
                            motion_idx = i // self.interleave_audios * self.interleave_motions
                            for j in range(self.interleave_motions):
                                if motion_idx + j < len(motion_tokens):
                                    full_sequence.append(motion_tokens[motion_idx + j].item())
                                    token_types.append(1)
                    
                    # 应用滑动窗口
                    self.apply_sliding_window(full_sequence, token_types)
                    self.stats['total_sequences'] += 1
                    
                except Exception as e:
                    print(f"Error processing line {line_num} in {jsonl_path}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def apply_sliding_window(self, full_seq, token_types):
        seq_len = len(full_seq)
        
        if seq_len > self.config.max_seq_length:
            return
        
        sub_seq = full_seq
        sub_types = token_types
        
        mask = [1 if t == 1 else 0 for t in sub_types]
        
        padded_seq = sub_seq + [self.SEQ_PAD_TOKEN] * (self.config.max_seq_length - len(sub_seq))
        padded_mask = mask + [0] * (self.config.max_seq_length - len(mask))
        
        self.samples.append({
            'tokens': torch.tensor(padded_seq),
            'mask': torch.tensor(padded_mask),
            'seq_length': len(sub_seq)
        })
        
        self.stats['generated_samples'] += 1
        self.stats['max_length'] = max(self.stats['max_length'], len(sub_seq))
    
    def pool_and_concat_samples(self, sep_token):
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
    
    # 删除 pool_and_concat_samples 相关调用，不做预处理拼接
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train Motion Adaptor')
parser.add_argument('--resume_from', type=str, default=None,
                   help='Checkpoint path to resume from (e.g., output/motion_adaptor_v2/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_300.pt)')
parser.add_argument('--datasets', type=str, nargs='+', default=None,
                   help='Specific datasets to use (e.g., BEAT or internet) - default: all')
parser.add_argument('--epochs', type=int, default=300,
                   help='Number of epochs to train')
args = parser.parse_args()

exp_name = "kimi_audio_motion_gpt2_brainco_synthetic_en"
os.makedirs(os.path.join("output/motion_adaptor_v5", exp_name), exist_ok=True)
os.makedirs(os.path.join("output/motion_adaptor_v5", exp_name, "checkpoints"), exist_ok=True)

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
    "BEAT": "/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl",
    "internet": "/root/workspace/HRI_MLLM/data/internet_data_v1_kimi_tokens.jsonl",
    "SG_2_or_3_long_sentence_1030_en_kimi_tokens": "/root/workspace/HRI_MLLM/data/SG_2_or_3_long_sentence_1030_en_kimi_tokens.jsonl",
}

# 根据命令行参数选择数据集
if args.datasets:
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
            "max_seq_length": 4096,
            "min_seq_length": 128,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "epochs": args.epochs,  # 使用命令行参数
            "sliding_window_step": 32,
            "pad_token_id": 512*2 + 1,
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
        "max_seq_length": 4096,
        "min_seq_length": 128,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": args.epochs,  # 使用命令行参数
        "sliding_window_step": 32,
        "pad_token_id": 512*2 + 1,
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
            ckpt_path = f"output/motion_adaptor_v5/{config.exp_name}/checkpoints/epoch_{epoch+1}.pt"
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
    model_to_save.save_pretrained(f"output/motion_adaptor_v5/{config.exp_name}")

if world_size > 1:
    dist.destroy_process_group()