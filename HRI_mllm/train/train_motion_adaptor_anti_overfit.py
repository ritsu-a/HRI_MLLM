import torch
import torch.distributed as dist
import torch.nn.functional as F
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
parser = argparse.ArgumentParser(description='Train Motion Adaptor with Anti-Overfitting')
parser.add_argument('--resume_from', type=str, default=None,
                   help='Checkpoint path to resume from')
parser.add_argument('--datasets', type=str, nargs='+', default=None,
                   help='Specific datasets to use (e.g., BEAT or internet) - default: all')
parser.add_argument('--jsonl_files', type=str, nargs='+', default=None,
                   help='Direct JSONL file paths to use for training')
parser.add_argument('--val_jsonl_files', type=str, nargs='+', default=None,
                   help='Validation JSONL file paths (e.g., synthetic_data_en_tokens_test.jsonl)')
parser.add_argument('--epochs', type=int, default=500,  # 减少默认epoch数
                   help='Number of epochs to train')
parser.add_argument('--version', type=str, default='v18',
                   help='Model version identifier (e.g., v18, v19)')
parser.add_argument('--weight_decay', type=float, default=0.01,  # 添加weight decay
                   help='Weight decay for optimizer')
parser.add_argument('--label_smoothing', type=float, default=0.1,  # 添加label smoothing
                   help='Label smoothing factor (0.0-1.0)')
parser.add_argument('--dropout', type=float, default=0.15,  # 增加dropout
                   help='Dropout rate (applied to resid_pdrop, embd_pdrop, attn_pdrop)')
parser.add_argument('--early_stop_patience', type=int, default=20,  # 早停耐心值
                   help='Early stopping patience (number of epochs without improvement)')
parser.add_argument('--early_stop_min_delta', type=float, default=0.001,  # 最小改进阈值
                   help='Minimum delta for early stopping')
parser.add_argument('--save_best_only', action='store_true',  # 只保存最佳模型
                   help='Only save best model based on validation loss')
parser.add_argument('--val_freq', type=int, default=50,  # 验证频率
                   help='Validation frequency (validate every N epochs, default: 50, set to 0 to disable)')
parser.add_argument('--no_validation', action='store_true',  # 完全禁用验证
                   help='Disable validation completely')
args = parser.parse_args()

exp_name = "kimi_audio_motion_gpt2_brainco_synthetic_en"
model_version = args.version
os.makedirs(os.path.join(f"output_disk0/motion_adaptor_{model_version}", exp_name), exist_ok=True)
os.makedirs(os.path.join(f"output_disk0/motion_adaptor_{model_version}", exp_name, "checkpoints"), exist_ok=True)

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

# 验证集文件路径
val_jsonl_files = args.val_jsonl_files
if val_jsonl_files is None:
    # 自动检测验证集：从训练集文件名推断
    val_jsonl_files = []
    for train_file in jsonl_files:
        if 'train' in train_file:
            val_file = train_file.replace('train', 'test')
            if os.path.exists(val_file):
                val_jsonl_files.append(val_file)
        elif 'synthetic_data_en_tokens_train.jsonl' in train_file:
            val_file = train_file.replace('train', 'test')
            if os.path.exists(val_file):
                val_jsonl_files.append(val_file)

if local_rank == 0:
    print(f"Using datasets: {args.datasets if args.datasets else 'all'}")
    print(f"Training JSONL files: {jsonl_files}")
    print(f"Validation JSONL files: {val_jsonl_files}")
    
# 检查文件是否存在
for jsonl_file in jsonl_files:
    if not os.path.exists(jsonl_file):
        if local_rank == 0:
            print(f"❌ Training JSONL file not found: {jsonl_file}")
        exit(1)

for jsonl_file in val_jsonl_files:
    if not os.path.exists(jsonl_file):
        if local_rank == 0:
            print(f"⚠️  Validation JSONL file not found: {jsonl_file}, skipping validation")

# 只在主进程初始化Weights & Biases
if local_rank == 0:
    wandb.init(
        project="audio-motion-BEAT-gpt2-adaptor",
        config={
            "jsonl_files": jsonl_files,
            "val_jsonl_files": val_jsonl_files,
            "audio_vocab_size": 16384,
            "motion_vocab_size": 512*2,
            "total_vocab_size": 512*2 + 10,
            "max_seq_length": 64,
            "min_seq_length": 32,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "dropout": args.dropout,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "val_freq": 0 if args.no_validation else args.val_freq,
            "no_validation": args.no_validation,
            "sliding_window_step": 32,
            "pad_token_id": 512*2 + 1,
            "gesture_start_token_id": 512*2 + 2,
            "audio_gesture_start_token_id": 512*2 + 3,
            "gesture_end_token_id": 512*2 + 4,
            "audio_gesture_end_token_id": 512*2 + 5,
            "audio_empty_token_id": 152063,
            "motion_empty_token_id": 512*2 + 7,
            "interleave_ratio": [1, 1],
            "exp_name": exp_name,
        }
    )
    config = wandb.config
else:
    config = type('Config', (), {
        "jsonl_files": jsonl_files,
        "val_jsonl_files": val_jsonl_files,
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512*2,
        "total_vocab_size": 512*2 + 10,
        "max_seq_length": 64,
        "min_seq_length": 32,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "val_freq": 0 if args.no_validation else args.val_freq,
        "no_validation": args.no_validation,
        "sliding_window_step": 32,
        "pad_token_id": 512*2 + 1,
        "gesture_start_token_id": 512*2 + 2,
        "audio_gesture_start_token_id": 512*2 + 3,
        "gesture_end_token_id": 512*2 + 4,
        "audio_gesture_end_token_id": 512*2 + 5,
        "audio_empty_token_id": 152063,
        "motion_empty_token_id": 512*2 + 7,
        "interleave_ratio": [1, 1],
        "exp_name": exp_name,
    })()

# 创建模型（增强dropout）
model_config = GPT2Config(
    vocab_size=config.total_vocab_size,
    n_positions=64,
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    resid_pdrop=config.dropout,  # 使用配置的dropout
    embd_pdrop=config.dropout,
    attn_pdrop=config.dropout,
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
best_val_loss = float('inf')
patience_counter = 0

if args.resume_from and os.path.exists(args.resume_from):
    if local_rank == 0:
        print(f"\n{'='*80}")
        print(f"🔄 Resuming from checkpoint: {args.resume_from}")
        print(f"{'='*80}")
    
    checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    patience_counter = checkpoint.get('patience_counter', 0)
    
    if local_rank == 0:
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Patience counter: {patience_counter}")
        print(f"   Resuming from epoch {start_epoch}")
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
    model_to_watch = model.module if world_size > 1 else model
    wandb.watch(model_to_watch, log="parameters", log_freq=100)

# 创建训练数据集
train_datasets_info = []
for jsonl_path in config.jsonl_files:
    if os.path.exists(jsonl_path):
        dataset = JSONLAudioMotionDataset(jsonl_path, config)
        train_datasets_info.append((dataset, jsonl_path))
        if local_rank == 0:
            print(f"Loaded {len(dataset)} training samples from {jsonl_path}")
    else:
        if local_rank == 0:
            print(f"Warning: {jsonl_path} not found, skipping...")

if len(train_datasets_info) == 0:
    if local_rank == 0:
        print("❌ No valid training JSONL files found!")
        print(f"Searched files: {jsonl_files}")
    exit(1)

# 处理训练数据集
if len(train_datasets_info) == 1:
    train_dataset = train_datasets_info[0][0]
else:
    min_size = min(len(d) for d, _ in train_datasets_info)
    if local_rank == 0:
        print(f"Balancing training datasets to {min_size} samples each")
    
    balanced_datasets = []
    for dataset, jsonl_path in train_datasets_info:
        if len(dataset) > min_size:
            selected_indices = torch.multinomial(
                torch.ones(len(dataset)), 
                min_size, 
                replacement=True
            ).tolist()
            subset = Subset(dataset, selected_indices)
            balanced_datasets.append(subset)
            if local_rank == 0:
                print(f"  {jsonl_path}: {len(dataset)} -> {len(subset)} samples")
        else:
            balanced_datasets.append(dataset)
            if local_rank == 0:
                print(f"  {jsonl_path}: {len(dataset)} samples (no sampling needed)")
    
    train_dataset = ConcatDataset(balanced_datasets)

if local_rank == 0:
    print(f"Total training samples: {len(train_dataset)}")

# 创建验证数据集
val_dataset = None
if val_jsonl_files:
    val_datasets_info = []
    for jsonl_path in val_jsonl_files:
        if os.path.exists(jsonl_path):
            dataset = JSONLAudioMotionDataset(jsonl_path, config)
            val_datasets_info.append((dataset, jsonl_path))
            if local_rank == 0:
                print(f"Loaded {len(dataset)} validation samples from {jsonl_path}")
    
    if val_datasets_info:
        if len(val_datasets_info) == 1:
            val_dataset = val_datasets_info[0][0]
        else:
            val_datasets = [d for d, _ in val_datasets_info]
            val_dataset = ConcatDataset(val_datasets)
        if local_rank == 0:
            print(f"Total validation samples: {len(val_dataset)}")

def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch]).long()
    masks = torch.stack([item['mask'] for item in batch]).long()
    lengths = torch.tensor([item['seq_length'] for item in batch])
    return {'tokens': tokens, 'mask': masks, 'lengths': lengths}

# 创建数据加载器
if world_size > 1:
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=local_rank,
        shuffle=False
    ) if val_dataset else None
else:
    train_sampler = None
    val_sampler = None

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size // world_size,
    sampler=train_sampler,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.batch_size // world_size,
    sampler=val_sampler,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4
) if val_dataset else None

# 优化器（添加weight decay）
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# 如果resume from checkpoint，也恢复optimizer状态
if args.resume_from and os.path.exists(args.resume_from):
    checkpoint = torch.load(args.resume_from, map_location='cpu')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

# 使用CosineAnnealingLR with warmup替代OneCycleLR
warmup_epochs = max(1, config.epochs // 10)  # 10%的epoch用于warmup
warmup_steps = warmup_epochs * len(train_dataloader)
total_steps = config.epochs * len(train_dataloader)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 或者使用ReduceLROnPlateau（基于验证loss）
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5, verbose=True
# )

accum_steps = 4 if config.max_seq_length > 1024 else 1

# 带label smoothing的loss函数
def compute_loss_with_label_smoothing(logits, labels, vocab_size, label_smoothing=0.0):
    """计算带label smoothing的交叉熵损失"""
    if label_smoothing == 0.0:
        return F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    
    # 获取有效标签（非-100的位置）
    valid_mask = labels.view(-1) != -100
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    valid_logits = logits.view(-1, vocab_size)[valid_mask]
    valid_labels = labels.view(-1)[valid_mask]
    
    # 计算标准交叉熵
    log_probs = F.log_softmax(valid_logits, dim=-1)
    nll_loss = F.nll_loss(log_probs, valid_labels, reduction='none')
    
    # 计算smoothing项
    smooth_loss = -log_probs.mean(dim=-1)
    
    # 组合
    loss = (1.0 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
    return loss.mean()

# 设置labels：只保留最后一个motion token的label，其他位置设为-100
def set_labels_for_last_motion_token(inputs, masks, pad_token_id):
    """
    设置labels，只计算最后一个motion token的loss
    
    策略：对于每个样本，只保留最后一个motion token的label用于计算loss，
    其他所有位置（包括前面的motion token）都设为-100（忽略）。
    这样可以训练模型只预测下一个motion token，而不是所有motion token。
    
    Args:
        inputs: [batch_size, seq_len] token序列，已经在device上
        masks: [batch_size, seq_len] mask，1表示motion token，0表示其他（audio/padding），已经在device上
        pad_token_id: padding token ID（未使用，保留以兼容接口）
    
    Returns:
        labels: [batch_size, seq_len] labels，只有最后一个motion token位置有label（token ID），其他都是-100
    """
    labels = inputs.clone().long()  # labels会在与inputs相同的device上
    batch_size, seq_len = labels.shape
    
    # 首先将所有非motion token位置设为-100（audio token和padding）
    labels[masks == 0] = -100
    
    # 对于每个样本，找到最后一个motion token的位置
    for i in range(batch_size):
        # 找到该样本中所有motion token的位置（mask == 1的位置）
        motion_positions = torch.where(masks[i] == 1)[0]
        
        if len(motion_positions) > 0:
            # 将除了最后一个motion token之外的所有motion token位置设为-100
            if len(motion_positions) > 1:
                # 将前面的motion token位置设为-100（只保留最后一个）
                labels[i, motion_positions[:-1]] = -100
            # 最后一个motion token位置保持原值（即token ID），用于计算loss
        # 如果没有motion token，所有位置都已经是-100了
    
    return labels

# 验证函数
def validate():
    """在验证集上评估模型"""
    if val_dataloader is None:
        return None
    
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['tokens'].to(device, non_blocking=True).long()
            masks = batch['mask'].to(device, non_blocking=True).long()
            
            attn_mask = (inputs != config.pad_token_id).float().to(device)
            # 只计算最后一个motion token的loss
            labels = set_labels_for_last_motion_token(inputs, masks, config.pad_token_id)
            
            outputs = model(inputs, labels=labels, attention_mask=attn_mask)
            
            # 如果使用label smoothing，需要手动计算loss
            if config.label_smoothing > 0.0:
                logits = outputs.logits
                loss = compute_loss_with_label_smoothing(
                    logits, labels, config.total_vocab_size, config.label_smoothing
                )
            else:
                loss = outputs.loss
            
            total_val_loss += loss.item()
            num_batches += 1
    
    # 在分布式训练中同步所有进程的loss
    if world_size > 1:
        # 收集所有进程的loss和batch数
        loss_tensor = torch.tensor(total_val_loss, device=device)
        num_batches_tensor = torch.tensor(num_batches, device=device)
        
        # 汇总所有进程的值
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        
        total_val_loss = loss_tensor.item()
        num_batches = num_batches_tensor.item()
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
    return avg_val_loss

# 从start_epoch开始训练
if local_rank == 0:
    print(f"\n🚀 Starting training from epoch {start_epoch} to {config.epochs}")
    print(f"   Total epochs to train: {config.epochs - start_epoch}")
    print(f"   Anti-overfitting measures:")
    print(f"     - Dropout: {config.dropout}")
    print(f"     - Weight decay: {config.weight_decay}")
    print(f"     - Label smoothing: {config.label_smoothing}")
    if config.no_validation:
        print(f"     - Validation: DISABLED")
    else:
        print(f"     - Early stopping patience: {args.early_stop_patience}")
        print(f"     - Validation frequency: every {config.val_freq} epochs")
        if val_dataset:
            print(f"     - Validation set: {len(val_dataset)} samples")
    print(f"{'='*80}\n")

for epoch in range(start_epoch, config.epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    if local_rank == 0:
        dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
    else:
        dataloader_iter = train_dataloader
        
    for step, batch in enumerate(dataloader_iter):
        inputs = batch['tokens'].to(device, non_blocking=True).long()
        masks = batch['mask'].to(device, non_blocking=True).long()
        lengths = batch['lengths']
        
        attn_mask = (inputs != config.pad_token_id).float().to(device)
        # 只计算最后一个motion token的loss
        labels = set_labels_for_last_motion_token(inputs, masks, config.pad_token_id)
        
        outputs = model(inputs, labels=labels, attention_mask=attn_mask)
        
        # 如果使用label smoothing，需要手动计算loss
        if config.label_smoothing > 0.0:
            logits = outputs.logits
            loss = compute_loss_with_label_smoothing(
                logits, labels, config.total_vocab_size, config.label_smoothing
            )
        else:
            loss = outputs.loss
        
        loss = loss / accum_steps
        loss.backward()
        
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        
        if local_rank == 0 and (step % 500 == 0 or step == len(train_dataloader) - 1):
            log_data = {
                "train/loss": loss.item() * accum_steps,
                "train/lr": scheduler.get_last_lr()[0],
                "train/seq_length": lengths.float().mean().item(),
                "train/step": epoch * len(train_dataloader) + step,
            }
            wandb.log(log_data)
            dataloader_iter.set_postfix(loss=loss.item() * accum_steps)
    
    if world_size > 1:
        dist.barrier()
    
    if local_rank == 0:
        avg_train_loss = total_loss / len(train_dataloader)
        
        # 只在指定频率时进行验证（每val_freq个epoch，或第一个epoch，或最后一个epoch）
        # 如果禁用了验证，完全跳过
        should_validate = False
        if not config.no_validation and config.val_freq > 0:
            should_validate = (
                (epoch + 1) % config.val_freq == 0 or 
                epoch == 0 or 
                epoch == config.epochs - 1
            )
        
        val_loss = None
        if should_validate and val_dataloader is not None:
            val_loss = validate()
        
        # 记录日志
        log_data = {
            "epoch/train_loss": avg_train_loss,
            "epoch": epoch,
            "epoch/lr": scheduler.get_last_lr()[0],
        }
        if val_loss is not None:
            log_data["epoch/val_loss"] = val_loss
        wandb.log(log_data)
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f}", end="")
        if config.no_validation:
            print(f" | Val Loss: Disabled", end="")
        elif val_loss is not None:
            print(f" | Val Loss: {val_loss:.4f}", end="")
        elif should_validate and val_dataloader is None:
            print(f" | Val Loss: N/A (no validation set)", end="")
        else:
            if config.val_freq > 0:
                next_val_epoch = ((epoch // config.val_freq) + 1) * config.val_freq
                if next_val_epoch > config.epochs:
                    next_val_epoch = config.epochs
                print(f" | Val Loss: Skipped (next at epoch {next_val_epoch})", end="")
            else:
                print(f" | Val Loss: Disabled", end="")
        print(f" | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 早停和模型保存
        should_save = False
        if config.no_validation:
            # 禁用验证时，定期保存（每50个epoch）
            should_save = (epoch + 1) % 50 == 0
        elif val_loss is not None:
            # 基于验证loss的早停
            if val_loss < best_val_loss - config.early_stop_min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                should_save = True
                print(f"✅ Validation loss improved! New best: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter}/{args.early_stop_patience} validation checks")
                
                if patience_counter >= args.early_stop_patience:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            # 有验证集但本次未验证，定期保存
            should_save = (epoch + 1) % 50 == 0
        
        # 保存checkpoint
        if should_save or not args.save_best_only:
            ckpt_path = f"output_disk0/motion_adaptor_{model_version}/{config.exp_name}/checkpoints/epoch_{epoch+1}.pt"
            best_ckpt_path = f"output_disk0/motion_adaptor_{model_version}/{config.exp_name}/checkpoints/best.pt"
            
            model_to_save = model.module if world_size > 1 else model
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }
            
            torch.save(checkpoint_data, ckpt_path)
            
            # 保存最佳模型
            if should_save and val_loss is not None:
                torch.save(checkpoint_data, best_ckpt_path)
                wandb.save(best_ckpt_path)
                print(f"💾 Saved best checkpoint: {best_ckpt_path}")
            else:
                wandb.save(ckpt_path)
                print(f"💾 Saved checkpoint: {ckpt_path}")

if local_rank == 0:
    # 保存最终模型
    model_to_save = model.module if world_size > 1 else model
    model_to_save.save_pretrained(f"output_disk0/motion_adaptor_{model_version}/{config.exp_name}")
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")

if world_size > 1:
    dist.destroy_process_group()

