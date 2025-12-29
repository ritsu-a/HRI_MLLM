"""
训练脚本：音频-动作未来预测任务
任务：根据25帧audio token + 25帧motion token历史，预测14帧future motion token
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config
import numpy as np
import os
# import wandb  # 已禁用wandb
import math
from tqdm import tqdm
from HRI_mllm.datasets.audio_motion_future_prediction_dataset import AudioMotionFuturePredictionDataset
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

import argparse

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser(description='Train Motion Future Prediction Model')
parser.add_argument('--jsonl_files', type=str, nargs='+', required=True,
                   help='Training JSONL file paths')
parser.add_argument('--val_jsonl_files', type=str, nargs='+', default=None,
                   help='Validation JSONL file paths')
parser.add_argument('--resume_from', type=str, default=None,
                   help='Checkpoint path to resume from')
parser.add_argument('--epochs', type=int, default=500,
                   help='Number of epochs to train')
parser.add_argument('--version', type=str, default='v1',
                   help='Model version identifier')
parser.add_argument('--batch_size', type=int, default=256,
                   help='Batch size per GPU')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                   help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01,
                   help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.1,
                   help='Dropout rate')
parser.add_argument('--val_freq', type=int, default=10,
                   help='Validation frequency (every N epochs)')
parser.add_argument('--save_freq', type=int, default=50,
                   help='Save checkpoint frequency (every N epochs)')
parser.add_argument('--padding_frames', type=int, default=3,
                   help='Number of initial padding frames')
parser.add_argument('--past_motion_frames', type=int, default=25,
                   help='Number of past motion frames')
parser.add_argument('--future_audio_frames', type=int, default=50,
                   help='Number of future audio frames')
parser.add_argument('--future_motion_frames', type=int, default=50,
                   help='Number of future motion frames to predict (input sequence uses padding, labels use real tokens)')
parser.add_argument('--max_samples', type=int, default=None,
                   help='Maximum number of samples to load for debugging (None = load all)')
parser.add_argument('--history_motion_mask_prob', type=float, default=0.3,
                   help='Probability of masking each history motion token during training (data augmentation)')
args = parser.parse_args()

# ==================== 配置 ====================
exp_name = "motion_future_prediction"
model_version = args.version
output_dir = f"output_disk0/motion_adaptor_{model_version}/{exp_name}"
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# os.environ["WANDB_MODE"] = "offline"  # 已禁用wandb

# ==================== 分布式训练设置 ====================
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

if 'RANK' in os.environ:
    local_rank, world_size = setup_distributed()
else:
    local_rank, world_size = 0, 1
    torch.cuda.set_device(0)

device = torch.device(f'cuda:{local_rank}')

# ==================== 模型配置 ====================
# Token配置
audio_vocab_size = 16384  # audio tokenizer vocab size
motion_vocab_size = 512 * 2  # motion tokenizer vocab size
total_vocab_size = motion_vocab_size + 10  # motion vocab + special tokens

# 特殊token ID
pad_token_id = motion_vocab_size + 1
gesture_start_token_id = motion_vocab_size + 2
audio_gesture_start_token_id = motion_vocab_size + 3
gesture_end_token_id = motion_vocab_size + 4
audio_gesture_end_token_id = motion_vocab_size + 5
audio_empty_token_id = 152063  # glm-voice-4的padding token
motion_empty_token_id = motion_vocab_size + 7

# 任务配置
padding_frames = args.padding_frames
past_motion_frames = args.past_motion_frames
future_audio_frames = args.future_audio_frames
future_motion_frames = args.future_motion_frames
# 输入序列长度：3*padding + 25*past_motion + 50*future_audio + 50*future_motion_padding = 128
# 输出序列长度：50*future_motion（作为监督，输入序列中对应位置是padding，labels中最后50个位置是真实token）
# 模型输入序列长度固定为128
input_seq_length = padding_frames + past_motion_frames + future_audio_frames + future_motion_frames  # 128
max_seq_length = input_seq_length  # 128（总序列长度，与输入序列长度一致）

# 数据增强配置
history_motion_mask_prob = args.history_motion_mask_prob  # 历史motion token的mask概率
history_motion_start_idx = padding_frames  # 历史motion在序列中的起始位置（3）
history_motion_end_idx = padding_frames + past_motion_frames  # 历史motion在序列中的结束位置（28）

# 创建配置对象
class Config:
    def __init__(self):
        self.pad_token_id = pad_token_id
        self.audio_empty_token_id = audio_empty_token_id
        self.motion_empty_token_id = motion_empty_token_id
        self.padding_frames = padding_frames
        self.past_motion_frames = past_motion_frames
        self.future_audio_frames = future_audio_frames
        self.future_motion_frames = future_motion_frames
        self.input_seq_length = input_seq_length  # 128，模型输入序列长度
        self.max_seq_length = max_seq_length  # 128，总序列长度（与输入序列长度一致）
        self.window_step = 1  # 滑动窗口步长

config = Config()

# ==================== WandB初始化 ====================
# 已禁用wandb
# if local_rank == 0:
#     wandb.init(
#         project="motion-future-prediction",
#         config={
#             "jsonl_files": args.jsonl_files,
#             "val_jsonl_files": args.val_jsonl_files,
#             "history_audio_frames": history_audio_frames,
#             "history_motion_frames": history_motion_frames,
#             "future_motion_frames": future_motion_frames,
#             "max_seq_length": max_seq_length,
#             "batch_size": args.batch_size,
#             "learning_rate": args.learning_rate,
#             "epochs": args.epochs,
#             "weight_decay": args.weight_decay,
#             "dropout": args.dropout,
#             "model_version": model_version,
#         }
#     )

# ==================== 创建模型 ====================
model_config = GPT2Config(
    vocab_size=total_vocab_size,
    n_positions=input_seq_length,  # 128，模型输入序列长度固定为128
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_inner=3072,
    resid_pdrop=args.dropout,
    embd_pdrop=args.dropout,
    attn_pdrop=args.dropout,
)

# 添加特殊token ID到模型配置
model_config.audio_gesture_start_token_id = audio_gesture_start_token_id
model_config.audio_gesture_end_token_id = audio_gesture_end_token_id
model_config.gesture_start_token_id = gesture_start_token_id
model_config.gesture_end_token_id = gesture_end_token_id

model = MixedInputGPT2(model_config, audio_hidden_size=3584)
model.to(device)

# ==================== 加载checkpoint ====================
start_epoch = 0
best_val_loss = float('inf')

if args.resume_from and os.path.exists(args.resume_from):
    if local_rank == 0:
        print(f"\n{'='*80}")
        print(f"🔄 Resuming from checkpoint: {args.resume_from}")
        print(f"{'='*80}")
    
    checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    if local_rank == 0:
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Resuming from epoch {start_epoch}")
        print(f"{'='*80}\n")

# ==================== DDP包装 ====================
if world_size > 1:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )

# 已禁用wandb
# if local_rank == 0:
#     model_to_watch = model.module if world_size > 1 else model
#     wandb.watch(model_to_watch, log="parameters", log_freq=100)

# ==================== 创建数据集 ====================
train_datasets = []
for jsonl_path in args.jsonl_files:
    if os.path.exists(jsonl_path):
        dataset = AudioMotionFuturePredictionDataset(jsonl_path, config, max_samples=args.max_samples)
        train_datasets.append(dataset)
        if local_rank == 0:
            print(f"Loaded {len(dataset)} training samples from {jsonl_path}")
    else:
        if local_rank == 0:
            print(f"⚠️  Training JSONL file not found: {jsonl_path}")

if len(train_datasets) == 0:
    if local_rank == 0:
        print("❌ No valid training JSONL files found!")
    exit(1)

if len(train_datasets) > 1:
    train_dataset = ConcatDataset(train_datasets)
else:
    train_dataset = train_datasets[0]

if local_rank == 0:
    print(f"Total training samples: {len(train_dataset)}")

# 验证数据集
val_dataset = None
if args.val_jsonl_files:
    val_datasets = []
    for jsonl_path in args.val_jsonl_files:
        if os.path.exists(jsonl_path):
            dataset = AudioMotionFuturePredictionDataset(jsonl_path, config, max_samples=args.max_samples)
            val_datasets.append(dataset)
            if local_rank == 0:
                print(f"Loaded {len(dataset)} validation samples from {jsonl_path}")
    
    if val_datasets:
        if len(val_datasets) > 1:
            val_dataset = ConcatDataset(val_datasets)
        else:
            val_dataset = val_datasets[0]
        if local_rank == 0:
            print(f"Total validation samples: {len(val_dataset)}")

# ==================== Collate函数 ====================
def collate_fn(batch):
    """简单的collate函数，因为所有序列长度都是固定的（128）"""
    tokens = torch.stack([item['tokens'] for item in batch]).long()
    labels = torch.stack([item['labels'] for item in batch]).long()
    attention_mask = torch.stack([item['attention_mask'] for item in batch]).long()
    return {
        'tokens': tokens,
        'labels': labels,
        'attention_mask': attention_mask,
    }

# ==================== 数据加载器 ====================
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
    batch_size=args.batch_size // world_size,
    sampler=train_sampler,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size // world_size,
    sampler=val_sampler,
    collate_fn=collate_fn,
    pin_memory=True,
    num_workers=4
) if val_dataset else None

# ==================== 优化器和学习率调度器 ====================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)

# CosineAnnealingLR with warmup
warmup_epochs = max(1, args.epochs // 10)
warmup_steps = warmup_epochs * len(train_dataloader)
total_steps = args.epochs * len(train_dataloader)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==================== 验证函数 ====================
def validate():
    """在验证集上评估模型"""
    if val_dataloader is None:
        return None
    
    # 所有进程都需要参与验证，避免死锁
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['tokens'].to(device, non_blocking=True).long()
            labels = batch['labels'].to(device, non_blocking=True).long()
            attention_mask = batch['attention_mask'].to(device, non_blocking=True).float()
            
            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            
            total_val_loss += loss.item()
            num_batches += 1
    
    # 分布式训练中同步loss
    if world_size > 1:
        loss_tensor = torch.tensor(total_val_loss, device=device)
        num_batches_tensor = torch.tensor(num_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        total_val_loss = loss_tensor.item()
        num_batches = num_batches_tensor.item()
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
    return avg_val_loss

# ==================== 训练循环 ====================
if local_rank == 0:
    print(f"\n🚀 Starting training from epoch {start_epoch} to {args.epochs}")
    print(f"   Task: Predict {future_motion_frames} future motion tokens")
    print(f"   Input: {padding_frames}*padding + {past_motion_frames}*past_motion + {future_audio_frames}*future_audio + {future_motion_frames}*future_motion_padding = {input_seq_length}")
    print(f"   Output: {future_motion_frames}*future_motion (supervision)")
    print(f"   Model input sequence length: {input_seq_length}")
    print(f"   Total sequence length (dataset): {max_seq_length}")
    print(f"   Data augmentation: History motion mask probability = {history_motion_mask_prob}")
    print(f"{'='*80}\n")

for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0
    
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    if local_rank == 0:
        dataloader_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    else:
        dataloader_iter = train_dataloader
    
    for step, batch in enumerate(dataloader_iter):
        inputs = batch['tokens'].to(device, non_blocking=True).long()
        labels = batch['labels'].to(device, non_blocking=True).long()
        attention_mask = batch['attention_mask'].to(device, non_blocking=True).float()
        
        # 数据增强：随机mask历史motion token
        if history_motion_mask_prob > 0:
            # 为整个batch的history motion部分生成随机mask（向量化操作，更高效）
            batch_size = inputs.shape[0]
            history_motion_length = history_motion_end_idx - history_motion_start_idx
            # 生成随机mask矩阵 [batch_size, history_motion_length]
            mask = torch.rand(batch_size, history_motion_length, device=device) < history_motion_mask_prob
            # 将被mask的位置替换为padding token
            inputs[:, history_motion_start_idx:history_motion_end_idx][mask] = pad_token_id
        
        outputs = model(inputs, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
   
        if isinstance(dataloader_iter, tqdm):
            dataloader_iter.set_postfix(loss=loss.item())
    
    if world_size > 1:
        dist.barrier()
    
    # 验证已禁用，避免卡死问题
    val_loss = None
    
    if local_rank == 0:
        avg_train_loss = total_loss / len(train_dataloader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 已禁用wandb
        # log_data = {
        #     "epoch/train_loss": avg_train_loss,
        #     "epoch": epoch,
        #     "epoch/lr": scheduler.get_last_lr()[0],
        # }
        # wandb.log(log_data)
        
        # 保存checkpoint（仅按频率保存，不基于验证loss）
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            
            model_to_save = model.module if world_size > 1 else model
            checkpoint_data = {
                'epoch': epoch,
                'model_state': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }
            
            try:
                torch.save(checkpoint_data, ckpt_path)
                print(f"💾 Saved checkpoint: {ckpt_path}")
                # wandb.save可能卡死，暂时注释
                # wandb.save(ckpt_path)
            except Exception as e:
                print(f"⚠️  Save checkpoint failed: {e}")
    
    # 同步所有进程，确保所有操作完成（必须在if块外部）
    if world_size > 1:
        dist.barrier()

if local_rank == 0:
    # 保存最终模型
    model_to_save = model.module if world_size > 1 else model
    model_to_save.save_pretrained(output_dir)
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")

if world_size > 1:
    dist.destroy_process_group()

