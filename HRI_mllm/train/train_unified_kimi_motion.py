#!/usr/bin/env python3
"""
统一Kimi-Motion模型训练脚本
训练流程：
1. Kimi模型：user_text + user_audio → assistant_audio (监督学习)
2. Adaptor：Kimi hidden states → motion_tokens (监督学习)
只训练MLP映射层和adaptor

使用预处理的hidden states训练（推荐）：
当提供 --preprocessed_hidden_states_dir 参数时：
- 自动跳过Kimi模型加载（节省GPU显存）
- 强制只训练Hidden State Mixer
- Adaptor将被冻结（不训练）
- 必须确保所有样本都有对应的预处理hidden states

示例命令：
python train_unified_kimi_motion.py \
    --train_json_paths data/train.jsonl \
    --preprocessed_hidden_states_dir output/preprocessed_hidden_states \
    --batch_size 4 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --save_dir output/mixer_model
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

# 设置MuJoCo渲染
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT
from HRI_mllm.model.unified_kimi_motion_model import UnifiedKimiMotionModel, create_unified_model
from HRI_mllm.datasets.json_audio_motion_dataset import create_dataloader, JSONAudioMotionDataset
from transformers import GPT2Config


class UnifiedModelTrainer:
    """
    统一模型训练器
    
    训练流程：
    1. Kimi模型：user_text + user_audio → assistant_audio (监督学习)
    2. Adaptor：Kimi hidden states → motion_tokens (监督学习)
    """
    
    def __init__(self, 
                 model: UnifiedKimiMotionModel,
                 train_dataloader: DataLoader,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 100,
                 max_grad_norm: float = 1.0,
                 gradient_accumulation_steps: int = 1,
                 device: str = "cuda",
                 debug: bool = False):
        """
        Args:
            model: 统一模型
            train_dataloader: 训练数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            max_grad_norm: 最大梯度范数
            gradient_accumulation_steps: 梯度累积步数（用于减少显存使用）
            device: 设备
            debug: 是否开启详细debug日志
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.debug = debug
        
        # 移动到设备
        self.model.to(device)
        
        # 获取实际模型（处理DDP包装）
        def get_actual_model(m):
            return m.module if isinstance(m, DDP) else m
        self.actual_model = get_actual_model(self.model)
        
        # 设置优化器（只训练可训练参数）
        trainable_params = self.actual_model.get_trainable_parameters()
        print(f"📊 Trainable parameters: {len(trainable_params)}")
        for name, param in trainable_params:
            print(f"   - {name}: {param.shape}")
        
        self.optimizer = optim.AdamW(
            [param for _, param in trainable_params],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 训练统计
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 只在主进程或单卡模式打印
        if not isinstance(model, DDP) or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
            print(f"✅ Trainer initialized")
            print(f"   - Device: {device}")
            print(f"   - Learning rate: {learning_rate}")
            print(f"   - Weight decay: {weight_decay}")
            print(f"   - Warmup steps: {warmup_steps}")
            print(f"   - Max grad norm: {max_grad_norm}")
            print(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        # 只在主进程或非分布式模式显示进度条
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        else:
            progress_bar = self.train_dataloader
        
        # 梯度累积：只在第一个batch时清零梯度
        accumulation_count = 0
        
        # 打印开始信息（只在主进程）
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"\n🔄 Starting epoch {self.epoch}...")
            print(f"   Dataloader has {len(self.train_dataloader)} batches")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 验证batch数据
            if batch is None:
                if self.debug:
                    print(f"⚠️  Skipping batch {batch_idx}: batch is None")
                continue
            
            # Debug模式：只处理第一个batch
            if self.debug and batch_idx > 0:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"🔍 Debug mode: Stopping after first batch")
                break
            
            # 详细的batch验证（仅在debug模式且主进程打印）
            if self.debug and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                print(f"🔍 Debug batch {batch_idx}:")
                print(f"   - Batch type: {type(batch)}")
                print(f"   - Batch keys: {list(batch.keys()) if batch else 'None'}")
                
                # 检查每个关键字段
                for key in ['interleaved_sequences', 'token_labels', 'attention_masks', 'user_audio_tokens', 'assistant_audio_tokens', 'motion_tokens']:
                    if key in batch:
                        value = batch[key]
                        print(f"   - {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}, device={value.device if hasattr(value, 'device') else 'N/A'}")
                        if value is None:
                            print(f"   - ❌ {key} is None!")
                        elif hasattr(value, 'min') and hasattr(value, 'max'):
                            print(f"     min={value.min().item()}, max={value.max().item()}")
                            # 检查是否超出Kimi模型的vocab_size (168448)
                            if value.max().item() >= 168448:
                                print(f"     ⚠️  WARNING: Token ID {value.max().item()} >= Kimi vocab_size (168448)")
                    else:
                        print(f"   - ❌ Missing key: {key}")
            
            # 准备数据
            try:
                # 逐个检查和转换tensor
                if batch['interleaved_sequences'] is None:
                    raise ValueError("interleaved_sequences is None")
                interleaved_sequences = batch['interleaved_sequences'].to(self.device)
                
                if batch['token_labels'] is None:
                    raise ValueError("token_labels is None")
                token_labels = batch['token_labels'].to(self.device)
                
                if batch['attention_masks'] is None:
                    raise ValueError("attention_masks is None")
                attention_masks = batch['attention_masks'].to(self.device)
                
            except Exception as e:
                # 只在主进程或非分布式模式打印错误
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"❌ Error preparing batch {batch_idx}: {e}")
                    if self.debug:
                        print(f"   Batch keys: {list(batch.keys()) if batch else 'None'}")
                        import traceback
                        traceback.print_exc()
                continue
            
            batch_size, seq_len = interleaved_sequences.shape
            
            # 梯度累积：只在累积的第一个batch清零梯度
            if accumulation_count == 0:
                self.optimizer.zero_grad()
            
            try:
                # 使用teacher forcing模式
                outputs = self.model(
                    user_text=batch.get('user_text'),
                    user_audio_tokens=batch.get('user_audio_tokens'),
                    assistant_audio_tokens=batch.get('assistant_audio_tokens'),
                    motion_tokens=batch.get('motion_tokens'),
                    interleaved_sequences=interleaved_sequences,
                    attention_mask=attention_masks,
                    labels=token_labels,
                    batch=batch  # 传递整个batch以访问预处理的hidden states
                )
                
                # 计算损失（除以累积步数，使得梯度平均）
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss / self.gradient_accumulation_steps
                else:
                    # 手动计算损失
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = token_labels[..., 1:].contiguous()
                    loss = self.criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    ) / self.gradient_accumulation_steps
                
                # 提取motion loss和audio loss（原始值，未应用权重）
                motion_loss_value = None
                audio_loss_value = None
                if hasattr(outputs, 'motion_loss'):
                    motion_loss_value = outputs.motion_loss.item() if outputs.motion_loss is not None else None
                if hasattr(outputs, 'audio_loss'):
                    audio_loss_value = outputs.audio_loss.item() if outputs.audio_loss is not None else None
                
                # 反向传播（累积梯度）
                loss.backward()
                
                # 统计（使用原始loss，不除以累积步数）
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    batch_loss = outputs.loss.item()
                else:
                    batch_loss = loss.item() * self.gradient_accumulation_steps
                
                batch_tokens = (token_labels != -100).sum().item()
                
                accumulation_count += 1
                
                # 只有当累积步数达到要求时才更新参数
                if accumulation_count >= self.gradient_accumulation_steps:
                    # 梯度裁剪
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [param for _, param in self.actual_model.get_trainable_parameters()],
                            self.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    # 重置累积计数
                    accumulation_count = 0
                    
                    # 更新step计数
                    self.step += 1
                
                # 更新统计
                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                num_batches += 1
                
                # 更新进度条（只在累积完成时或最后一个batch显示）
                should_update_progress = (accumulation_count == 0) or (batch_idx == len(self.train_dataloader) - 1)
                if should_update_progress:
                    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # 显示累积状态和loss信息
                    acc_display = f'{self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}' if accumulation_count == 0 else f'{accumulation_count}/{self.gradient_accumulation_steps}'
                    
                    # 构建loss显示字符串
                    loss_info = f'total:{batch_loss:.4f}'
                    if motion_loss_value is not None:
                        loss_info += f' | motion:{motion_loss_value:.4f}'
                    if audio_loss_value is not None:
                        loss_info += f' | audio:{audio_loss_value:.4f}'
                    
                    progress_bar.set_postfix({
                        'loss': loss_info,
                        'avg': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'acc': acc_display
                    })
                    
                    # 记录到wandb（只在累积完成时记录）
                    if accumulation_count == 0 and wandb.run is not None:
                        log_dict = {
                            'train/total_loss': batch_loss,
                            'train/avg_loss': avg_loss,
                            'train/learning_rate': current_lr,
                            'train/batch_tokens': batch_tokens,
                            'train/step': self.step
                        }
                        if motion_loss_value is not None:
                            log_dict['train/motion_loss'] = motion_loss_value
                        if audio_loss_value is not None:
                            log_dict['train/audio_loss'] = audio_loss_value
                        wandb.log(log_dict)
                
            except Exception as e:
                # 只在主进程或非分布式模式打印错误
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                
                # Debug模式：遇到错误立即退出
                if self.debug:
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        print(f"🔍 Debug mode: Exiting due to error in batch {batch_idx}")
                    break
                    
                continue
        
        # 处理最后一个未完成的累积batch（如果有）
        if accumulation_count > 0:
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for _, param in self.actual_model.get_trainable_parameters()],
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
        
        # 计算epoch统计
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        
        return {
            'train_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
        # 保存实际模型的状态（DDP模式下自动去除wrapper）
        model_state = self.actual_model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved to: {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """加载检查点"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载到实际模型
        self.actual_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        
        print(f"✅ Checkpoint loaded from: {load_path}")
        print(f"   - Epoch: {self.epoch}")
        print(f"   - Step: {self.step}")
    
    def train(self, 
              num_epochs: int,
              save_every: int = 100,
              save_every_epochs: int = 2,
              save_dir: str = "output/unified_model"):
        """训练模型
        
        Args:
            num_epochs: 总训练轮数
            save_every: 每N个step保存（已弃用，保留用于兼容）
            save_every_epochs: 每N个epoch保存一次
            save_dir: 保存目录
        """
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 只在主进程打印
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"🚀 Starting training for {num_epochs} epochs")
            print(f"   - Save every: {save_every_epochs} epochs")
            print(f"   - Save directory: {save_dir}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            train_stats = self.train_epoch()
            
            # 记录epoch统计（只在主进程）
            if wandb.run is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_stats['train_loss'],
                    'train/epoch_tokens': train_stats['total_tokens'],
                    'train/epoch_batches': train_stats['num_batches']
                })
            
            # 每N个epoch保存一次（只在主进程保存）
            if (epoch + 1) % save_every_epochs == 0 or epoch == num_epochs - 1:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    self.save_checkpoint(
                        os.path.join(save_dir, f"epoch_{epoch + 1}.pt")
                    )
        
        # 训练完成（只在主进程打印）
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            total_time = time.time() - start_time
            print(f"\n🎉 Training completed!")
            print(f"   - Total time: {total_time:.2f}s")
            
            # 保存最终模型
            self.save_checkpoint(
                os.path.join(save_dir, "final_model.pt")
            )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Unified Kimi-Motion Model')
    
    # 模型参数
    parser.add_argument('--kimi_model_path', type=str, 
                       default="moonshotai/Kimi-Audio-7B",
                       help='Kimi模型路径')
    parser.add_argument('--gpt2_config_path', type=str, default=None,
                       help='GPT2配置文件路径')
    parser.add_argument('--freeze_kimi', action='store_true', default=True,
                       help='是否冻结Kimi模型')
    parser.add_argument('--freeze_adaptor', action='store_true', default=True,
                       help='是否冻结adaptor')
    parser.add_argument('--train_mixer_only', action='store_true', default=True,
                       help='是否只训练mixer')
    
    # 数据参数
    parser.add_argument('--train_json_paths', type=str, nargs='+', required=True,
                       help='训练JSON文件路径列表')
    parser.add_argument('--dataset_weights', type=float, nargs='+', default=None,
                       help='数据集权重列表')
    parser.add_argument('--max_audio_length', type=int, default=2048,
                       help='最大音频token长度')
    parser.add_argument('--max_motion_length', type=int, default=1024,
                       help='最大motion token长度')
    parser.add_argument('--interleave_ratio', type=int, nargs=2, default=[1, 1],
                       help='交错比例 [audio_tokens_per_motion, motion_tokens_per_audio]')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='预热步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='最大梯度范数')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数（用于减少显存使用，有效batch_size = batch_size * gradient_accumulation_steps * num_gpus）')
    parser.add_argument('--motion_loss_weight', type=float, default=1.0,
                       help='Motion token loss权重（默认1.0，更高）')
    parser.add_argument('--audio_loss_weight', type=float, default=0.1,
                       help='Audio token loss权重（默认0.1，较低）')
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default="output/unified_model",
                       help='模型保存目录')
    parser.add_argument('--save_every', type=int, default=100,
                       help='保存间隔步数（已弃用，保留用于兼容）')
    parser.add_argument('--save_every_epochs', type=int, default=50,
                       help='每N个epoch保存一次（默认50）')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练')
    
    # 其他参数
    parser.add_argument('--device', type=str, default="cuda",
                       help='设备')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    parser.add_argument('--use_wandb', action='store_true',
                       help='使用wandb记录')
    parser.add_argument('--wandb_project', type=str, default="unified-kimi-motion",
                       help='wandb项目名称')
    parser.add_argument('--preprocessed_hidden_states_dir', type=str, default=None,
                       help='预处理hidden states目录路径（如果提供，将使用预处理的hidden states进行训练，跳过Kimi模型加载，只训练mixer）')
    parser.add_argument('--adaptor_checkpoint_path', type=str, default=None,
                       help='预训练adaptor的checkpoint路径（如果提供，将加载预训练权重）')
    parser.add_argument('--skip_kimi_model', action='store_true',
                       help='跳过Kimi模型加载（当使用预处理的hidden states时自动启用）')
    
    return parser.parse_args()


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 分布式模式
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        # 单机模式
        return 0, 0, 1


def main():
    """主函数"""
    # 初始化分布式训练（如果需要）
    rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)
    
    args = parse_args()
    
    if is_main_process:
        print("🚀 Starting Unified Kimi-Motion Model Training")
        print(f"📋 Arguments: {vars(args)}")
        if world_size > 1:
            print(f"   - Distributed training: {world_size} GPUs")
    
    # 设置设备
    if world_size > 1:
        device = f"cuda:{local_rank}"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            device = "cpu"
    
    # 初始化wandb（只在主进程）
    if args.use_wandb and is_main_process:
        try:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"unified_model_{int(time.time())}",
                mode="offline"  # 设置为offline模式，避免login卡住
            )
            print("✅ Wandb initialized (offline mode)")
        except Exception as e:
            print(f"⚠️  Failed to initialize wandb: {e}")
            print("   Continuing without wandb logging...")
            args.use_wandb = False  # 禁用wandb，避免后续错误
    
    # 创建模型
    if is_main_process:
        print("🔄 Creating unified model...")
    
    # 如果使用了预处理的hidden states，强制跳过Kimi模型加载以节省显存
    # 并强制只训练mixer
    if args.preprocessed_hidden_states_dir is None:
        if is_main_process:
            print("⚠️  Warning: --preprocessed_hidden_states_dir not provided")
            print("   Will load Kimi model (requires more GPU memory)")
    else:
        if not os.path.exists(args.preprocessed_hidden_states_dir):
            if is_main_process:
                print(f"❌ Error: Preprocessed hidden states directory not found: {args.preprocessed_hidden_states_dir}")
                print("   Please run prepare_mixer_training_data.py first to generate hidden states")
            raise FileNotFoundError(f"Preprocessed hidden states directory not found: {args.preprocessed_hidden_states_dir}")
    
    skip_kimi_model = args.preprocessed_hidden_states_dir is not None and os.path.exists(args.preprocessed_hidden_states_dir)
    if skip_kimi_model:
        if is_main_process:
            print("💡 Using preprocessed hidden states - will skip Kimi model and adaptor loading to save GPU memory")
            print("   - Setting skip_kimi_model=True")
            print("   - Setting skip_adaptor=True (using lightweight loss head)")
            print("   - Setting train_mixer_only=True (only training mixer)")
        
        # 强制设置参数以确保只训练mixer，并跳过adaptor加载
        args.skip_kimi_model = True
        args.train_mixer_only = True
        args.freeze_adaptor = True
        args.skip_adaptor = True  # 跳过adaptor加载，使用轻量级loss head
        
        # 检查预处理目录是否有效
        index_path = os.path.join(args.preprocessed_hidden_states_dir, "index.json")
        if not os.path.exists(index_path):
            if is_main_process:
                print(f"⚠️  Warning: index.json not found in {args.preprocessed_hidden_states_dir}")
                print(f"   Training will continue, but hidden states must be provided in batch")
        else:
            if is_main_process:
                import json
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    num_samples = len(index_data.get('samples', []))
                    print(f"✅ Found preprocessed hidden states: {num_samples} samples")
                    if 'text_hidden_size' in index_data:
                        print(f"   - Text hidden size: {index_data['text_hidden_size']}")
                    if 'audio_hidden_size' in index_data:
                        print(f"   - Audio hidden size: {index_data['audio_hidden_size']}")
    elif args.preprocessed_hidden_states_dir is not None:
        if is_main_process:
            print(f"⚠️  Warning: Preprocessed hidden states directory not found: {args.preprocessed_hidden_states_dir}")
            print(f"   Will fall back to loading Kimi model")
    
    model = create_unified_model(
        kimi_model_path=args.kimi_model_path,
        gpt2_config_path=args.gpt2_config_path,
        freeze_kimi=args.freeze_kimi,
        freeze_adaptor=args.freeze_adaptor,
        train_mixer_only=args.train_mixer_only,
        motion_loss_weight=args.motion_loss_weight,
        audio_loss_weight=args.audio_loss_weight,
        debug=args.debug,
        skip_kimi_model=skip_kimi_model,
        skip_adaptor=args.skip_adaptor if hasattr(args, 'skip_adaptor') else skip_kimi_model,  # 如果skip_kimi_model，也skip_adaptor
        adaptor_checkpoint_path=args.adaptor_checkpoint_path,
        preprocessed_hidden_states_dir=args.preprocessed_hidden_states_dir
    )
    
    if args.adaptor_checkpoint_path and is_main_process:
        print(f"✅ Using pre-trained adaptor from: {args.adaptor_checkpoint_path}")
    model.to(device)  # 确保模型在正确的设备上
    
    # 创建数据加载器
    if is_main_process:
        print("🔄 Creating data loaders...")
    
    # 在分布式训练中，batch_size是每个GPU的batch size
    train_dataloader = create_dataloader(
        json_paths=args.train_json_paths,
        batch_size=args.batch_size,
        shuffle=(world_size == 1),  # 分布式时由DistributedSampler处理shuffle
        num_workers=args.num_workers,
        dataset_weights=args.dataset_weights,
        debug=args.debug,
        max_audio_length=args.max_audio_length,
        max_motion_length=args.max_motion_length,
        interleave_ratio=tuple(args.interleave_ratio),
        preprocessed_hidden_states_dir=args.preprocessed_hidden_states_dir
    )
    
    if args.preprocessed_hidden_states_dir and is_main_process:
        print(f"✅ Using preprocessed hidden states from: {args.preprocessed_hidden_states_dir}")
        print(f"   This will significantly speed up training by skipping Kimi model forward pass")
    
    # 测试数据加载器（加载第一个batch以验证是否工作正常）
    # 注意：如果num_workers > 0，在多进程环境下测试可能会卡住
    # 所以我们跳过这个测试，直接进入训练
    if is_main_process:
        print(f"✅ Data loader created successfully")
        print(f"   - Dataset length: {len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else 'N/A'}")
        print(f"   - Number of batches: {len(train_dataloader)}")
        print(f"   - Num workers: {args.num_workers}")
        if args.num_workers > 0:
            print(f"   ⚠️  Note: With num_workers > 0, first batch loading may take a moment")
        
        # 验证预处理hidden states的使用
        if skip_kimi_model:
            skip_adaptor = args.skip_adaptor if hasattr(args, 'skip_adaptor') else skip_kimi_model
            print(f"\n📋 Training Configuration (with preprocessed hidden states):")
            print(f"   ✅ Kimi model: Skipped (using preprocessed hidden states)")
            if skip_adaptor:
                print(f"   ✅ Adaptor: Skipped (using lightweight loss head)")
            else:
                print(f"   ✅ Adaptor: Frozen (not training)")
            print(f"   ✅ Only training: Hidden State Mixer")
            print(f"   ✅ Preprocessed hidden states: Required")
            print(f"\n   If any sample is missing preprocessed hidden states, training will fail.")
            print(f"   Please ensure all samples in the dataset have been preprocessed.")
    
    # 如果是分布式训练，需要包装DataLoader
    if world_size > 1:
        # 注意：create_dataloader返回的dataloader需要重新创建以使用DistributedSampler
        # 这里简化处理：假设create_dataloader内部会处理分布式情况
        # 如果不行，需要修改create_dataloader以支持DistributedSampler
        if is_main_process:
            print("🔄 Distributed training detected, waiting for all processes to sync...")
        # 同步所有进程
        if world_size > 1:
            dist.barrier()
    
    # 如果是分布式训练，包装模型
    if world_size > 1:
        if is_main_process:
            print("🔄 Wrapping model with DDP...")
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # 因为只训练部分参数
        )
        if is_main_process:
            print("✅ Model wrapped with DDP")
    
    # 创建训练器（传入模型，trainer内部会处理DDP包装）
    if is_main_process:
        print("🔄 Creating trainer...")
    trainer = UnifiedModelTrainer(
        model=model,  # 可能是DDP包装的，也可能不是，trainer内部会处理
        train_dataloader=train_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=device,
        debug=args.debug
    )
    
    # 恢复训练
    if args.resume_from and is_main_process:
        print(f"🔄 Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # 开始训练前的最后检查
    if is_main_process:
        print("=" * 80)
        print("🚀 Starting training...")
        print("=" * 80)
        print(f"   - Total epochs: {args.num_epochs}")
        print(f"   - Dataloader length: {len(train_dataloader)} batches per epoch")
        print(f"   - Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
        if skip_kimi_model:
            skip_adaptor = args.skip_adaptor if hasattr(args, 'skip_adaptor') else skip_kimi_model
            print(f"   - Training mode: Hidden State Mixer only (using preprocessed hidden states)")
            print(f"   - Kimi model: Not loaded (saving GPU memory)")
            if skip_adaptor:
                print(f"   - Adaptor: Not loaded (using lightweight loss head, saving GPU memory)")
            else:
                print(f"   - Adaptor: Frozen")
        else:
            print(f"   - Training mode: Full model")
            print(f"   - Kimi model: Loaded ({'frozen' if args.freeze_kimi else 'trainable'})")
            print(f"   - Adaptor: {'frozen' if args.freeze_adaptor else 'trainable'}")
            print(f"   - Mixer: {'trainable' if args.train_mixer_only else ('trainable' if not args.freeze_adaptor else 'frozen')}")
        print("=" * 80)
    
    # 开始训练
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        save_every_epochs=args.save_every_epochs,
        save_dir=args.save_dir
    )
    
    # 关闭wandb（只在主进程）
    if args.use_wandb and is_main_process:
        wandb.finish()
    
    # 清理分布式训练
    if world_size > 1:
        dist.destroy_process_group()
    
    print("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
