#!/usr/bin/env python3
"""
统一Kimi-Motion模型训练脚本
训练流程：
1. Kimi模型：user_text + user_audio → assistant_audio (监督学习)
2. Adaptor：Kimi hidden states → motion_tokens (监督学习)
只训练MLP映射层和adaptor
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
            device: 设备
            debug: 是否开启详细debug日志
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.debug = debug
        
        # 移动到设备
        self.model.to(device)
        
        # 设置优化器（只训练可训练参数）
        trainable_params = self.model.get_trainable_parameters()
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
        
        print(f"✅ Trainer initialized")
        print(f"   - Device: {device}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Weight decay: {weight_decay}")
        print(f"   - Warmup steps: {warmup_steps}")
        print(f"   - Max grad norm: {max_grad_norm}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 验证batch数据
            if batch is None:
                print(f"⚠️  Skipping batch {batch_idx}: batch is None")
                continue
            
            # 详细的batch验证
            if self.debug:
                print(f"🔍 Debug batch {batch_idx}:")
                print(f"   - Batch type: {type(batch)}")
                print(f"   - Batch keys: {list(batch.keys()) if batch else 'None'}")
                
                # 检查每个关键字段
                for key in ['interleaved_sequences', 'token_labels', 'attention_masks']:
                    if key in batch:
                        value = batch[key]
                        print(f"   - {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}, device={value.device if hasattr(value, 'device') else 'N/A'}")
                        if value is None:
                            print(f"   - ❌ {key} is None!")
                    else:
                        print(f"   - ❌ Missing key: {key}")
            
            # 准备数据
            try:
                # 逐个检查和转换tensor
                if batch['interleaved_sequences'] is None:
                    raise ValueError("interleaved_sequences is None")
                interleaved_sequences = batch['interleaved_sequences'].to(self.device)
                if self.debug:
                    print(f"   ✅ interleaved_sequences converted to device: {interleaved_sequences.device}")
                
                if batch['token_labels'] is None:
                    raise ValueError("token_labels is None")
                token_labels = batch['token_labels'].to(self.device)
                if self.debug:
                    print(f"   ✅ token_labels converted to device: {token_labels.device}")
                
                if batch['attention_masks'] is None:
                    raise ValueError("attention_masks is None")
                attention_masks = batch['attention_masks'].to(self.device)
                if self.debug:
                    print(f"   ✅ attention_masks converted to device: {attention_masks.device}")
                
            except Exception as e:
                print(f"❌ Error preparing batch {batch_idx}: {e}")
                print(f"   Batch keys: {list(batch.keys()) if batch else 'None'}")
                import traceback
                traceback.print_exc()
                continue
            
            batch_size, seq_len = interleaved_sequences.shape
            
            # 前向传播
            self.optimizer.zero_grad()
            
            try:
                # 使用teacher forcing模式
                if self.debug:
                    print(f"🔍 Debug model forward pass for batch {batch_idx}:")
                    print(f"   - interleaved_sequences: shape={interleaved_sequences.shape}, dtype={interleaved_sequences.dtype}, device={interleaved_sequences.device}")
                    print(f"   - attention_masks: shape={attention_masks.shape}, dtype={attention_masks.dtype}, device={attention_masks.device}")
                    print(f"   - token_labels: shape={token_labels.shape}, dtype={token_labels.dtype}, device={token_labels.device}")
                    
                    # 检查tensor中是否有异常值
                    if torch.isnan(interleaved_sequences).any():
                        print(f"   ❌ Found NaN in interleaved_sequences!")
                    if torch.isinf(interleaved_sequences).any():
                        print(f"   ❌ Found Inf in interleaved_sequences!")
                
                outputs = self.model(
                    user_text=batch.get('user_text'),
                    user_audio_tokens=batch.get('user_audio_tokens'),
                    assistant_audio_tokens=batch.get('assistant_audio_tokens'),
                    motion_tokens=batch.get('motion_tokens'),
                    attention_mask=attention_masks,
                    labels=token_labels
                )
                
                if self.debug:
                    print(f"   ✅ Model forward pass completed")
                    print(f"   - Outputs type: {type(outputs)}")
                    if hasattr(outputs, 'logits'):
                        print(f"   - Logits shape: {outputs.logits.shape if outputs.logits is not None else 'None'}")
                    if hasattr(outputs, 'loss'):
                        print(f"   - Loss: {outputs.loss if outputs.loss is not None else 'None'}")
                
                # 计算损失
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # 手动计算损失
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = token_labels[..., 1:].contiguous()
                    loss = self.criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for _, param in self.model.get_trainable_parameters()],
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # 统计
                batch_loss = loss.item()
                batch_tokens = (token_labels != -100).sum().item()
                
                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                num_batches += 1
                
                # 更新进度条
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'tokens': batch_tokens
                })
                
                # 记录到wandb
                if wandb.run is not None:
                    wandb.log({
                        'train/batch_loss': batch_loss,
                        'train/avg_loss': avg_loss,
                        'train/learning_rate': current_lr,
                        'train/batch_tokens': batch_tokens,
                        'train/step': self.step
                    })
                
                self.step += 1
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算epoch统计
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        
        return {
            'train_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved to: {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """加载检查点"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
              save_dir: str = "output/unified_model"):
        """训练模型"""
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"   - Save every: {save_every} steps")
        print(f"   - Save directory: {save_dir}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            train_stats = self.train_epoch()
            
            # 记录epoch统计
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_stats['train_loss'],
                    'train/epoch_tokens': train_stats['total_tokens'],
                    'train/epoch_batches': train_stats['num_batches']
                })
            
            print(f"📊 Epoch {epoch} completed:")
            print(f"   - Train loss: {train_stats['train_loss']:.4f}")
            print(f"   - Total tokens: {train_stats['total_tokens']}")
            print(f"   - Batches: {train_stats['num_batches']}")
            
            # 定期保存
            if self.step % save_every == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"step_{self.step}.pt")
                )
        
        # 训练完成
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
    
    # 保存和日志参数
    parser.add_argument('--save_dir', type=str, default="output/unified_model",
                       help='模型保存目录')
    parser.add_argument('--save_every', type=int, default=100,
                       help='保存间隔步数')
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
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("🚀 Starting Unified Kimi-Motion Model Training")
    print(f"📋 Arguments: {vars(args)}")
    
    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"
    
    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"unified_model_{int(time.time())}"
        )
        print("✅ Wandb initialized")
    
    # 创建模型
    print("🔄 Creating unified model...")
    model = create_unified_model(
        kimi_model_path=args.kimi_model_path,
        gpt2_config_path=args.gpt2_config_path,
        freeze_kimi=args.freeze_kimi,
        freeze_adaptor=args.freeze_adaptor,
        train_mixer_only=args.train_mixer_only,
        debug=args.debug
    )
    
    # 创建数据加载器
    print("🔄 Creating data loaders...")
    
    train_dataloader = create_dataloader(
        json_paths=args.train_json_paths,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        dataset_weights=args.dataset_weights,
        debug=args.debug,
        max_audio_length=args.max_audio_length,
        max_motion_length=args.max_motion_length,
        interleave_ratio=tuple(args.interleave_ratio)
    )
    
    
    # 创建训练器
    print("🔄 Creating trainer...")
    trainer = UnifiedModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        device=device,
        debug=args.debug
    )
    
    # 恢复训练
    if args.resume_from:
        print(f"🔄 Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # 开始训练
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        save_dir=args.save_dir
    )
    
    # 关闭wandb
    if args.use_wandb:
        wandb.finish()
    
    print("🎉 Training completed successfully!")


if __name__ == "__main__":
    main()
