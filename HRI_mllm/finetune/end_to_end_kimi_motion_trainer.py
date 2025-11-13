"""
端到端Kimi + Motion Adaptor微调脚本
支持三种重参数化方案：
1. Gumbel-Softmax
2. Straight-Through Estimator (STE)
3. Hidden States（推荐）
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Literal
import yaml
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import wandb

from kimia_infer.api.kimia import KimiAudio
from HRI_mllm.model.gpt2_adaptor.hidden_state_adaptor import (
    HiddenStateGPT2Adaptor,
    EndToEndHiddenStateModel,
    setup_lora_finetuning
)
from HRI_mllm.model.gpt2_adaptor.gumbel_softmax_adaptor import (
    GumbelSoftmaxGPT2Adaptor,
    EndToEndKimiMotionModel as GumbelEndToEndModel
)
from HRI_mllm.model.gpt2_adaptor.ste_adaptor import (
    EndToEndSTEModel
)


class AudioMotionDataset(Dataset):
    """
    音频-动作对数据集
    
    数据格式：
    {
        'user_audio_path': str,  # 用户音频路径
        'assistant_audio_tokens': List[int],  # 助手音频tokens（GT）
        'motion_tokens': List[int],  # 动作tokens（GT）
    }
    """
    
    def __init__(self, data_file, kimi_model_path, max_audio_len=2048, max_motion_len=1024):
        self.data = self.load_data(data_file)
        self.max_audio_len = max_audio_len
        self.max_motion_len = max_motion_len
        
        # 初始化Kimi的tokenizer（用于处理音频）
        from kimia_infer.api.kimia import KimiAudio
        self.kimi = KimiAudio(model_path=kimi_model_path, load_detokenizer=False)
    
    def load_data(self, data_file):
        """加载数据（支持多种格式）"""
        # TODO: 根据你的数据格式实现
        # 例如：jsonl, pickle, hdf5等
        import pickle
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 处理用户音频
        user_audio_path = item['user_audio_path']
        user_audio_tokens = self.kimi.tokenize_audio(user_audio_path)  # [seq_len]
        
        # 2. 获取GT
        assistant_audio_tokens = torch.tensor(item['assistant_audio_tokens'], dtype=torch.long)
        motion_tokens = torch.tensor(item['motion_tokens'], dtype=torch.long)
        
        # 3. 构建interleaved sequence和labels
        # 这里需要根据你的interleave策略实现
        # 简化版本：audio tokens的label为-100，motion tokens的label为自身
        
        interleaved_seq, labels = self.build_interleaved_sequence(
            assistant_audio_tokens, motion_tokens
        )
        
        return {
            'user_audio_tokens': user_audio_tokens,
            'interleaved_seq': interleaved_seq,
            'labels': labels,
            'assistant_audio_tokens': assistant_audio_tokens,
            'motion_tokens': motion_tokens,
        }
    
    def build_interleaved_sequence(self, audio_tokens, motion_tokens, interleave_ratio=(1, 1)):
        """
        构建交错序列
        
        Args:
            audio_tokens: [audio_seq_len]
            motion_tokens: [motion_seq_len]  （已经是body/hand交错的）
            interleave_ratio: (num_audio, num_motion)
        
        Returns:
            interleaved_seq: [total_seq_len]
            labels: [total_seq_len] （audio位置为-100）
        """
        num_audio, num_motion = interleave_ratio
        
        interleaved = []
        labels = []
        
        audio_idx = 0
        motion_idx = 0
        
        while audio_idx < len(audio_tokens) or motion_idx < len(motion_tokens):
            # 添加audio tokens
            for _ in range(num_audio):
                if audio_idx < len(audio_tokens):
                    interleaved.append(audio_tokens[audio_idx].item())
                    labels.append(-100)  # audio token的label
                    audio_idx += 1
            
            # 添加motion tokens
            for _ in range(num_motion):
                if motion_idx < len(motion_tokens):
                    interleaved.append(motion_tokens[motion_idx].item())
                    labels.append(motion_tokens[motion_idx].item())  # motion token的label
                    motion_idx += 1
        
        return torch.tensor(interleaved, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    """DataLoader的collate函数，处理变长序列"""
    # Padding to max length in batch
    max_user_len = max(item['user_audio_tokens'].shape[0] for item in batch)
    max_seq_len = max(item['interleaved_seq'].shape[0] for item in batch)
    
    user_audio_tokens = []
    interleaved_seqs = []
    labels_list = []
    attention_masks = []
    
    for item in batch:
        # Pad user audio tokens
        user_tokens = item['user_audio_tokens']
        pad_len = max_user_len - len(user_tokens)
        user_audio_tokens.append(torch.cat([user_tokens, torch.zeros(pad_len, dtype=torch.long)]))
        
        # Pad interleaved sequence
        seq = item['interleaved_seq']
        labels = item['labels']
        pad_len = max_seq_len - len(seq)
        
        interleaved_seqs.append(torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)]))
        labels_list.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)]))
        
        # Attention mask
        mask = torch.cat([torch.ones(len(seq)), torch.zeros(pad_len)])
        attention_masks.append(mask)
    
    return {
        'user_audio_tokens': torch.stack(user_audio_tokens),
        'interleaved_seq': torch.stack(interleaved_seqs),
        'labels': torch.stack(labels_list),
        'attention_mask': torch.stack(attention_masks),
    }


class EndToEndTrainer:
    """
    端到端训练器
    """
    
    def __init__(
        self,
        config: Dict,
        method: Literal['hidden_state', 'gumbel_softmax', 'ste'] = 'hidden_state',
    ):
        self.config = config
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"Initializing End-to-End Trainer (method={method})")
        print(f"{'='*60}\n")
        
        # 加载模型
        self.kimi_model, self.gpt2_adaptor, self.model = self.build_model()
        
        # 加载数据
        self.train_loader, self.val_loader = self.build_dataloaders()
        
        # 设置优化器
        self.optimizer, self.scheduler = self.setup_optimizer()
        
        # WandB
        if config.get('use_wandb', False):
            wandb.init(project=config['wandb_project'], name=config['experiment_name'], config=config)
    
    def build_model(self):
        """构建模型"""
        config = self.config
        
        # 1. 加载Kimi model
        print("📦 Loading Kimi model...")
        kimi_model = KimiAudio(
            model_path=config['kimi_model_path'],
            load_detokenizer=False
        )
        
        # 2. 加载/创建GPT2 adaptor
        print("📦 Loading GPT2 adaptor...")
        
        if self.method == 'hidden_state':
            from transformers import GPT2Config
            gpt2_config = GPT2Config(
                vocab_size=config['motion_vocab_size'],
                n_positions=config['max_seq_length'],
                n_embd=config['gpt2_hidden_size'],
                n_layer=config['gpt2_num_layers'],
                n_head=config['gpt2_num_heads'],
            )
            gpt2_adaptor = HiddenStateGPT2Adaptor(
                gpt2_config,
                kimi_hidden_size=config['kimi_hidden_size'],
            )
            
            # 可选：加载预训练权重
            if config.get('gpt2_checkpoint'):
                checkpoint = torch.load(config['gpt2_checkpoint'], weights_only=False)
                gpt2_adaptor.load_state_dict(checkpoint['model_state'], strict=False)
                print(f"✅ Loaded GPT2 checkpoint from {config['gpt2_checkpoint']}")
            
            # 3. 创建端到端模型
            if config.get('use_lora', False):
                kimi_model.model, gpt2_adaptor = setup_lora_finetuning(
                    kimi_model.model,
                    gpt2_adaptor,
                    lora_r=config.get('lora_r', 8),
                    lora_alpha=config.get('lora_alpha', 16),
                )
            
            model = EndToEndHiddenStateModel(
                kimi_model.model,
                gpt2_adaptor,
                freeze_kimi=config.get('freeze_kimi', True),
                use_gradient_checkpointing=config.get('use_gradient_checkpointing', False),
            )
        
        elif self.method == 'gumbel_softmax':
            # TODO: 实现Gumbel-Softmax版本
            raise NotImplementedError("Gumbel-Softmax method not fully implemented yet")
        
        elif self.method == 'ste':
            # TODO: 实现STE版本
            raise NotImplementedError("STE method not fully implemented yet")
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        model = model.to(self.device)
        
        return kimi_model, gpt2_adaptor, model
    
    def build_dataloaders(self):
        """构建数据加载器"""
        config = self.config
        
        train_dataset = AudioMotionDataset(
            config['train_data_file'],
            config['kimi_model_path'],
            max_audio_len=config['max_audio_len'],
            max_motion_len=config['max_motion_len'],
        )
        
        val_dataset = AudioMotionDataset(
            config['val_data_file'],
            config['kimi_model_path'],
            max_audio_len=config['max_audio_len'],
            max_motion_len=config['max_motion_len'],
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            collate_fn=collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            collate_fn=collate_fn,
        )
        
        print(f"✅ Train dataset: {len(train_dataset)} samples")
        print(f"✅ Val dataset: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        config = self.config
        
        # 只优化requires_grad=True的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        num_training_steps = len(self.train_loader) * config['num_epochs']
        num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        print(f"✅ Optimizer: AdamW (lr={config['learning_rate']})")
        print(f"✅ Scheduler: Linear warmup + decay ({num_warmup_steps} warmup steps)")
        
        return optimizer, scheduler
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_motion_loss = 0
        total_audio_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward
            outputs = self.model(
                user_audio_tokens=batch['user_audio_tokens'],
                motion_tokens_gt=batch['interleaved_seq'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            
            loss = outputs['total_loss']
            motion_loss = outputs['motion_loss']
            audio_loss = outputs.get('audio_loss', 0)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            total_motion_loss += motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss
            total_audio_loss += audio_loss.item() if isinstance(audio_loss, torch.Tensor) else audio_loss
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'motion': f"{motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
            
            # WandB logging
            if self.config.get('use_wandb') and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/motion_loss': motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss,
                    'train/audio_loss': audio_loss.item() if isinstance(audio_loss, torch.Tensor) else audio_loss,
                    'train/lr': self.scheduler.get_last_lr()[0],
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_motion_loss = total_motion_loss / len(self.train_loader)
        
        return avg_loss, avg_motion_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        total_motion_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                outputs = self.model(
                    user_audio_tokens=batch['user_audio_tokens'],
                    motion_tokens_gt=batch['interleaved_seq'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                
                loss = outputs['total_loss']
                motion_loss = outputs['motion_loss']
                
                total_loss += loss.item()
                total_motion_loss += motion_loss.item() if isinstance(motion_loss, torch.Tensor) else motion_loss
        
        avg_loss = total_loss / len(self.val_loader)
        avg_motion_loss = total_motion_loss / len(self.val_loader)
        
        return avg_loss, avg_motion_loss
    
    def train(self):
        """完整训练流程"""
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_motion_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_motion_loss = self.validate()
            
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f} (motion: {train_motion_loss:.4f})")
            print(f"   Val Loss: {val_loss:.4f} (motion: {val_motion_loss:.4f})")
            
            # WandB
            if self.config.get('use_wandb'):
                wandb.log({
                    'epoch': epoch,
                    'val/loss': val_loss,
                    'val/motion_loss': val_motion_loss,
                })
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f"✅ Saved best checkpoint (val_loss={val_loss:.4f})")
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存checkpoint"""
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }
        
        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)


def main():
    """主函数"""
    
    # 配置
    config = {
        # 模型路径
        'kimi_model_path': '/DATA/disk1/Kimi-Audio-7B-Instruct',
        'gpt2_checkpoint': '/root/workspace/HRI_MLLM/output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1700.pt',
        
        # 数据
        'train_data_file': '/path/to/train_data.pkl',
        'val_data_file': '/path/to/val_data.pkl',
        'max_audio_len': 2048,
        'max_motion_len': 1024,
        
        # 模型配置
        'motion_vocab_size': 1034,
        'max_seq_length': 4096,
        'gpt2_hidden_size': 768,
        'gpt2_num_layers': 12,
        'gpt2_num_heads': 12,
        'kimi_hidden_size': 3584,
        
        # 训练策略
        'freeze_kimi': True,  # 冻结Kimi，只训练adaptor
        'use_lora': True,  # 使用LoRA微调
        'lora_r': 8,
        'lora_alpha': 16,
        'use_gradient_checkpointing': True,  # 节省显存
        
        # 训练超参数
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        
        # Logging & Checkpointing
        'checkpoint_dir': '/root/workspace/HRI_MLLM/output/end_to_end_kimi_motion',
        'save_every': 5,
        'use_wandb': False,
        'wandb_project': 'kimi-motion-e2e',
        'experiment_name': 'hidden_state_lora',
        
        # 其他
        'num_workers': 4,
    }
    
    # 创建trainer
    trainer = EndToEndTrainer(config, method='hidden_state')
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()


