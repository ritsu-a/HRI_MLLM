"""
基于Gumbel-Softmax的端到端Kimi + Motion Adaptor训练脚本

核心思路：
1. 修改Kimi model返回logits（而不是采样的tokens）
2. 使用Gumbel-Softmax从logits生成soft tokens（可微）
3. 将soft tokens通过audio embedding得到soft embeddings
4. GPT2 adaptor基于soft embeddings生成motion tokens
5. Motion loss的梯度可以通过Gumbel-Softmax回传到Kimi
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np
from tqdm import tqdm
import yaml
from transformers import GPT2Config
from dotenv import load_dotenv

# Kimi
from kimia_infer.api.kimia import KimiAudio

# Motion 相关
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats
from HRI_mllm import ROOT

# Gumbel-Softmax Adaptor
from HRI_mllm.model.gpt2_adaptor.gumbel_softmax_adaptor import GumbelSoftmaxGPT2Adaptor
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2


class GumbelKimiWrapper(nn.Module):
    """
    包装Kimi model，使其支持返回logits
    用于训练时的Gumbel-Softmax
    """
    
    def __init__(self, kimi_model):
        super().__init__()
        self.kimi = kimi_model.model  # 内部的模型
        self.kimia_token_offset = kimi_model.kimia_token_offset
        
    def forward_with_logits(
        self,
        audio_input_ids,
        continous_feature=None,
        text_input_ids=None,
        attention_mask=None,
        max_new_tokens=256,
    ):
        """
        修改的forward，返回logits而不是采样的tokens
        
        Args:
            audio_input_ids: [batch, seq_len] 用户音频tokens
            continous_feature: Optional whisper features
            max_new_tokens: 最大生成token数
            
        Returns:
            logits: [batch, generated_seq_len, vocab_size] assistant audio logits
        """
        # 构建输入
        batch_size = audio_input_ids.shape[0]
        device = audio_input_ids.device
        
        # 准备is_continuous_mask（如果使用whisper features）
        if continous_feature is not None:
            is_continuous_mask = torch.ones(
                batch_size, audio_input_ids.shape[1],
                dtype=torch.bool, device=device
            )
        else:
            is_continuous_mask = None
        
        # 调用模型获取outputs
        outputs = self.kimi(
            input_ids=audio_input_ids,
            whisper_input_feature=continous_feature,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # 获取logits
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # 只返回assistant音频部分的logits
        # 简化：返回最后max_new_tokens个位置的logits
        if max_new_tokens < logits.shape[1]:
            logits = logits[:, -max_new_tokens:, :]
        
        return logits


def gumbel_softmax_sampling(logits, tau=1.0, hard=False, dim=-1):
    """
    Gumbel-Softmax采样
    
    Args:
        logits: [batch, seq_len, vocab_size]
        tau: 温度参数（越小越接近one-hot）
        hard: 是否使用straight-through（前向hard，反向soft）
        
    Returns:
        soft_tokens: [batch, seq_len, vocab_size] soft token distribution
    """
    # 添加Gumbel噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    gumbel_logits = (logits + gumbel_noise) / tau
    
    # Softmax
    soft_tokens = F.softmax(gumbel_logits, dim=dim)
    
    if hard:
        # Straight-through estimator
        hard_tokens = torch.zeros_like(soft_tokens)
        hard_tokens.scatter_(dim, soft_tokens.argmax(dim=dim, keepdim=True), 1.0)
        
        # 前向hard，反向soft
        soft_tokens = hard_tokens - soft_tokens.detach() + soft_tokens
    
    return soft_tokens


class EndToEndGumbelModel(nn.Module):
    """
    端到端Kimi + Gumbel-Softmax + GPT2 Adaptor模型
    """
    
    def __init__(
        self,
        kimi_model,
        gpt2_adaptor,
        audio_embedding_weight,  # Kimi的audio embedding权重
        freeze_kimi=True,
        freeze_audio_embedding=True,
        initial_tau=1.0,
        min_tau=0.5,
        tau_decay_rate=0.9999,
    ):
        super().__init__()
        
        self.kimi_wrapper = GumbelKimiWrapper(kimi_model)
        self.gpt2_adaptor = gpt2_adaptor
        
        # Audio embedding（用于将soft tokens转为soft embeddings）
        self.audio_embedding = nn.Embedding.from_pretrained(
            audio_embedding_weight,
            freeze=freeze_audio_embedding,
        )
        
        # Projection layer（对齐audio hidden size和GPT2 hidden size）
        audio_hidden_size = audio_embedding_weight.shape[1]
        gpt2_hidden_size = gpt2_adaptor.config.hidden_size
        
        if audio_hidden_size != gpt2_hidden_size:
            self.audio_projection = nn.Linear(audio_hidden_size, gpt2_hidden_size)
        else:
            self.audio_projection = nn.Identity()
        
        # 冻结策略
        if freeze_kimi:
            for param in self.kimi_wrapper.parameters():
                param.requires_grad = False
            print("❄️  Kimi model frozen")
        
        if freeze_audio_embedding:
            for param in self.audio_embedding.parameters():
                param.requires_grad = False
            print("❄️  Audio embedding frozen")
        
        # Gumbel温度调度
        self.current_tau = initial_tau
        self.min_tau = min_tau
        self.tau_decay_rate = tau_decay_rate
        self.step_count = 0
        
        print(f"✅ EndToEndGumbelModel initialized")
        print(f"   Initial tau: {initial_tau}")
        print(f"   Min tau: {min_tau}")
        print(f"   Tau decay rate: {tau_decay_rate}")
    
    def update_tau(self):
        """更新Gumbel温度（每个训练step调用）"""
        self.current_tau = max(
            self.min_tau,
            self.current_tau * self.tau_decay_rate
        )
        self.step_count += 1
    
    def forward(
        self,
        user_audio_tokens,  # [batch, user_seq_len]
        motion_tokens_gt,   # [batch, motion_seq_len]
        user_audio_features=None,  # Optional whisper features
        use_gumbel_hard=False,  # 是否使用straight-through
        interleave_ratio=(1, 1),  # (audio, motion)交错比例
    ):
        """
        端到端前向传播
        
        Args:
            user_audio_tokens: 用户音频tokens（输入）
            motion_tokens_gt: Ground truth motion tokens
            use_gumbel_hard: 是否使用hard Gumbel（straight-through）
            
        Returns:
            outputs: {
                'total_loss': total loss,
                'motion_loss': motion generation loss,
                'audio_loss': optional audio reconstruction loss,
                'gumbel_tau': current temperature,
            }
        """
        device = user_audio_tokens.device
        batch_size = user_audio_tokens.shape[0]
        
        # 1️⃣ Kimi生成assistant audio logits
        with torch.set_grad_enabled(not self.kimi_wrapper.training):
            audio_logits = self.kimi_wrapper.forward_with_logits(
                audio_input_ids=user_audio_tokens,
                continous_feature=user_audio_features,
                max_new_tokens=512,
            )
        
        # [batch, audio_seq_len, audio_vocab_size]
        print(f"🔥 Audio logits shape: {audio_logits.shape}")
        
        # 2️⃣ Gumbel-Softmax采样（可微！）
        soft_audio_tokens = gumbel_softmax_sampling(
            audio_logits,
            tau=self.current_tau,
            hard=use_gumbel_hard,
        )
        # [batch, audio_seq_len, audio_vocab_size]
        print(f"🔥 Soft audio tokens shape: {soft_audio_tokens.shape}")
        print(f"   Current tau: {self.current_tau:.4f}")
        
        # 3️⃣ 通过audio embedding得到soft embeddings
        # soft_embeddings = soft_tokens @ embedding_matrix
        soft_audio_embeds = torch.matmul(
            soft_audio_tokens,
            self.audio_embedding.weight
        )
        # [batch, audio_seq_len, audio_hidden_dim]
        
        # Project到GPT2 hidden size
        projected_audio_embeds = self.audio_projection(soft_audio_embeds)
        # [batch, audio_seq_len, gpt2_hidden_dim]
        
        print(f"🔥 Projected audio embeds shape: {projected_audio_embeds.shape}")
        
        # 4️⃣ 构建interleaved sequence和labels
        # 这里需要根据您的数据格式构建
        # 简化版本：假设audio和motion按照interleave_ratio交错
        
        interleaved_embeds, labels = self.build_interleaved_sequence(
            projected_audio_embeds,
            motion_tokens_gt,
            interleave_ratio,
        )
        
        # 5️⃣ GPT2 Adaptor处理
        # 注意：这里需要修改adaptor使其接受embeddings而不是token ids
        outputs = self.gpt2_adaptor(
            inputs_embeds=interleaved_embeds,  # 直接传embeddings
            attention_mask=None,
            labels=labels,
        )
        
        motion_loss = outputs.loss
        
        # 更新温度
        if self.training:
            self.update_tau()
        
        return {
            'total_loss': motion_loss,
            'motion_loss': motion_loss,
            'gumbel_tau': self.current_tau,
            'logits': outputs.logits,
        }
    
    def build_interleaved_sequence(
        self,
        audio_embeds,  # [batch, audio_seq_len, hidden_dim]
        motion_tokens,  # [batch, motion_seq_len]
        interleave_ratio=(1, 1),
    ):
        """
        构建audio和motion的交错序列
        
        Returns:
            interleaved_embeds: [batch, total_seq_len, hidden_dim]
            labels: [batch, total_seq_len] (audio位置=-100, motion位置=token_id)
        """
        batch_size = audio_embeds.shape[0]
        audio_seq_len = audio_embeds.shape[1]
        motion_seq_len = motion_tokens.shape[1]
        hidden_dim = audio_embeds.shape[2]
        device = audio_embeds.device
        
        num_audio, num_motion = interleave_ratio
        
        # 估算总序列长度
        max_groups = min(
            audio_seq_len // num_audio,
            motion_seq_len // num_motion
        )
        total_seq_len = max_groups * (num_audio + num_motion)
        
        # 初始化
        interleaved_embeds = torch.zeros(
            batch_size, total_seq_len, hidden_dim,
            dtype=audio_embeds.dtype, device=device
        )
        labels = torch.full(
            (batch_size, total_seq_len),
            -100, dtype=torch.long, device=device
        )
        
        # 获取motion embeddings
        motion_embeds = self.gpt2_adaptor.transformer.wte(motion_tokens)
        
        # 交错填充
        for i in range(max_groups):
            # Audio positions
            start_pos = i * (num_audio + num_motion)
            audio_start_idx = i * num_audio
            audio_end_idx = audio_start_idx + num_audio
            
            interleaved_embeds[:, start_pos:start_pos+num_audio, :] = \
                audio_embeds[:, audio_start_idx:audio_end_idx, :]
            # labels保持-100（audio不计算loss）
            
            # Motion positions
            motion_start_idx = i * num_motion
            motion_end_idx = motion_start_idx + num_motion
            motion_start_pos = start_pos + num_audio
            
            interleaved_embeds[:, motion_start_pos:motion_start_pos+num_motion, :] = \
                motion_embeds[:, motion_start_idx:motion_end_idx, :]
            labels[:, motion_start_pos:motion_start_pos+num_motion] = \
                motion_tokens[:, motion_start_idx:motion_end_idx]
        
        return interleaved_embeds, labels


def load_audio_motion_dataset(data_file):
    """加载音频-动作对数据集"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Loaded {len(data)} samples from {data_file}")
    return data


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_motion_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        user_audio_tokens = batch['user_audio_tokens'].to(device)
        motion_tokens_gt = batch['motion_tokens'].to(device)
        
        # Forward
        outputs = model(
            user_audio_tokens=user_audio_tokens,
            motion_tokens_gt=motion_tokens_gt,
            use_gumbel_hard=(epoch > 10),  # 训练后期使用hard模式
        )
        
        loss = outputs['total_loss']
        motion_loss = outputs['motion_loss']
        tau = outputs['gumbel_tau']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        total_motion_loss += motion_loss.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'motion': f"{motion_loss.item():.4f}",
            'tau': f"{tau:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_motion_loss = total_motion_loss / len(dataloader)
    
    return avg_loss, avg_motion_loss


def main():
    """主训练函数"""
    
    print("\n" + "="*60)
    print("🔥 Gumbel-Softmax端到端训练")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 配置 ====================
    config = {
        # 模型路径
        'kimi_model_path': '/DATA/disk1/Kimi-Audio-7B-Instruct',
        'gpt2_checkpoint': '/root/workspace/HRI_MLLM/output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1700.pt',
        'audio_embedding_weight': '/root/workspace/HRI_MLLM/output/motion_adaptor_v1/embed_tokens_weight.pt',
        
        # 数据
        'train_data_file': '/path/to/your/train_data.pkl',  # 🔥 修改为您的数据路径
        
        # Gumbel-Softmax参数
        'initial_tau': 1.0,
        'min_tau': 0.5,
        'tau_decay_rate': 0.9999,  # 每步衰减
        
        # 训练策略
        'freeze_kimi': True,
        'freeze_audio_embedding': True,
        
        # 训练超参数
        'batch_size': 2,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        
        # 保存
        'output_dir': '/root/workspace/HRI_MLLM/output/gumbel_kimi_motion',
        'save_every': 5,
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # ==================== 加载模型 ====================
    
    print("📦 Loading Kimi model...")
    kimi_model = KimiAudio(
        model_path=config['kimi_model_path'],
        load_detokenizer=False,
    )
    
    print("📦 Loading GPT2 adaptor...")
    checkpoint = torch.load(config['gpt2_checkpoint'], map_location='cpu', weights_only=False)
    
    gpt2_config = GPT2Config(
        vocab_size=1034,
        n_positions=4096,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    
    # 使用原来的MixedInputGPT2
    gpt2_adaptor = MixedInputGPT2(gpt2_config, audio_hidden_size=3584)
    gpt2_adaptor.load_state_dict(checkpoint['model_state'], strict=False)
    
    print("📦 Loading audio embedding...")
    audio_embedding_weight = torch.load(
        config['audio_embedding_weight'],
        weights_only=True
    )
    
    # 创建端到端模型
    print("🔧 Creating end-to-end model...")
    model = EndToEndGumbelModel(
        kimi_model=kimi_model,
        gpt2_adaptor=gpt2_adaptor,
        audio_embedding_weight=audio_embedding_weight,
        freeze_kimi=config['freeze_kimi'],
        freeze_audio_embedding=config['freeze_audio_embedding'],
        initial_tau=config['initial_tau'],
        min_tau=config['min_tau'],
        tau_decay_rate=config['tau_decay_rate'],
    )
    model = model.to(device)
    
    # ==================== 加载数据 ====================
    
    print("📦 Loading dataset...")
    # 🔥 这里需要您准备数据
    # data = load_audio_motion_dataset(config['train_data_file'])
    
    # 暂时使用dummy data演示
    print("⚠️  使用dummy data进行演示，请替换为真实数据")
    
    # ==================== 优化器 ====================
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    # 简单的学习率调度器
    from transformers import get_linear_schedule_with_warmup
    num_training_steps = 1000  # 根据您的数据集大小调整
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    print(f"\n✅ Setup complete!")
    print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"   Device: {device}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    
    print(f"\n🚀 开始训练...")
    print(f"   注意：请准备好您的数据集并修改data loading部分")


if __name__ == "__main__":
    main()


