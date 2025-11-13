"""
混合微调策略：Kimi LoRA + Adaptor 全参数微调

策略说明：
1. Kimi (7B参数): 使用LoRA微调，只训练少量参数（~8M）
2. GPT2 Adaptor (100M参数): 全参数微调，充分训练
3. 使用不同的学习率：Kimi LoRA用较小lr，Adaptor用较大lr

优势：
- 参数高效：Kimi只增加1-2%的参数
- 训练充分：Adaptor可以完全适配新数据
- 稳定性好：Kimi的LoRA防止过拟合，Adaptor全参数保证表达能力
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
from transformers import GPT2Config, get_linear_schedule_with_warmup

# LoRA
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Kimi
from kimia_infer.api.kimia import KimiAudio

# Motion 相关
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats
from HRI_mllm import ROOT, OUTPUT_ROOT

# Adaptor
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2


def setup_kimi_lora(kimi_model, lora_r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    """
    为Kimi model添加LoRA
    
    Args:
        kimi_model: Kimi的原始model（通常是Qwen2模型）
        lora_r: LoRA rank（秩），越大表达能力越强但参数越多
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA的dropout
        target_modules: 要应用LoRA的模块名称
        
    Returns:
        lora_model: 添加了LoRA的模型
    """
    print("\n" + "="*60)
    print("🔧 为Kimi添加LoRA")
    print("="*60)
    
    # 🔥 Monkey patch: 添加prepare_inputs_for_generation方法（peft要求）
    if not hasattr(kimi_model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
            if past_key_values is not None:
                input_ids = input_ids[:, -1:]
            
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}
            
            model_inputs.update({
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "text_input_ids": kwargs.get("text_input_ids"),
                "whisper_input_feature": kwargs.get("whisper_input_feature"),
                "is_continuous_mask": kwargs.get("is_continuous_mask"),
            })
            return model_inputs
        
        import types
        kimi_model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, kimi_model
        )
        print("✅ Added prepare_inputs_for_generation method to Kimi model")
    
    # 默认target modules（针对Qwen2/Llama架构）
    if target_modules is None:
        target_modules = [
            "q_proj",      # Query projection
            "k_proj",      # Key projection  
            "v_proj",      # Value projection
            "o_proj",      # Output projection
            "gate_proj",   # FFN gate
            "up_proj",     # FFN up
            "down_proj",   # FFN down
        ]
    
    # 创建LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",  # 不训练bias
        inference_mode=False,  # 训练模式
    )
    
    print(f"📋 LoRA配置:")
    print(f"   Rank (r): {lora_r}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Dropout: {lora_dropout}")
    print(f"   Target modules: {target_modules}")
    
    # 应用LoRA
    lora_model = get_peft_model(kimi_model, lora_config)
    
    # 打印参数统计
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    
    print(f"\n📊 参数统计:")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # 验证LoRA是否正确应用
    lora_layers = []
    for name, module in lora_model.named_modules():
        if 'lora' in name.lower():
            lora_layers.append(name)
    
    print(f"\n✅ LoRA layers添加成功: {len(lora_layers)} 个LoRA模块")
    if len(lora_layers) > 0:
        print(f"   示例: {lora_layers[:3]}")
    
    return lora_model


def setup_adaptor_full_finetune(gpt2_adaptor, freeze_embedding=True):
    """
    设置GPT2 Adaptor为全参数微调模式
    
    Args:
        gpt2_adaptor: GPT2 adaptor模型
        freeze_embedding: 是否冻结embedding层（推荐True，保持稳定）
        
    Returns:
        gpt2_adaptor: 配置好的模型
    """
    print("\n" + "="*60)
    print("🔧 配置GPT2 Adaptor为全参数微调")
    print("="*60)
    
    # 解冻所有参数
    for param in gpt2_adaptor.parameters():
        param.requires_grad = True
    
    # 可选：冻结embedding层（提高稳定性）
    if freeze_embedding:
        if hasattr(gpt2_adaptor, 'transformer') and hasattr(gpt2_adaptor.transformer, 'wte'):
            for param in gpt2_adaptor.transformer.wte.parameters():
                param.requires_grad = False
            print("❄️  Embedding层已冻结")
        
        if hasattr(gpt2_adaptor, 'audio_tokenizer'):
            for param in gpt2_adaptor.audio_tokenizer.parameters():
                param.requires_grad = False
            print("❄️  Audio tokenizer已冻结")
    
    # 统计参数
    trainable_params = sum(p.numel() for p in gpt2_adaptor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in gpt2_adaptor.parameters())
    
    print(f"\n📊 Adaptor参数统计:")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return gpt2_adaptor


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """Gumbel-Softmax采样（用于端到端训练）"""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    gumbel_logits = (logits + gumbel_noise) / tau
    soft_tokens = F.softmax(gumbel_logits, dim=dim)
    
    if hard:
        hard_tokens = torch.zeros_like(soft_tokens)
        hard_tokens.scatter_(dim, soft_tokens.argmax(dim=dim, keepdim=True), 1.0)
        soft_tokens = hard_tokens - soft_tokens.detach() + soft_tokens
    
    return soft_tokens


class MixedFineTuneModel(nn.Module):
    """
    混合微调模型：Kimi LoRA + Adaptor Full
    """
    
    def __init__(
        self,
        kimi_lora_model,
        gpt2_adaptor,
        audio_embedding,
        projection_layer=None,
        gumbel_tau=1.0,
    ):
        super().__init__()
        
        self.kimi = kimi_lora_model
        self.adaptor = gpt2_adaptor
        self.audio_embedding = audio_embedding
        
        # Projection layer（对齐audio和GPT2的hidden size）
        if projection_layer is None:
            audio_hidden = audio_embedding.embedding_dim
            gpt2_hidden = gpt2_adaptor.config.hidden_size
            if audio_hidden != gpt2_hidden:
                self.projection = nn.Linear(audio_hidden, gpt2_hidden)
            else:
                self.projection = nn.Identity()
        else:
            self.projection = projection_layer
        
        self.gumbel_tau = gumbel_tau
        
        # 冻结audio embedding（它是预训练的，不需要训练）
        for param in self.audio_embedding.parameters():
            param.requires_grad = False
    
    def forward_with_logits(
        self,
        user_audio_tokens,
        user_audio_features=None,
        max_new_tokens=512,
    ):
        """
        Kimi forward并返回logits
        
        注意：这里需要修改以适配您的Kimi版本
        """
        # 方法1：直接使用model.forward（简单但不完整）
        outputs = self.kimi(
            input_ids=user_audio_tokens,
            attention_mask=None,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # 取最后若干个token的logits作为assistant audio
        logits = outputs.logits[:, -max_new_tokens:, :]
        
        return logits
    
    def forward(
        self,
        user_audio_tokens,
        motion_tokens_gt,
        user_audio_features=None,
        use_gumbel_hard=False,
        interleave_ratio=(1, 1),
    ):
        """
        端到端forward（支持Gumbel-Softmax）
        
        Args:
            user_audio_tokens: [batch, user_seq_len]
            motion_tokens_gt: [batch, motion_seq_len]
            use_gumbel_hard: 是否使用straight-through
            
        Returns:
            outputs: {
                'total_loss': loss,
                'logits': motion_logits,
                'kimi_grad_norm': LoRA梯度norm（调试用）
            }
        """
        device = user_audio_tokens.device
        batch_size = user_audio_tokens.shape[0]
        
        # 1. Kimi LoRA生成audio logits
        audio_logits = self.forward_with_logits(
            user_audio_tokens,
            user_audio_features,
            max_new_tokens=512,
        )
        
        # 2. Gumbel-Softmax
        soft_audio_tokens = gumbel_softmax(
            audio_logits,
            tau=self.gumbel_tau,
            hard=use_gumbel_hard,
        )
        
        # 3. 转为embeddings
        soft_audio_embeds = torch.matmul(
            soft_audio_tokens,
            self.audio_embedding.weight
        )
        
        # 4. Projection
        projected_embeds = self.projection(soft_audio_embeds)
        
        # 5. 构建interleaved sequence
        inputs_embeds, labels = self.build_interleaved_sequence(
            projected_embeds,
            motion_tokens_gt,
            interleave_ratio,
        )
        
        # 6. GPT2 Adaptor forward（全参数训练）
        transformer_outputs = self.adaptor.transformer(inputs_embeds=inputs_embeds)
        hidden_states = transformer_outputs[0]
        logits = self.adaptor.lm_head(hidden_states)
        
        # 7. 计算loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return {
            'total_loss': loss,
            'motion_loss': loss,
            'logits': logits,
        }
    
    def build_interleaved_sequence(
        self,
        audio_embeds,
        motion_tokens,
        interleave_ratio=(1, 1),
    ):
        """构建audio和motion的交错序列"""
        batch_size = audio_embeds.shape[0]
        audio_seq_len = audio_embeds.shape[1]
        motion_seq_len = motion_tokens.shape[1]
        hidden_dim = audio_embeds.shape[2]
        device = audio_embeds.device
        
        num_audio, num_motion = interleave_ratio
        
        # 计算序列长度
        max_groups = min(
            audio_seq_len // num_audio,
            motion_seq_len // num_motion
        )
        total_len = max_groups * (num_audio + num_motion)
        
        # 初始化
        inputs_embeds = torch.zeros(
            batch_size, total_len, hidden_dim,
            dtype=audio_embeds.dtype, device=device
        )
        labels = torch.full(
            (batch_size, total_len),
            -100, dtype=torch.long, device=device
        )
        
        # 获取motion embeddings
        motion_embeds = self.adaptor.transformer.wte(motion_tokens)
        
        # 交错填充
        for i in range(max_groups):
            # Audio
            start_pos = i * (num_audio + num_motion)
            audio_start = i * num_audio
            audio_end = audio_start + num_audio
            
            inputs_embeds[:, start_pos:start_pos+num_audio, :] = \
                audio_embeds[:, audio_start:audio_end, :]
            
            # Motion
            motion_start = i * num_motion
            motion_end = motion_start + num_motion
            motion_pos = start_pos + num_audio
            
            inputs_embeds[:, motion_pos:motion_pos+num_motion, :] = \
                motion_embeds[:, motion_start:motion_end, :]
            labels[:, motion_pos:motion_pos+num_motion] = \
                motion_tokens[:, motion_start:motion_end]
        
        return inputs_embeds, labels


def create_optimizer_with_different_lrs(model, kimi_lr=5e-5, adaptor_lr=1e-4, weight_decay=0.01):
    """
    创建使用不同学习率的优化器
    
    Args:
        model: MixedFineTuneModel
        kimi_lr: Kimi LoRA的学习率（较小）
        adaptor_lr: Adaptor的学习率（较大）
        weight_decay: 权重衰减
    
    Returns:
        optimizer: 配置好的优化器
    """
    print("\n" + "="*60)
    print("🔧 配置优化器（不同学习率）")
    print("="*60)
    
    # 分组参数
    kimi_params = []
    adaptor_params = []
    projection_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'kimi' in name or 'lora' in name.lower():
            kimi_params.append(param)
        elif 'projection' in name:
            projection_params.append(param)
        else:
            adaptor_params.append(param)
    
    # 创建参数组
    param_groups = [
        {
            'params': kimi_params,
            'lr': kimi_lr,
            'name': 'kimi_lora',
        },
        {
            'params': projection_params,
            'lr': adaptor_lr,  # projection用adaptor的lr
            'name': 'projection',
        },
        {
            'params': adaptor_params,
            'lr': adaptor_lr,
            'name': 'adaptor',
        },
    ]
    
    # 打印参数组信息
    print(f"📊 参数分组:")
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"   {group['name']:15s}: {num_params:10,} params, lr={group['lr']:.2e}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    print(f"\n✅ 优化器创建成功")
    
    return optimizer


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        user_audio_tokens = batch['user_audio_tokens'].to(device)
        motion_tokens_gt = batch['motion_tokens'].to(device)
        
        # Forward
        outputs = model(
            user_audio_tokens=user_audio_tokens,
            motion_tokens_gt=motion_tokens_gt,
            use_gumbel_hard=(epoch > 10),
        )
        
        loss = outputs['total_loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        
        # 显示不同参数组的学习率
        lrs = {group['name']: group['lr'] for group in optimizer.param_groups}
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'kimi_lr': f"{lrs.get('kimi_lora', 0):.2e}",
            'adapt_lr': f"{lrs.get('adaptor', 0):.2e}",
        })
    
    return total_loss / len(dataloader)


def main():
    """主训练函数"""
    
    print("\n" + "🔥"*30)
    print("混合微调：Kimi LoRA + Adaptor Full")
    print("🔥"*30 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 配置 ====================
    config = {
        # 模型路径
        'kimi_model_path': '/DATA/disk1/Kimi-Audio-7B-Instruct',
        'gpt2_checkpoint': '/root/workspace/HRI_MLLM/output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1700.pt',
        'audio_embedding_weight': f'{OUTPUT_ROOT}/motion_adaptor_v1/embed_tokens_weight.pt',
        
        # LoRA配置
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        
        # 训练策略
        'kimi_lr': 5e-5,      # Kimi LoRA用较小学习率
        'adaptor_lr': 1e-4,   # Adaptor用较大学习率
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'num_epochs': 30,
        'batch_size': 2,
        'gradient_accumulation_steps': 4,
        
        # Gumbel-Softmax
        'initial_tau': 1.0,
        'min_tau': 0.5,
        'tau_decay': 0.9999,
        
        # 输出
        'output_dir': '/root/workspace/HRI_MLLM/output/mixed_finetune_kimi_lora_adaptor_full',
        'save_every': 5,
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # ==================== 加载模型 ====================
    
    # 1. 加载Kimi并添加LoRA
    print("📦 Loading Kimi model...")
    kimi_audio = KimiAudio(
        model_path=config['kimi_model_path'],
        load_detokenizer=False,
    )
    
    kimi_lora = setup_kimi_lora(
        kimi_audio.alm,  # 🔥 修复：使用alm而不是model
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
    )
    
    # 2. 加载GPT2 Adaptor（全参数微调）
    print("\n📦 Loading GPT2 Adaptor...")
    checkpoint = torch.load(config['gpt2_checkpoint'], map_location='cpu', weights_only=False)
    
    gpt2_config = GPT2Config(
        vocab_size=1034,
        n_positions=4096,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    
    gpt2_adaptor = MixedInputGPT2(gpt2_config, audio_hidden_size=3584)
    gpt2_adaptor.load_state_dict(checkpoint['model_state'], strict=False)
    
    gpt2_adaptor = setup_adaptor_full_finetune(
        gpt2_adaptor,
        freeze_embedding=True,
    )
    
    # 3. 加载audio embedding
    print("\n📦 Loading audio embedding...")
    audio_emb_weight = torch.load(config['audio_embedding_weight'], weights_only=True)
    audio_embedding = nn.Embedding.from_pretrained(audio_emb_weight, freeze=True)
    
    # 4. 创建混合模型
    print("\n🔧 Creating mixed finetune model...")
    model = MixedFineTuneModel(
        kimi_lora_model=kimi_lora,
        gpt2_adaptor=gpt2_adaptor,
        audio_embedding=audio_embedding,
        gumbel_tau=config['initial_tau'],
    )
    model = model.to(device)
    
    # ==================== 优化器 ====================
    
    optimizer = create_optimizer_with_different_lrs(
        model,
        kimi_lr=config['kimi_lr'],
        adaptor_lr=config['adaptor_lr'],
        weight_decay=config['weight_decay'],
    )
    
    # 学习率调度器
    num_training_steps = 1000  # 根据数据集大小调整
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # ==================== 总结 ====================
    
    print("\n" + "="*60)
    print("📋 训练配置总结")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    print(f"\n学习率配置:")
    print(f"  Kimi LoRA: {config['kimi_lr']:.2e} (较小，稳定)")
    print(f"  Adaptor: {config['adaptor_lr']:.2e} (较大，充分训练)")
    
    print(f"\n训练设置:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    
    print(f"\n💾 输出目录: {config['output_dir']}")
    
    print(f"\n⚠️  注意: 请准备您的数据集并开始训练")
    
    # ==================== 保存LoRA配置 ====================
    
    # 保存LoRA adapter
    lora_save_path = os.path.join(config['output_dir'], 'kimi_lora_adapter')
    os.makedirs(lora_save_path, exist_ok=True)
    
    print(f"\n💡 训练完成后，LoRA adapter将保存到: {lora_save_path}")
    print(f"   使用以下代码加载:")
    print(f"   >>> from peft import PeftModel")
    print(f"   >>> model = PeftModel.from_pretrained(base_model, '{lora_save_path}')")


if __name__ == "__main__":
    main()

