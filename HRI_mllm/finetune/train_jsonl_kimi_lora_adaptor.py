"""
基于JSONL数据的完整训练脚本
数据格式：/root/workspace/HRI_MLLM/data/HAND_RING_1111_tokens.jsonl

数据结构：
- user.text: 系统提示（不用）
- user.audio_tokens: Kimi的输入（不计算loss）
- assistant.audio_tokens: Kimi的输出GT（计算audio loss）
- assistant.motion_tokens: Adaptor的输出GT（计算motion loss）

训练策略：
- Kimi: LoRA微调（r=8）
- Adaptor: 全参数微调
- 使用Gumbel-Softmax连接两个模型
- 计算两个loss: audio_loss + motion_loss
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from transformers import GPT2Config, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# Kimi
from kimia_infer.api.kimia import KimiAudio

# Adaptor
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from HRI_mllm import OUTPUT_ROOT

torch.cuda.set_device(0)


# ==================== Dataset ====================

class JSONLAudioMotionDataset(Dataset):
    """
    加载JSONL格式的audio-motion数据集
    """
    
    def __init__(self, jsonl_file, max_audio_len=512, max_motion_len=512):
        self.jsonl_file = jsonl_file
        self.max_audio_len = max_audio_len
        self.max_motion_len = max_motion_len
        
        # 加载数据
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"✅ Loaded {len(self.data)} samples from {jsonl_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 解析conversation
        conversation = item['conversation']
        
        # 提取user audio tokens（输入，不计算loss）
        user_audio_tokens = None
        for msg in conversation:
            if msg['role'] == 'user' and msg.get('message_type') == 'audio':
                user_audio_tokens = msg['audio_tokens']
                break
        
        # 提取assistant audio和motion tokens（输出，计算loss）
        assistant_audio_tokens = None
        motion_tokens = None
        for msg in conversation:
            if msg['role'] == 'assistant' and msg.get('message_type') == 'audio_motion':
                assistant_audio_tokens = msg['audio_tokens']
                motion_tokens = msg['motion_tokens']
                break
        
        # 验证数据
        if user_audio_tokens is None or assistant_audio_tokens is None or motion_tokens is None:
            print(f"⚠️  Warning: Missing data at index {idx}")
            # 返回dummy数据
            return {
                'user_audio_tokens': torch.zeros(10, dtype=torch.long),
                'assistant_audio_tokens': torch.zeros(10, dtype=torch.long),
                'motion_tokens': torch.zeros(10, dtype=torch.long),
            }
        
        # 截断到最大长度
        user_audio_tokens = user_audio_tokens[:self.max_audio_len]
        assistant_audio_tokens = assistant_audio_tokens[:self.max_audio_len]
        motion_tokens = motion_tokens[:self.max_motion_len]
        
        return {
            'user_audio_tokens': torch.tensor(user_audio_tokens, dtype=torch.long),
            'assistant_audio_tokens': torch.tensor(assistant_audio_tokens, dtype=torch.long),
            'motion_tokens': torch.tensor(motion_tokens, dtype=torch.long),
        }


def collate_fn(batch):
    """
    DataLoader的collate函数，处理变长序列
    """
    # 找到最大长度
    max_user_len = max(item['user_audio_tokens'].shape[0] for item in batch)
    max_assistant_len = max(item['assistant_audio_tokens'].shape[0] for item in batch)
    max_motion_len = max(item['motion_tokens'].shape[0] for item in batch)
    
    # Padding
    user_audio_tokens_list = []
    assistant_audio_tokens_list = []
    motion_tokens_list = []
    user_attention_masks = []
    assistant_attention_masks = []
    
    for item in batch:
        # User audio
        user_tokens = item['user_audio_tokens']
        user_pad_len = max_user_len - len(user_tokens)
        user_audio_tokens_list.append(
            torch.cat([user_tokens, torch.zeros(user_pad_len, dtype=torch.long)])
        )
        user_attention_masks.append(
            torch.cat([torch.ones(len(user_tokens)), torch.zeros(user_pad_len)])
        )
        
        # Assistant audio
        assistant_tokens = item['assistant_audio_tokens']
        assistant_pad_len = max_assistant_len - len(assistant_tokens)
        assistant_audio_tokens_list.append(
            torch.cat([assistant_tokens, torch.zeros(assistant_pad_len, dtype=torch.long)])
        )
        assistant_attention_masks.append(
            torch.cat([torch.ones(len(assistant_tokens)), torch.zeros(assistant_pad_len)])
        )
        
        # Motion
        motion_tokens = item['motion_tokens']
        motion_pad_len = max_motion_len - len(motion_tokens)
        motion_tokens_list.append(
            torch.cat([motion_tokens, torch.zeros(motion_pad_len, dtype=torch.long)])
        )
    
    return {
        'user_audio_tokens': torch.stack(user_audio_tokens_list),
        'assistant_audio_tokens': torch.stack(assistant_audio_tokens_list),
        'motion_tokens': torch.stack(motion_tokens_list),
        'user_attention_mask': torch.stack(user_attention_masks),
        'assistant_attention_mask': torch.stack(assistant_attention_masks),
    }


# ==================== Gumbel-Softmax ====================

def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """Gumbel-Softmax重参数化"""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    gumbel_logits = (logits + gumbel_noise) / tau
    soft_tokens = F.softmax(gumbel_logits, dim=dim)
    
    if hard:
        hard_tokens = torch.zeros_like(soft_tokens)
        hard_tokens.scatter_(dim, soft_tokens.argmax(dim=dim, keepdim=True), 1.0)
        soft_tokens = hard_tokens - soft_tokens.detach() + soft_tokens
    
    return soft_tokens


# ==================== Model ====================

class KimiMotionModel(nn.Module):
    """
    端到端Kimi + Adaptor模型（支持双loss）
    """
    
    def __init__(
        self,
        kimi_model,
        gpt2_adaptor,
        audio_embedding,
        projection_layer=None,
        gumbel_tau=1.0,
        audio_loss_weight=0.1,
        motion_loss_weight=1.0,
    ):
        super().__init__()
        
        self.kimi = kimi_model
        self.adaptor = gpt2_adaptor
        self.audio_embedding = audio_embedding
        
        # Projection layer
        if projection_layer is None:
            audio_hidden = audio_embedding.embedding_dim
            gpt2_hidden = gpt2_adaptor.config.hidden_size
            if audio_hidden != gpt2_hidden:
                self.projection = nn.Linear(audio_hidden, gpt2_hidden)
                # 🔥 修复：将projection layer转为bfloat16，与Kimi保持一致
                self.projection = self.projection.to(torch.bfloat16)
            else:
                self.projection = nn.Identity()
        else:
            self.projection = projection_layer
        
        self.gumbel_tau = gumbel_tau
        self.audio_loss_weight = audio_loss_weight
        self.motion_loss_weight = motion_loss_weight
        
        # 冻结audio embedding
        for param in self.audio_embedding.parameters():
            param.requires_grad = False
        
        # 将audio_embedding也转为bfloat16
        self.audio_embedding = self.audio_embedding.to(torch.bfloat16)
    
    def forward_kimi_with_loss(
        self,
        user_audio_tokens,
        assistant_audio_tokens_gt,
        user_attention_mask=None,
    ):
        """
        Kimi forward并计算audio loss
        
        注意：这里需要修改以获取logits
        目前简化为直接用forward
        """
        # 方式1：直接用model.forward（简化版）
        # 实际需要修改Kimi支持teacher forcing
        
        outputs = self.kimi(
            input_ids=user_audio_tokens,
            attention_mask=user_attention_mask,
            output_hidden_states=False,
            return_dict=False,  # 🔥 改为False，返回tuple更容易处理
        )
        
        # 🔥 修复：MoonshotKimiaForCausalLM的outputs.logits是tuple
        # outputs是tuple时：(logits_text, logits_audio, past_key_values)
        # outputs是dict时：outputs.logits = (logits_text, logits_audio)
        
        if isinstance(outputs, tuple):
            # return_dict=False的情况
            # outputs = (logits_text, logits_audio, past_key_values, ...)
            # 取第一个3D tensor作为logits（通常是audio logits）
            logits = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[1]
        elif hasattr(outputs, 'logits'):
            # return_dict=True的情况
            logits_tuple = outputs.logits
            if isinstance(logits_tuple, tuple) and len(logits_tuple) >= 2:
                # logits是tuple: (text_logits, audio_logits)
                # 根据vocab_size判断：audio logits的vocab通常更大
                text_logits = logits_tuple[0]  # [batch, seq, vocab_text]
                audio_logits = logits_tuple[1]  # [batch, seq, vocab_audio]
                
                # 使用audio_logits（vocab更大的那个，或第二个）
                logits = audio_logits
                print(f"🔍 Logits tuple: text_logits={text_logits.shape}, audio_logits={audio_logits.shape}")
            else:
                logits = logits_tuple
        else:
            raise ValueError(f"Cannot extract logits from outputs: {type(outputs)}")
        
        # 确保logits是tensor
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"Logits is not a tensor: {type(logits)}")
        
        if logits.dim() != 3:
            raise ValueError(f"Logits should be 3D [batch, seq, vocab], got shape: {logits.shape}")
        
        print(f"✅ Got audio logits: shape={logits.shape}")
        
        # 计算audio loss（对比assistant GT）
        # 简化：取最后N个token的logits对比
        seq_len = min(logits.shape[1], assistant_audio_tokens_gt.shape[1])
        pred_logits = logits[:, -seq_len:, :]
        gt_tokens = assistant_audio_tokens_gt[:, :seq_len]
        
        audio_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            gt_tokens.reshape(-1),
            ignore_index=0,  # padding
        )
        
        # 返回assistant部分的logits（用于后续的Gumbel-Softmax）
        assistant_logits = logits[:, -seq_len:, :]
        
        return assistant_logits, audio_loss
    
    def forward(
        self,
        user_audio_tokens,
        assistant_audio_tokens_gt,
        motion_tokens_gt,
        user_attention_mask=None,
        use_gumbel_hard=False,
        interleave_ratio=(1, 1),
    ):
        """
        完整的forward（计算audio loss和motion loss）
        
        Args:
            user_audio_tokens: [batch, user_seq_len] Kimi输入
            assistant_audio_tokens_gt: [batch, assistant_seq_len] Kimi输出GT
            motion_tokens_gt: [batch, motion_seq_len] Adaptor输出GT
            
        Returns:
            {
                'total_loss': audio_loss + motion_loss,
                'audio_loss': Kimi的loss,
                'motion_loss': Adaptor的loss,
                'audio_logits': Kimi的输出logits,
                'motion_logits': Adaptor的输出logits,
            }
        """
        device = user_audio_tokens.device
        batch_size = user_audio_tokens.shape[0]
        
        # 1️⃣ Kimi forward + audio loss
        assistant_audio_logits, audio_loss = self.forward_kimi_with_loss(
            user_audio_tokens,
            assistant_audio_tokens_gt,
            user_attention_mask,
        )
        
        # [batch, assistant_seq_len, audio_vocab_size]
        
        # 2️⃣ Gumbel-Softmax
        soft_audio_tokens = gumbel_softmax(
            assistant_audio_logits,
            tau=self.gumbel_tau,
            hard=use_gumbel_hard,
        )
        # [batch, assistant_seq_len, audio_vocab_size]
        
        # 3️⃣ 转为embeddings
        soft_audio_embeds = torch.matmul(
            soft_audio_tokens,
            self.audio_embedding.weight
        )
        # [batch, assistant_seq_len, audio_hidden_dim]
        
        # 4️⃣ Projection
        projected_embeds = self.projection(soft_audio_embeds)
        # [batch, assistant_seq_len, gpt2_hidden_dim]
        
        # 🔥 转换为float32用于GPT2（GPT2通常用float32训练）
        projected_embeds = projected_embeds.to(torch.float32)
        
        # 5️⃣ 构建interleaved sequence（audio + motion）
        inputs_embeds, labels = self.build_interleaved_sequence(
            projected_embeds,
            motion_tokens_gt,
            interleave_ratio,
        )
        
        # 6️⃣ GPT2 Adaptor forward
        transformer_outputs = self.adaptor.transformer(inputs_embeds=inputs_embeds)
        hidden_states = transformer_outputs[0]
        motion_logits = self.adaptor.lm_head(hidden_states)
        
        # 7️⃣ 计算motion loss
        shift_logits = motion_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        motion_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # 8️⃣ 总loss（加权）
        total_loss = (
            self.audio_loss_weight * audio_loss +
            self.motion_loss_weight * motion_loss
        )
        
        return {
            'total_loss': total_loss,
            'audio_loss': audio_loss,
            'motion_loss': motion_loss,
            'audio_logits': assistant_audio_logits,
            'motion_logits': motion_logits,
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


# ==================== Training ====================

def create_optimizer(model, kimi_lr=5e-5, adaptor_lr=1e-4, projection_lr=1e-4):
    """创建使用不同学习率的优化器"""
    
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
    
    optimizer = torch.optim.AdamW([
        {'params': kimi_params, 'lr': kimi_lr, 'name': 'kimi_lora'},
        {'params': projection_params, 'lr': projection_lr, 'name': 'projection'},
        {'params': adaptor_params, 'lr': adaptor_lr, 'name': 'adaptor'},
    ], weight_decay=0.01)
    
    print(f"\n📊 Optimizer groups:")
    print(f"   Kimi LoRA: {sum(p.numel() for p in kimi_params):,} params, lr={kimi_lr}")
    print(f"   Projection: {sum(p.numel() for p in projection_params):,} params, lr={projection_lr}")
    print(f"   Adaptor: {sum(p.numel() for p in adaptor_params):,} params, lr={adaptor_lr}")
    
    return optimizer


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, tau_scheduler):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_audio_loss = 0
    total_motion_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        user_audio = batch['user_audio_tokens'].to(device)
        assistant_audio = batch['assistant_audio_tokens'].to(device)
        motion = batch['motion_tokens'].to(device)
        user_mask = batch['user_attention_mask'].to(device)
        
        # 更新Gumbel温度
        current_tau = tau_scheduler(epoch * len(dataloader) + batch_idx)
        model.gumbel_tau = current_tau
        
        # Forward
        outputs = model(
            user_audio_tokens=user_audio,
            assistant_audio_tokens_gt=assistant_audio,
            motion_tokens_gt=motion,
            user_attention_mask=user_mask,
            use_gumbel_hard=(epoch > 10),  # 后期使用hard模式
        )
        
        loss = outputs['total_loss']
        audio_loss = outputs['audio_loss']
        motion_loss = outputs['motion_loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        total_audio_loss += audio_loss.item()
        total_motion_loss += motion_loss.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'audio': f"{audio_loss.item():.4f}",
            'motion': f"{motion_loss.item():.4f}",
            'tau': f"{current_tau:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_audio_loss = total_audio_loss / len(dataloader)
    avg_motion_loss = total_motion_loss / len(dataloader)
    
    return avg_loss, avg_audio_loss, avg_motion_loss


def main():
    """主训练函数"""
    
    print("\n" + "🔥"*30)
    print("JSONL数据训练：Kimi LoRA + Adaptor Full")
    print("🔥"*30 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 配置 ====================
    config = {
        # 数据
        'train_data_file': '/root/workspace/HRI_MLLM/data/HAND_RING_1111_tokens.jsonl',
        'max_audio_len': 512,
        'max_motion_len': 512,
        
        # 模型路径
        'kimi_model_path': '/DATA/disk1/Kimi-Audio-7B-Instruct',
        'gpt2_checkpoint': '/root/workspace/HRI_MLLM/output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1700.pt',
        'audio_embedding_weight': f'{OUTPUT_ROOT}/motion_adaptor_v1/embed_tokens_weight.pt',
        
        # LoRA配置
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        
        # 训练超参数
        'batch_size': 2,
        'num_epochs': 30,
        'kimi_lr': 5e-5,
        'adaptor_lr': 1e-4,
        'projection_lr': 1e-4,
        'warmup_ratio': 0.1,
        
        # Loss权重（🔥 改为1:1）
        'audio_loss_weight': 1.0,    # audio loss权重
        'motion_loss_weight': 1.0,   # motion loss权重
        
        # Gumbel-Softmax
        'initial_tau': 1.0,
        'min_tau': 0.5,
        'tau_decay': 0.9999,
        
        # 输出
        'output_dir': '/root/workspace/HRI_MLLM/output/jsonl_kimi_lora_adaptor',
        'save_every': 5,
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # ==================== 加载数据 ====================
    
    print("📦 Loading dataset...")
    train_dataset = JSONLAudioMotionDataset(
        config['train_data_file'],
        max_audio_len=config['max_audio_len'],
        max_motion_len=config['max_motion_len'],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    # ==================== 加载模型 ====================
    
    # 1. Kimi + LoRA
    print("\n📦 Loading Kimi model...")
    kimi_audio = KimiAudio(
        model_path=config['kimi_model_path'],
        load_detokenizer=False,
    )
    
    # 🔥 Monkey patch: 添加prepare_inputs_for_generation方法（peft要求）
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            # 如果有past_key_values，只需要最后一个token
            input_ids = input_ids[:, -1:]
        
        # 如果传入了inputs_embeds，优先使用
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
    
    # 绑定方法到模型
    import types
    kimi_audio.alm.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation, kimi_audio.alm
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # 🔥 修复：KimiAudio的模型属性是alm，不是model
    kimi_lora = get_peft_model(kimi_audio.alm, lora_config)
    
    trainable = sum(p.numel() for p in kimi_lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in kimi_lora.parameters())
    print(f"✅ Kimi LoRA: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # 2. GPT2 Adaptor
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
    
    # 全参数可训练
    for param in gpt2_adaptor.parameters():
        param.requires_grad = True
    
    adaptor_trainable = sum(p.numel() for p in gpt2_adaptor.parameters() if p.requires_grad)
    print(f"✅ Adaptor: {adaptor_trainable:,} params (全参数)")
    
    # 3. Audio embedding
    print("\n📦 Loading audio embedding...")
    audio_emb_weight = torch.load(config['audio_embedding_weight'], weights_only=True)
    audio_embedding = nn.Embedding.from_pretrained(audio_emb_weight, freeze=True)
    
    # 4. 创建端到端模型
    print("\n🔧 Creating end-to-end model...")
    model = KimiMotionModel(
        kimi_model=kimi_lora,
        gpt2_adaptor=gpt2_adaptor,
        audio_embedding=audio_embedding,
        gumbel_tau=config['initial_tau'],
        audio_loss_weight=config['audio_loss_weight'],
        motion_loss_weight=config['motion_loss_weight'],
    )
    model = model.to(device)
    
    # ==================== 优化器 ====================
    
    optimizer = create_optimizer(
        model,
        kimi_lr=config['kimi_lr'],
        adaptor_lr=config['adaptor_lr'],
        projection_lr=config['projection_lr'],
    )
    
    # 学习率调度器
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # 温度调度器
    tau_scheduler = lambda step: max(
        config['min_tau'],
        config['initial_tau'] * (config['tau_decay'] ** step)
    )
    
    # ==================== 训练 ====================
    
    print("\n" + "="*60)
    print("🚀 开始训练")
    print("="*60)
    
    print(f"\n配置:")
    print(f"  Dataset: {len(train_dataset)} samples")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Audio loss weight: {config['audio_loss_weight']}")
    print(f"  Motion loss weight: {config['motion_loss_weight']}")
    
    best_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        avg_loss, avg_audio_loss, avg_motion_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, tau_scheduler
        )
        
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Total Loss: {avg_loss:.4f}")
        print(f"   Audio Loss: {avg_audio_loss:.4f}")
        print(f"   Motion Loss: {avg_motion_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(config['output_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, save_path)
            print(f"✅ Saved best model (loss={avg_loss:.4f})")
        
        # Regular save
        if epoch % config['save_every'] == 0:
            save_path = os.path.join(config['output_dir'], f'epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, save_path)
            print(f"💾 Saved checkpoint at epoch {epoch}")
    
    print(f"\n🎉 训练完成!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   输出目录: {config['output_dir']}")


if __name__ == "__main__":
    main()

