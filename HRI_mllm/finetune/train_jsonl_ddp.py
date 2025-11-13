"""
8卡分布式训练脚本：Kimi LoRA + Adaptor Full
使用DDP (DistributedDataParallel)

运行方式：
    torchrun --nproc_per_node=8 train_jsonl_ddp.py
    或
    bash scripts/train_8gpu.sh
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
from transformers import GPT2Config, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import logging
from datetime import datetime
import types

# Kimi
from kimia_infer.api.kimia import KimiAudio

# Adaptor
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from HRI_mllm import OUTPUT_ROOT


# ==================== Logging Setup ====================

def setup_logging(output_dir, rank):
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # 只在rank 0上输出到文件和console
    if rank == 0:
        # 文件handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'train_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"日志保存到: {log_file}")
    
    return logger


# ==================== DDP Setup ====================

def setup_ddp():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("⚠️  未检测到分布式环境，使用单GPU模式")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


# ==================== Dataset ====================

class JSONLAudioMotionDataset(Dataset):
    """JSONL数据集（支持多文件混合训练）"""
    
    def __init__(self, jsonl_files, text_tokenizer=None, max_audio_len=512, max_motion_len=512, max_text_len=128, 
                 crop_configs=None, kimia_text_blank=18):
        """
        Args:
            jsonl_files: str or list of str, JSONL文件路径
            text_tokenizer: text tokenizer
            max_audio_len: 最大音频token长度
            max_motion_len: 最大动作token长度
            max_text_len: 最大文本token长度
            crop_configs: dict, 每个文件是否需要裁剪的配置，例如:
                         {'beat': True, 'hand_ring': False}
                         如果为None，则所有文件都不裁剪
            kimia_text_blank: kimia_text_blank token ID（用于填充）
        """
        # 支持单个文件或多个文件
        if isinstance(jsonl_files, str):
            jsonl_files = [jsonl_files]
        
        self.jsonl_files = jsonl_files
        self.text_tokenizer = text_tokenizer
        self.max_audio_len = max_audio_len
        self.max_motion_len = max_motion_len
        self.max_text_len = max_text_len
        self.crop_configs = crop_configs or {}
        self.kimia_text_blank = kimia_text_blank  # 🔥 用于填充 text_input_ids
        
        # 加载数据，并记录来源
        self.data = []
        for jsonl_file in jsonl_files:
            # 判断数据来源（基于文件名）
            # 注意：顺序很重要！先检查更具体的模式，再检查通用模式
            filename = jsonl_file.lower()
            if 'fist-beat' in filename or 'fist_beat' in filename:
                data_source = 'fist'
            elif 'thumb_forefinger_and_little_finger' in filename or 'thumb_forefinger' in filename:
                data_source = 'thumb_forefinger_and_little_finger'
            elif 'forefinger' in filename:
                data_source = 'forefinger'
            elif 'beat_v2' in filename or 'beat' in filename:
                data_source = 'beat'
            elif 'hand_call' in filename:
                data_source = 'hand_call'
            elif 'hand_ring' in filename:
                data_source = 'hand_ring'
            elif 'hand_v' in filename:
                data_source = 'hand_v'
            elif 'palm' in filename:
                data_source = 'palm'
            elif 'thumb_up' in filename:
                data_source = 'thumb_up'
            else:
                data_source = 'unknown'
            
            # 读取文件
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item['_data_source'] = data_source  # 添加来源标记
                        self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item['conversation']
        data_source = item.get('_data_source', 'unknown')
        
        # 🔥 提取数据（包括text prompt）
        user_text_prompt = None
        user_audio_tokens = None
        assistant_audio_tokens = None
        motion_tokens = None
        
        for msg in conversation:
            if msg['role'] == 'user' and msg.get('message_type') == 'text':
                # 🔥 提取text prompt（重要！）
                user_text_prompt = msg.get('content')
            elif msg['role'] == 'user' and msg.get('message_type') == 'audio':
                user_audio_tokens = msg['audio_tokens']
            elif msg['role'] == 'assistant' and msg.get('message_type') == 'audio_motion':
                assistant_audio_tokens = msg['audio_tokens']
                motion_tokens = msg['motion_tokens']
        
        if user_audio_tokens is None or assistant_audio_tokens is None or motion_tokens is None:
            return {
                'user_text_tokens': torch.zeros(1, dtype=torch.long),
                'user_audio_tokens': torch.zeros(10, dtype=torch.long),
                'assistant_audio_tokens': torch.zeros(10, dtype=torch.long),
                'motion_tokens': torch.zeros(10, dtype=torch.long),
            }
        
        # 🔥 根据数据来源决定是否裁剪
        # BEAT数据：序列很长（~1451 tokens），需要随机裁剪
        # HAND_RING数据：序列较短（~70-190 tokens），不需要裁剪
        should_crop = self.crop_configs.get(data_source, False)
        
        min_len = min(len(user_audio_tokens), len(assistant_audio_tokens), len(motion_tokens))
        
        if should_crop and min_len > self.max_audio_len:
            # 🔥 BEAT数据：随机裁剪，确保audio和motion token序列时间对齐
            # 随机选择起始位置
            start_idx = np.random.randint(0, min_len - self.max_audio_len + 1)
            end_idx = start_idx + self.max_audio_len
            
            # 使用相同的区间裁剪所有序列，保证时间对齐
            user_audio_tokens = user_audio_tokens[start_idx:end_idx]
            assistant_audio_tokens = assistant_audio_tokens[start_idx:end_idx]
            motion_tokens = motion_tokens[start_idx:min(end_idx, len(motion_tokens))]
        else:
            # 🔥 HAND_RING数据或不需要裁剪：直接截断（通常不会超过max_len）
            user_audio_tokens = user_audio_tokens[:self.max_audio_len]
            assistant_audio_tokens = assistant_audio_tokens[:self.max_audio_len]
            motion_tokens = motion_tokens[:self.max_motion_len]
        
        # 🔥 构建与prompt_manager完全一致的序列
        # 参考：kimia_infer/api/prompt_manager.py 第78-128行
        # 
        # Conversation结构：
        #   1. user text message: "Please repeat..."
        #   2. user audio message: audio_tokens
        # 
        # 构建后的序列（与prompt_manager一致）：
        #   audio_ids = [blank]*len(text_tokens) + audio_tokens
        #   text_ids  = text_tokens              + [blank]*len(audio_tokens)
        # 
        # 长度必须相同！
        
        # 1. Tokenize text prompt
        text_prompt_tokens = []
        if self.text_tokenizer is not None and user_text_prompt:
            try:
                text_prompt_tokens = self.text_tokenizer.encode(user_text_prompt, bos=False, eos=False)
                # 截断到合理长度（避免太长）
                if len(text_prompt_tokens) > 128:
                    text_prompt_tokens = text_prompt_tokens[:128]
            except Exception as e:
                print(f"Warning: Text tokenization failed: {e}")
                text_prompt_tokens = []
        
        # 2. 构建对齐的序列（模拟prompt_manager的行为）
        text_len = len(text_prompt_tokens)
        audio_len = len(user_audio_tokens)
        
        # 构建user_audio_input_ids: [blank]*text_len + audio_tokens
        audio_input_ids = [self.kimia_text_blank] * text_len + user_audio_tokens
        
        # 构建user_text_input_ids: text_tokens + [blank]*audio_len
        text_input_ids = text_prompt_tokens + [self.kimia_text_blank] * audio_len
        
        # 验证长度一致
        assert len(audio_input_ids) == len(text_input_ids), \
            f"audio_ids length {len(audio_input_ids)} != text_ids length {len(text_input_ids)}"
        
        # 🔥 返回数据（与prompt_manager对齐）
        return {
            'user_audio_tokens': torch.tensor(audio_input_ids, dtype=torch.long),      # [blank]*text_len + audio
            'user_text_tokens': torch.tensor(text_input_ids, dtype=torch.long),        # text + [blank]*audio_len
            'assistant_audio_tokens': torch.tensor(assistant_audio_tokens, dtype=torch.long),
            'motion_tokens': torch.tensor(motion_tokens, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function"""
    # 🔥 重要：user_audio_tokens 和 user_text_tokens 必须长度相同（已在Dataset中保证）
    max_user_len = max(item['user_audio_tokens'].shape[0] for item in batch)
    max_assistant_len = max(item['assistant_audio_tokens'].shape[0] for item in batch)
    max_motion_len = max(item['motion_tokens'].shape[0] for item in batch)
    
    # 验证：audio和text长度应该相同
    # （因为在Dataset中已经保证 len(audio_ids) == len(text_ids)）
    for item in batch:
        assert item['user_audio_tokens'].shape[0] == item['user_text_tokens'].shape[0], \
            f"Audio length {item['user_audio_tokens'].shape[0]} != Text length {item['user_text_tokens'].shape[0]}"
    
    user_text_tokens_list = []
    user_audio_tokens_list = []
    assistant_audio_tokens_list = []
    motion_tokens_list = []
    user_attention_masks = []
    assistant_attention_masks = []
    text_attention_masks = []
    
    for item in batch:
        # 🔥 audio 和 text 使用相同的长度和padding
        user_tokens = item['user_audio_tokens']
        text_tokens = item['user_text_tokens']
        user_pad_len = max_user_len - len(user_tokens)
        
        # Padding audio tokens
        user_audio_tokens_list.append(
            torch.cat([user_tokens, torch.zeros(user_pad_len, dtype=torch.long)])
        )
        
        # Padding text tokens（使用相同的pad长度）
        user_text_tokens_list.append(
            torch.cat([text_tokens, torch.zeros(user_pad_len, dtype=torch.long)])
        )
        
        # Attention masks（audio和text使用相同的mask）
        user_attention_masks.append(
            torch.cat([torch.ones(len(user_tokens)), torch.zeros(user_pad_len)])
        )
        text_attention_masks.append(
            torch.cat([torch.ones(len(text_tokens)), torch.zeros(user_pad_len)])
        )
        
        assistant_tokens = item['assistant_audio_tokens']
        assistant_pad_len = max_assistant_len - len(assistant_tokens)
        assistant_audio_tokens_list.append(
            torch.cat([assistant_tokens, torch.zeros(assistant_pad_len, dtype=torch.long)])
        )
        assistant_attention_masks.append(
            torch.cat([torch.ones(len(assistant_tokens)), torch.zeros(assistant_pad_len)])
        )
        
        motion_tokens = item['motion_tokens']
        motion_pad_len = max_motion_len - len(motion_tokens)
        motion_tokens_list.append(
            torch.cat([motion_tokens, torch.zeros(motion_pad_len, dtype=torch.long)])
        )
    
    return {
        'user_text_tokens': torch.stack(user_text_tokens_list),  # 🔥 Text tokens
        'text_attention_mask': torch.stack(text_attention_masks),  # 🔥 Text mask
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
    """端到端Kimi + Adaptor模型（与单卡版本相同，但添加了dtype处理）"""
    
    def __init__(
        self,
        kimi_model,
        gpt2_adaptor,
        audio_embedding,
        projection_layer=None,
        gumbel_tau=1.0,
        audio_loss_weight=1.0,  # 🔥 改为1.0
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
        
        self.audio_embedding = self.audio_embedding.to(torch.bfloat16)
    
    def forward_kimi_with_loss(
        self,
        user_audio_tokens,
        user_text_tokens=None,  # 🔥 新增：text tokens
        assistant_audio_tokens_gt=None,
        user_attention_mask=None,
        text_attention_mask=None,  # 🔥 新增：text attention mask
    ):
        """Kimi forward并计算audio loss"""
        outputs = self.kimi(
            input_ids=user_audio_tokens,
            text_input_ids=user_text_tokens,  # 🔥 传入text tokens
            attention_mask=user_attention_mask,
            output_hidden_states=False,
            return_dict=False,
        )
        
        # 提取logits
        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif hasattr(outputs, 'logits'):
            logits_tuple = outputs.logits
            if isinstance(logits_tuple, tuple) and len(logits_tuple) >= 2:
                logits = logits_tuple[1]  # audio_logits
            else:
                logits = logits_tuple
        else:
            raise ValueError(f"Cannot extract logits from outputs: {type(outputs)}")
        
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # 计算audio loss
        seq_len = min(logits.shape[1], assistant_audio_tokens_gt.shape[1])
        pred_logits = logits[:, -seq_len:, :]
        gt_tokens = assistant_audio_tokens_gt[:, :seq_len]
        
        audio_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            gt_tokens.reshape(-1),
            ignore_index=0,
        )
        
        assistant_logits = logits[:, -seq_len:, :]
        
        return assistant_logits, audio_loss
    
    def forward(
        self,
        user_audio_tokens,
        user_text_tokens=None,  # 🔥 新增
        assistant_audio_tokens_gt=None,
        motion_tokens_gt=None,
        user_attention_mask=None,
        text_attention_mask=None,  # 🔥 新增
        use_gumbel_hard=False,
        interleave_ratio=(1, 1),
    ):
        """完整forward"""
        device = user_audio_tokens.device
        
        # 1. Kimi + audio loss
        assistant_audio_logits, audio_loss = self.forward_kimi_with_loss(
            user_audio_tokens,
            user_text_tokens=user_text_tokens,  # 🔥 传入
            assistant_audio_tokens_gt=assistant_audio_tokens_gt,
            user_attention_mask=user_attention_mask,
            text_attention_mask=text_attention_mask,  # 🔥 传入
        )
        
        # 2. Gumbel-Softmax
        soft_audio_tokens = gumbel_softmax(
            assistant_audio_logits,
            tau=self.gumbel_tau,
            hard=use_gumbel_hard,
        )
        
        # 3. 转为embeddings
        soft_audio_embeds = torch.matmul(
            soft_audio_tokens,
            self.audio_embedding.weight
        )
        
        # 4. Projection + dtype转换
        projected_embeds = self.projection(soft_audio_embeds)
        projected_embeds = projected_embeds.to(torch.float32)
        
        # 5. 构建interleaved sequence
        inputs_embeds, labels = self.build_interleaved_sequence(
            projected_embeds,
            motion_tokens_gt,
            interleave_ratio,
        )
        
        # 6. GPT2 Adaptor
        transformer_outputs = self.adaptor.transformer(inputs_embeds=inputs_embeds)
        hidden_states = transformer_outputs[0]
        motion_logits = self.adaptor.lm_head(hidden_states)
        
        # 7. Motion loss
        shift_logits = motion_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        motion_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # 8. Total loss
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
    
    def build_interleaved_sequence(self, audio_embeds, motion_tokens, interleave_ratio=(1, 1)):
        """构建交错序列"""
        batch_size = audio_embeds.shape[0]
        audio_seq_len = audio_embeds.shape[1]
        motion_seq_len = motion_tokens.shape[1]
        hidden_dim = audio_embeds.shape[2]
        device = audio_embeds.device
        
        num_audio, num_motion = interleave_ratio
        max_groups = min(audio_seq_len // num_audio, motion_seq_len // num_motion)
        total_len = max_groups * (num_audio + num_motion)
        
        inputs_embeds = torch.zeros(
            batch_size, total_len, hidden_dim,
            dtype=audio_embeds.dtype, device=device
        )
        labels = torch.full(
            (batch_size, total_len),
            -100, dtype=torch.long, device=device
        )
        
        motion_embeds = self.adaptor.transformer.wte(motion_tokens)
        
        for i in range(max_groups):
            start_pos = i * (num_audio + num_motion)
            audio_start = i * num_audio
            audio_end = audio_start + num_audio
            
            inputs_embeds[:, start_pos:start_pos+num_audio, :] = \
                audio_embeds[:, audio_start:audio_end, :]
            
            motion_start = i * num_motion
            motion_end = motion_start + num_motion
            motion_pos = start_pos + num_audio
            
            inputs_embeds[:, motion_pos:motion_pos+num_motion, :] = \
                motion_embeds[:, motion_start:motion_end, :]
            labels[:, motion_pos:motion_pos+num_motion] = \
                motion_tokens[:, motion_start:motion_end]
        
        return inputs_embeds, labels


# ==================== Training ====================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, tau_scheduler, logger, rank):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_audio_loss = 0
    total_motion_loss = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        user_text = batch['user_text_tokens'].to(device)  # 🔥 新增
        text_mask = batch['text_attention_mask'].to(device)  # 🔥 新增
        user_audio = batch['user_audio_tokens'].to(device)
        assistant_audio = batch['assistant_audio_tokens'].to(device)
        motion = batch['motion_tokens'].to(device)
        user_mask = batch['user_attention_mask'].to(device)
        
        # 更新温度
        global_step = (epoch - 1) * len(dataloader) + batch_idx
        current_tau = tau_scheduler(global_step)
        model.module.gumbel_tau = current_tau  # DDP模型需要用.module访问
        
        # Forward
        outputs = model(
            user_audio_tokens=user_audio,
            user_text_tokens=user_text,  # 🔥 传入text tokens
            assistant_audio_tokens_gt=assistant_audio,
            motion_tokens_gt=motion,
            user_attention_mask=user_mask,
            text_attention_mask=text_mask,  # 🔥 传入text mask
            use_gumbel_hard=(epoch > 10),
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
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'audio': f"{audio_loss.item():.4f}",
                'motion': f"{motion_loss.item():.4f}",
                'tau': f"{current_tau:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            # 每100步详细记录
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch} Step {batch_idx}/{len(dataloader)}: "
                    f"loss={loss.item():.4f}, audio={audio_loss.item():.4f}, "
                    f"motion={motion_loss.item():.4f}, tau={current_tau:.4f}"
                )
    
    avg_loss = total_loss / len(dataloader)
    avg_audio_loss = total_audio_loss / len(dataloader)
    avg_motion_loss = total_motion_loss / len(dataloader)
    
    return avg_loss, avg_audio_loss, avg_motion_loss


def main():
    """主训练函数"""
    
    # 初始化DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # 配置
    config = {
        # 数据（🔥 混合训练：BEAT + 8种手势数据集）
        'train_data_files': [
            '/root/workspace/HRI_MLLM/data/BEAT_v2_1110_tokens.jsonl',
            # 8种手势数据集 (1112版本，总计约7710样本)
            '/root/workspace/HRI_MLLM/data/FIST-BEAT_1112_tokens.jsonl',                                    # 994
            '/root/workspace/HRI_MLLM/data/FOREFINGER_RAISE_ONE-2_1112_tokens.jsonl',                      # 982
            '/root/workspace/HRI_MLLM/data/HAND_CALL_1112_tokens.jsonl',                                   # 806
            '/root/workspace/HRI_MLLM/data/HAND_RING_1112_tokens.jsonl',                                   # 980
            '/root/workspace/HRI_MLLM/data/HAND_V_SIGN_1112_tokens.jsonl',                                 # 988
            '/root/workspace/HRI_MLLM/data/PALM_HALT_1112_tokens.jsonl',                                   # 984
            '/root/workspace/HRI_MLLM/data/THUMB_FOREFINGER_AND_LITTLE_FINGER_RAISE_1112_tokens.jsonl',   # 994
            '/root/workspace/HRI_MLLM/data/THUMB_UP_1112_tokens.jsonl',                                    # 982
        ],
        # 裁剪配置：只对BEAT数据进行随机裁剪，手势数据都不裁剪
        'crop_configs': {
            'beat': True,                                       # BEAT数据需要裁剪（序列太长）
            'fist': False,                                      # 手势数据不需要裁剪
            'forefinger': False,                                # 手势数据不需要裁剪
            'hand_call': False,                                 # 手势数据不需要裁剪
            'hand_ring': False,                                 # 手势数据不需要裁剪
            'hand_v': False,                                    # 手势数据不需要裁剪
            'palm': False,                                      # 手势数据不需要裁剪
            'thumb_forefinger_and_little_finger': False,       # 手势数据不需要裁剪
            'thumb_up': False,                                  # 手势数据不需要裁剪
        },
        'max_audio_len': 200,  # 🔥 BEAT裁剪到200，手势数据直接使用原长度
        'max_motion_len': 200,  # 🔥 保证时间对齐
        
        # 模型路径
        'kimi_model_path': '/DATA/disk1/Kimi-Audio-7B-Instruct',
        'gpt2_checkpoint': '/root/workspace/HRI_MLLM/output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1700.pt',
        'audio_embedding_weight': f'{OUTPUT_ROOT}/motion_adaptor_v1/embed_tokens_weight.pt',
        
        # LoRA配置
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        
        # 训练超参数
        'batch_size': 2,  # 每个GPU的batch size
        'num_epochs': 3000,
        'kimi_lr': 5e-5,
        'adaptor_lr': 1e-4,
        'projection_lr': 1e-4,
        'warmup_ratio': 0.1,
        
        # Loss权重（🔥 改为1:1）
        'audio_loss_weight': 1.0,
        'motion_loss_weight': 1.0,
        
        # Gumbel-Softmax
        'initial_tau': 1.0,
        'min_tau': 0.5,
        'tau_decay': 0.9999,
        
        # 输出
        'output_dir': '/root/workspace/HRI_MLLM/output_disk0/ddp_kimi_lora_adaptor_beat_gestures',
        'save_every': 10,  # 🔥 每10个epoch保存一次
    }
    
    # 只在rank 0创建输出目录
    if rank == 0:
        os.makedirs(config['output_dir'], exist_ok=True)
    
    # 设置日志
    logger = setup_logging(config['output_dir'], rank)
    
    if rank == 0:
        logger.info("="*60)
        logger.info("🔥 8卡分布式训练：Kimi LoRA + Adaptor Full (混合数据集)")
        logger.info("="*60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Batch size per GPU: {config['batch_size']}")
        logger.info(f"Effective batch size: {config['batch_size'] * world_size}")
        logger.info(f"Training files: {len(config['train_data_files'])} files")
        for f in config['train_data_files']:
            logger.info(f"  - {f}")
        logger.info(f"Crop configs: {config['crop_configs']}")
        logger.info(f"Audio loss weight: {config['audio_loss_weight']}")
        logger.info(f"Motion loss weight: {config['motion_loss_weight']}")
    
    # ==================== 加载Text Tokenizer ====================
    
    if rank == 0:
        logger.info("\n📦 Loading text tokenizer...")
    
    # 🔥 加载Kimi的text tokenizer（用于tokenize text prompt）
    from transformers import AutoTokenizer
    from kimia_infer.utils.special_tokens import instantiate_extra_tokens
    
    text_tokenizer = AutoTokenizer.from_pretrained(
        config['kimi_model_path'],
        trust_remote_code=True
    )
    
    # 🔥 获取 extra_tokens（包括 kimia_text_blank）
    extra_tokens = instantiate_extra_tokens(text_tokenizer)
    kimia_text_blank = extra_tokens.kimia_text_blank
    
    if rank == 0:
        logger.info(f"✅ Text tokenizer loaded")
        logger.info(f"   kimia_text_blank: {kimia_text_blank}")
    
    # ==================== 加载数据 ====================
    
    if rank == 0:
        logger.info("\n📦 Loading dataset...")
    
    train_dataset = JSONLAudioMotionDataset(
        jsonl_files=config['train_data_files'],  # 🔥 传入多个文件
        text_tokenizer=text_tokenizer,  # 🔥 传入text tokenizer
        max_audio_len=config['max_audio_len'],
        max_motion_len=config['max_motion_len'],
        max_text_len=128,  # text prompt的最大长度
        crop_configs=config['crop_configs'],  # 🔥 传入裁剪配置
        kimia_text_blank=kimia_text_blank,  # 🔥 传入kimia_text_blank用于填充
    )
    
    # 🔥 DDP: 使用DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,  # 使用sampler而不是shuffle
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    if rank == 0:
        logger.info(f"✅ Dataset: {len(train_dataset)} samples")
        logger.info(f"   Steps per epoch: {len(train_loader)}")
        logger.info(f"   Effective batch size: {config['batch_size'] * world_size}")
    
    # ==================== 加载模型 ====================
    
    if rank == 0:
        logger.info("\n📦 Loading models...")
    
    # 1. Kimi + LoRA
    kimi_audio = KimiAudio(
        model_path=config['kimi_model_path'],
        load_detokenizer=False,
    )
    
    # 添加prepare_inputs_for_generation
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
    
    kimi_lora = get_peft_model(kimi_audio.alm, lora_config)
    
    if rank == 0:
        trainable = sum(p.numel() for p in kimi_lora.parameters() if p.requires_grad)
        total = sum(p.numel() for p in kimi_lora.parameters())
        logger.info(f"✅ Kimi LoRA: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # 2. GPT2 Adaptor
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
    
    for param in gpt2_adaptor.parameters():
        param.requires_grad = True
    
    if rank == 0:
        adaptor_trainable = sum(p.numel() for p in gpt2_adaptor.parameters() if p.requires_grad)
        logger.info(f"✅ Adaptor: {adaptor_trainable:,} params (全参数)")
    
    # 3. Audio embedding
    audio_emb_weight = torch.load(config['audio_embedding_weight'], weights_only=True)
    audio_embedding = nn.Embedding.from_pretrained(audio_emb_weight, freeze=True)
    
    # 4. 创建模型
    model = KimiMotionModel(
        kimi_model=kimi_lora,
        gpt2_adaptor=gpt2_adaptor,
        audio_embedding=audio_embedding,
        gumbel_tau=config['initial_tau'],
        audio_loss_weight=config['audio_loss_weight'],
        motion_loss_weight=config['motion_loss_weight'],
    )
    
    # 🔥 DDP: 包装模型
    model = model.to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,  # 因为Kimi部分可能frozen
    )
    
    if rank == 0:
        logger.info("✅ Model wrapped with DDP")
    
    # ==================== 优化器 ====================
    
    # 参数分组
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
        {'params': kimi_params, 'lr': config['kimi_lr'], 'name': 'kimi_lora'},
        {'params': projection_params, 'lr': config['projection_lr'], 'name': 'projection'},
        {'params': adaptor_params, 'lr': config['adaptor_lr'], 'name': 'adaptor'},
    ], weight_decay=0.01)
    
    if rank == 0:
        logger.info(f"\n📊 Optimizer groups:")
        logger.info(f"   Kimi LoRA: {sum(p.numel() for p in kimi_params):,} params, lr={config['kimi_lr']}")
        logger.info(f"   Projection: {sum(p.numel() for p in projection_params):,} params, lr={config['projection_lr']}")
        logger.info(f"   Adaptor: {sum(p.numel() for p in adaptor_params):,} params, lr={config['adaptor_lr']}")
    
    # 学习率调度
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # 温度调度
    tau_scheduler = lambda step: max(
        config['min_tau'],
        config['initial_tau'] * (config['tau_decay'] ** step)
    )
    
    if rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 开始训练")
        logger.info(f"{'='*60}")
        logger.info(f"Total steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
    
    # ==================== 训练循环 ====================
    
    best_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        # 🔥 DDP: 设置epoch（用于正确shuffle）
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{config['num_epochs']}")
            logger.info(f"{'='*60}")
        
        # Train
        avg_loss, avg_audio_loss, avg_motion_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, tau_scheduler, logger, rank
        )
        
        # 只在rank 0保存和记录
        if rank == 0:
            logger.info(f"\n📊 Epoch {epoch} Summary:")
            logger.info(f"   Total Loss: {avg_loss:.4f}")
            logger.info(f"   Audio Loss: {avg_audio_loss:.4f}")
            logger.info(f"   Motion Loss: {avg_motion_loss:.4f}")
            
            # 保存best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(config['output_dir'], 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),  # DDP需要用.module
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'loss': avg_loss,
                    'audio_loss': avg_audio_loss,
                    'motion_loss': avg_motion_loss,
                    'config': config,
                }, save_path)
                logger.info(f"✅ Saved best model (loss={avg_loss:.4f})")
            
            # 定期保存
            if epoch % config['save_every'] == 0:
                save_path = os.path.join(config['output_dir'], f'epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'loss': avg_loss,
                    'audio_loss': avg_audio_loss,
                    'motion_loss': avg_motion_loss,
                    'config': config,
                }, save_path)
                logger.info(f"💾 Saved checkpoint at epoch {epoch}")
    
    if rank == 0:
        logger.info(f"\n🎉 训练完成!")
        logger.info(f"   Best loss: {best_loss:.4f}")
        logger.info(f"   输出目录: {config['output_dir']}")
    
    # 清理
    cleanup_ddp()


if __name__ == "__main__":
    main()

