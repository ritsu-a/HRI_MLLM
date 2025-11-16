#!/usr/bin/env python3
"""
测试train_jsonl_ddp.py训练的模型
支持从checkpoint加载模型并在JSONL数据上进行推理测试

使用示例：
    python HRI_mllm/test/test_trained_model_from_jsonl.py \
        --checkpoint output_disk0/ddp_kimi_lora_adaptor_beat_gestures/epoch_300.pt \
        --sample_idx 0 \
        --output_dir test_output/sample_0 \
        --temperature 1.0 \
        --top_k 50 \
        --max_motion_tokens 512
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT
from kimia_infer.api.kimia import KimiAudio
from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from transformers import GPT2Config
from peft import LoraConfig, get_peft_model, TaskType
import shutil
import yaml
import soundfile as sf

# VQ-VAE 相关
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import feats2datapkl, load_normalization_stats
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion


# ==================== Model ====================

class KimiMotionModel(nn.Module):
    """端到端Kimi + Adaptor模型"""
    
    def __init__(
        self,
        kimi_model,
        gpt2_adaptor,
        audio_embedding,
        kimi_audio_obj=None,  # 保存完整的 KimiAudio 对象用于推理
        projection_layer=None,
        gumbel_tau=1.0,
        kimia_text_blank=18,  # kimia_text_blank token ID
    ):
        super().__init__()
        
        self.kimi = kimi_model
        self.adaptor = gpt2_adaptor
        self.audio_embedding = audio_embedding
        self.kimi_audio_obj = kimi_audio_obj  # 用于推理
        self.kimia_text_blank = kimia_text_blank
        
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
        
        # 冻结audio embedding
        for param in self.audio_embedding.parameters():
            param.requires_grad = False
        
        self.audio_embedding = self.audio_embedding.to(torch.bfloat16)
    
    @torch.no_grad()
    def generate_motion_from_audio_tokens(
        self,
        audio_tokens,
        max_motion_tokens=512,
        temperature=1.0,
        top_k=50,
        interleave_ratio=(1, 1),
    ):
        """从audio tokens生成motion tokens"""
        self.adaptor.eval()
        device = audio_tokens.device
        batch_size = audio_tokens.shape[0]
        
        # 确保 audio_tokens 是 long 类型
        audio_tokens = audio_tokens.long()
        
        # 检查并限制 audio_tokens 范围
        audio_vocab_size = self.audio_embedding.num_embeddings
        print(f"\n[Debug] Audio tokens stats:")
        print(f"  audio_tokens shape: {audio_tokens.shape}")
        print(f"  audio_tokens min: {audio_tokens.min().item()}, max: {audio_tokens.max().item()}")
        print(f"  audio_embedding vocab size: {audio_vocab_size}")
        
        # 限制在有效范围内
        audio_tokens = torch.clamp(audio_tokens, 0, audio_vocab_size - 1)
        
        # 1. Audio tokens -> embeddings
        audio_embeds = self.audio_embedding(audio_tokens)
        audio_embeds = self.projection(audio_embeds).to(torch.float32)
        
        # 2. 构建interleaved sequence的初始部分
        audio_ratio, motion_ratio = interleave_ratio
        audio_chunk_size = audio_ratio
        motion_chunk_size = motion_ratio
        
        # 开始生成
        generated_motion_tokens = []
        audio_idx = 0
        
        from tqdm import tqdm
        for _ in tqdm(range(max_motion_tokens), desc="Generating motion tokens", leave=False):
            # 决定当前应该生成音频还是动作
            current_step = len(generated_motion_tokens)
            chunk_idx = current_step // motion_chunk_size
            
            # 构建输入序列
            inputs_list = []
            
            # 添加已处理的audio和motion chunks
            for i in range(chunk_idx + 1):
                # 添加audio chunk
                audio_start = i * audio_chunk_size
                audio_end = min(audio_start + audio_chunk_size, audio_embeds.shape[1])
                if audio_start < audio_embeds.shape[1]:
                    inputs_list.append(audio_embeds[:, audio_start:audio_end, :])
                
                # 添加motion chunk
                motion_start = i * motion_chunk_size
                motion_end = min(motion_start + motion_chunk_size, current_step)
                if motion_start < current_step:
                    motion_chunk = generated_motion_tokens[motion_start:motion_end]
                    if motion_chunk:
                        motion_tensor = torch.tensor([motion_chunk], dtype=torch.long, device=device)
                        motion_embeds = self.adaptor.transformer.wte(motion_tensor)
                        inputs_list.append(motion_embeds)
            
            # 拼接输入
            if not inputs_list:
                break
            inputs_embeds = torch.cat(inputs_list, dim=1)
            
            # 前向传播
            outputs = self.adaptor.transformer(inputs_embeds=inputs_embeds)
            hidden_states = outputs[0]
            logits = self.adaptor.lm_head(hidden_states)
            
            # 取最后一个位置的logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_motion_tokens.append(next_token.item())
            
            # 检查是否遇到结束标记
            if next_token.item() == 0:  # padding token作为结束标记
                break
        
        return torch.tensor([generated_motion_tokens], dtype=torch.long, device=device)


# ==================== Loading ====================

def load_model(checkpoint_path, device='cuda'):
    """从checkpoint加载模型"""
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    print(f"📊 Checkpoint info:")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"   Audio Loss: {checkpoint.get('audio_loss', 'N/A'):.4f}")
    print(f"   Motion Loss: {checkpoint.get('motion_loss', 'N/A'):.4f}")
    
    # 0. 加载 text tokenizer（用于tokenize text prompt）
    from transformers import AutoTokenizer
    from kimia_infer.utils.special_tokens import instantiate_extra_tokens
    
    kimi_model_path = config.get('kimi_model_path', 'moonshotai/Kimi-Audio-7B')
    print(f"\n🔄 Loading text tokenizer from: {kimi_model_path}")
    text_tokenizer = AutoTokenizer.from_pretrained(kimi_model_path, trust_remote_code=True)
    
    # 获取 kimia_text_blank token ID（用于填充对齐）
    extra_tokens = instantiate_extra_tokens(text_tokenizer)
    kimia_text_blank = extra_tokens.kimia_text_blank
    
    print(f"✅ Loaded text tokenizer")
    print(f"   kimia_text_blank token ID: {kimia_text_blank}")
    
    # 1. 创建基础模型结构
    # Kimi + LoRA
    print(f"\n🔄 Loading Kimi model structure: {kimi_model_path}")
    kimi_audio = KimiAudio(
        model_path=kimi_model_path,
        load_detokenizer=True,  # 需要加载 detokenizer 来生成音频
    )
    
    # 添加prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        
        return model_inputs
    
    kimi_audio.alm.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_for_generation, kimi_audio.alm
    )
    
    # 应用LoRA结构（权重会从checkpoint加载）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    kimi_lora = get_peft_model(kimi_audio.alm, lora_config)
    print(f"✅ Created LoRA structure (r={config.get('lora_r', 8)})")
    
    # 2. 创建GPT2 Adaptor结构（从config获取配置）
    print(f"\n🔄 Creating GPT2 Adaptor structure")
    gpt2_config = GPT2Config(
        vocab_size=1034,
        n_positions=4096,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    gpt2_adaptor = MixedInputGPT2(gpt2_config, audio_hidden_size=3584)
    print(f"✅ Created GPT2 Adaptor structure")
    
    # 3. 创建Audio embedding结构
    print(f"\n🔄 Creating Audio Embedding structure")
    # 从checkpoint的state_dict中推断维度
    audio_emb_key = 'audio_embedding.weight'
    if audio_emb_key in checkpoint['model_state']:
        audio_emb_shape = checkpoint['model_state'][audio_emb_key].shape
        audio_vocab_size, audio_embed_dim = audio_emb_shape
        print(f"   Detected embedding shape from checkpoint: {audio_emb_shape}")
    else:
        # 使用默认值
        audio_vocab_size = 4096
        audio_embed_dim = 3584
        print(f"   Using default embedding shape: ({audio_vocab_size}, {audio_embed_dim})")
    
    audio_embedding = nn.Embedding(audio_vocab_size, audio_embed_dim)
    print(f"✅ Created Audio Embedding structure ({audio_vocab_size} tokens, dim={audio_embed_dim})")
    
    # 4. 组装模型
    print(f"\n🔄 Assembling model...")
    model = KimiMotionModel(
        kimi_model=kimi_lora,
        gpt2_adaptor=gpt2_adaptor,
        audio_embedding=audio_embedding,
        kimi_audio_obj=kimi_audio,  # 保存完整的 KimiAudio 对象
        gumbel_tau=config.get('gumbel_tau', 1.0),
        kimia_text_blank=kimia_text_blank,  # 传入 blank token ID
    )
    
    # 5. 从checkpoint加载所有训练好的权重
    print(f"\n🔄 Loading trained weights from checkpoint...")
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model = model.to(device)
    model.eval()
    
    # 6. 更新 kimi_audio_obj.alm 为训练后的模型（重要！）
    print(f"\n🔄 Updating KimiAudio.alm with trained LoRA model...")
    kimi_audio.alm = model.kimi  # 使用训练后的 LoRA 模型
    print(f"✅ KimiAudio.alm updated with trained weights")
    
    print(f"✅ Model loaded successfully!")
    
    return model, config, text_tokenizer, kimia_text_blank


def load_jsonl_sample(jsonl_path, sample_idx):
    """从JSONL文件加载指定样本"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == sample_idx:
                return json.loads(line)
    raise ValueError(f"Sample {sample_idx} not found in {jsonl_path}")


def extract_tokens_from_sample(sample):
    """从样本中提取tokens"""
    conversation = sample['conversation']
    
    user_text_prompt = None
    user_audio_tokens = None
    assistant_audio_tokens_gt = None
    motion_tokens_gt = None
    
    for msg in conversation:
        if msg['role'] == 'user' and msg.get('message_type') == 'text':
            user_text_prompt = msg.get('content')
        elif msg['role'] == 'user' and msg.get('message_type') == 'audio':
            # 字段名是 audio_tokens，不是 audio_token_ids
            user_audio_tokens = msg.get('audio_tokens', [])
        elif msg['role'] == 'assistant':
            # 字段名是 audio_tokens 和 motion_tokens
            assistant_audio_tokens_gt = msg.get('audio_tokens', [])
            motion_tokens_gt = msg.get('motion_tokens', [])
    
    return {
        'user_text_prompt': user_text_prompt,
        'user_audio_tokens': user_audio_tokens if user_audio_tokens else [],
        'assistant_audio_tokens_gt': assistant_audio_tokens_gt if assistant_audio_tokens_gt else [],
        'motion_tokens_gt': motion_tokens_gt if motion_tokens_gt else [],
    }


# ==================== VQ-VAE Loading ====================

def load_vqvae_model(vqvae_config_name='g1_vqvae_arbitrary_length_balanced.yaml', 
                     vqvae_checkpoint=None, device='cuda'):
    """加载 VQ-VAE 模型"""
    # 加载配置
    config_path = os.path.join(ROOT, 'model/motion_encoder', vqvae_config_name)
    
    if not os.path.exists(config_path):
        print(f"⚠️  VQ-VAE config not found: {config_path}")
        return None, None, None
    
    print(f"\n🔄 Loading VQ-VAE config from: {config_path}")
    with open(config_path, 'r') as f:
        motion_config = yaml.safe_load(f)
    
    # 确定 checkpoint 路径
    if vqvae_checkpoint:
        checkpoint_path = vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        # 尝试查找常见的checkpoint路径
        finetuned_checkpoint = "output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt"
        pretrained_checkpoint = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
        
        if os.path.exists(finetuned_checkpoint):
            checkpoint_path = finetuned_checkpoint
        elif os.path.exists(pretrained_checkpoint):
            checkpoint_path = pretrained_checkpoint
        else:
            print(f"⚠️  No VQ-VAE checkpoint found, will skip decoding")
            return None, None, None
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    print(f"🔄 Loading VQ-VAE checkpoint from: {checkpoint_path}")
    
    # 加载归一化统计信息
    test_mean, test_std = load_normalization_stats(motion_config)
    print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")
    
    # 创建模型（直接解包配置）
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device=device)
    print(f"✅ VQ-VAE model loaded successfully!")
    
    return motion_vae, test_mean, test_std


def decode_motion_tokens(motion_tokens, vqvae_model, mean, std, device='cuda'):
    """使用VQ-VAE解码motion tokens为joint vectors"""
    if vqvae_model is None:
        print(f"⚠️  No VQ-VAE model available for decoding")
        return None
    
    try:
        # 过滤掉特殊tokens（1024-1033）
        filtered_tokens = [t for t in motion_tokens if t < 1024]
        
        if len(filtered_tokens) != len(motion_tokens):
            print(f"⚠️  Filtered {len(motion_tokens) - len(filtered_tokens)} special tokens")
            motion_tokens = filtered_tokens
        
        # 确保是偶数长度（body/hand 交替）
        if len(motion_tokens) % 2 != 0:
            motion_tokens = motion_tokens[:-1]
        
        if len(motion_tokens) == 0:
            print("❌ No motion tokens after processing")
            return None
        
        # 分离 body 和 hand tokens
        body_tokens = motion_tokens[0::2]  # 偶数索引
        hand_tokens = motion_tokens[1::2]  # 奇数索引
        
        # 限制 body tokens 在有效范围内 (0-511)
        body_tokens = [max(0, min(511, int(t))) for t in body_tokens]
        
        # 修正 hand tokens（应该在512-1023范围内）
        corrected_hand_tokens = []
        for t in hand_tokens:
            t_int = int(t)
            if t_int < 512:
                corrected_hand_tokens.append(512 + (t_int % 512))
            elif t_int > 1023:
                corrected_hand_tokens.append(512 + (t_int % 512))
            else:
                corrected_hand_tokens.append(t_int)
        hand_tokens = corrected_hand_tokens
        
        # 转换为tensor并减去偏移
        body_tokens_tensor = torch.tensor(body_tokens).unsqueeze(0).to(device)
        hand_tokens_tensor = torch.tensor(hand_tokens).unsqueeze(0).to(device) - 512
        
        with torch.no_grad():
            # 解码 tokens -> features (normalized)
            decoded = vqvae_model.decode((body_tokens_tensor, hand_tokens_tensor))
            # decoded shape: [1, seq_len, 491]
        
        # 转换为 numpy
        decoded_features_np = decoded[0].cpu().numpy()
        
        return decoded_features_np
    except Exception as e:
        print(f"⚠️  VQ-VAE decoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Test trained model from JSONL')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--jsonl_path', type=str,
                       default='data/BEAT_v2_1110_tokens.jsonl',
                       help='Path to JSONL file')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to test')
    parser.add_argument('--output_dir', type=str, default='test_output',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--max_motion_tokens', type=int, default=512,
                       help='Maximum motion tokens to generate')
    parser.add_argument('--vqvae_config', type=str, 
                       default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='VQ-VAE config file name')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                       help='VQ-VAE checkpoint path (optional)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🧪 Testing Trained Model from JSONL")
    print("=" * 80)
    
    # 1. 加载模型
    model, config, text_tokenizer, kimia_text_blank = load_model(args.checkpoint, device=args.device)
    
    # 2. 加载测试样本
    print(f"\n📂 Loading sample {args.sample_idx} from {args.jsonl_path}")
    sample = load_jsonl_sample(args.jsonl_path, args.sample_idx)
    tokens = extract_tokens_from_sample(sample)
    
    print(f"\n📊 Sample info:")
    print(f"   Text prompt: {tokens['user_text_prompt']}")
    print(f"   User audio tokens: {len(tokens['user_audio_tokens'])}")
    print(f"   GT audio tokens: {len(tokens['assistant_audio_tokens_gt'])}")
    print(f"   GT motion tokens: {len(tokens['motion_tokens_gt'])}")
    
    # 3. 准备输入 - 需要对齐 audio 和 text tokens 的长度
    user_audio_tokens_raw = tokens['user_audio_tokens']
    
    if tokens['user_text_prompt']:
        # Tokenize text prompt (需要指定 bos 和 eos 参数)
        text_token_ids = text_tokenizer.encode(tokens['user_text_prompt'], bos=False, eos=False)
        text_len = len(text_token_ids)
        audio_len = len(user_audio_tokens_raw)
        
        # 构建对齐的序列（与训练代码一致）
        # audio_input_ids: [blank]*text_len + audio_tokens
        audio_input_ids = [kimia_text_blank] * text_len + user_audio_tokens_raw
        # text_input_ids: text_tokens + [blank]*audio_len
        text_input_ids = text_token_ids + [kimia_text_blank] * audio_len
        
        # 验证长度一致
        assert len(audio_input_ids) == len(text_input_ids), \
            f"Length mismatch: audio_ids={len(audio_input_ids)} vs text_ids={len(text_input_ids)}"
        
        user_audio_tokens = torch.tensor([audio_input_ids], dtype=torch.long, device=args.device)
        user_text_tokens = torch.tensor([text_input_ids], dtype=torch.long, device=args.device)
        
        print(f"   Tokenized text prompt: {text_len} tokens")
        print(f"   Aligned sequence length: {len(audio_input_ids)} tokens")
    else:
        # 没有 text prompt，只使用 audio tokens
        user_audio_tokens = torch.tensor([user_audio_tokens_raw], dtype=torch.long, device=args.device)
        user_text_tokens = None
    
    # 4. 方案A：使用 Kimi 生成 audio（使用 KimiAudio.generate 方法）
    print(f"\n🚀 [方案A] Generating audio with Kimi...")
    
    # 获取原始音频路径
    user_audio_file = None
    for msg in sample['conversation']:
        if msg['role'] == 'user' and msg.get('message_type') == 'audio':
            user_audio_file = msg.get('content')
            break
    
    if user_audio_file and os.path.exists(user_audio_file) and model.kimi_audio_obj:
        try:
            # 构建 messages（暂时移除 text prompt，只使用 audio）
            messages = [
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": user_audio_file,
                }
            ]
            
            print(f"   Audio file: {user_audio_file}")
            print(f"   Messages: {len(messages)} message (audio only, no text prompt)")
            
            # 使用 KimiAudio.generate() 生成
            sampling_params = {
                "audio_temperature": args.temperature,
                "audio_top_k": args.top_k,
                "text_temperature": 0.0,
                "text_top_k": 5,
                "audio_repetition_penalty": 1.0,
                "audio_repetition_window_size": 64,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16,
            }
            
            print(f"   Calling KimiAudio.generate() with params:")
            print(f"      audio_temperature={args.temperature}, audio_top_k={args.top_k}")
            
            # 清理 CUDA 缓存
            torch.cuda.empty_cache()
            generated_wav, kimi_audio_tokens_result, generated_text = model.kimi_audio_obj.generate(
                messages, 
                **sampling_params, 
                output_type="both"
            )
            
            # 检查返回类型并转换为 tensor
            print(f"   [Debug] audio_tokens type: {type(kimi_audio_tokens_result)}")
            if isinstance(kimi_audio_tokens_result, torch.Tensor):
                # 已经是 tensor
                kimi_generated_audio_tokens = kimi_audio_tokens_result.to(args.device)
                if kimi_generated_audio_tokens.dim() == 1:
                    kimi_generated_audio_tokens = kimi_generated_audio_tokens.unsqueeze(0)
                kimi_audio_tokens_list = kimi_generated_audio_tokens[0].cpu().tolist()
            elif isinstance(kimi_audio_tokens_result, list):
                # 是 list
                kimi_audio_tokens_list = kimi_audio_tokens_result
                kimi_generated_audio_tokens = torch.tensor([kimi_audio_tokens_list], dtype=torch.long, device=args.device)
            else:
                raise ValueError(f"Unexpected audio_tokens type: {type(kimi_audio_tokens_result)}")
            
            print(f"✅ Kimi generated {len(kimi_audio_tokens_list)} audio tokens")
            if generated_text:
                print(f"✅ Kimi generated text: {generated_text}")
            
            # 保存 Kimi 生成的音频
            kimi_audio_wav_path = os.path.join(args.output_dir, 'kimi_generated_audio.wav')
            sf.write(
                kimi_audio_wav_path,
                generated_wav.detach().cpu().view(-1).numpy(),
                24000,
            )
            print(f"✅ Kimi audio saved to: {kimi_audio_wav_path}")
            
        except Exception as e:
            print(f"⚠️  Kimi generation failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"   将使用 GT audio tokens 作为 fallback")
            # 使用简单赋值，避免再次出错
            assistant_tokens = tokens['assistant_audio_tokens_gt']
            if len(assistant_tokens) > 0:
                kimi_generated_audio_tokens = torch.tensor([assistant_tokens], dtype=torch.long, device=args.device)
            else:
                kimi_generated_audio_tokens = torch.zeros((1, 1), dtype=torch.long, device=args.device)
            kimi_audio_wav_path = None
            torch.cuda.empty_cache()
    else:
        print(f"⚠️  User audio file not found or KimiAudio not available")
        print(f"   将使用 GT audio tokens 作为 fallback")
        # 使用简单赋值，避免出错
        assistant_tokens = tokens['assistant_audio_tokens_gt']
        if len(assistant_tokens) > 0:
            kimi_generated_audio_tokens = torch.tensor([assistant_tokens], dtype=torch.long, device=args.device)
        else:
            kimi_generated_audio_tokens = torch.zeros((1, 1), dtype=torch.long, device=args.device)
        kimi_audio_wav_path = None
    
    # 5. 从 Kimi 生成的 audio tokens 生成 motion tokens
    print(f"\n🚀 [方案A] Generating motion tokens from Kimi audio...")
    kimi_generated_motion_tokens = model.generate_motion_from_audio_tokens(
        audio_tokens=kimi_generated_audio_tokens,
        max_motion_tokens=args.max_motion_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"✅ Generated {kimi_generated_motion_tokens.shape[1]} motion tokens (from Kimi audio)")
    
    # 6. 方案B：使用 GT assistant audio tokens 生成 motion tokens
    print(f"\n📦 [方案B] Using GT assistant audio tokens...")
    assistant_audio_tokens_gt = tokens['assistant_audio_tokens_gt']
    if len(assistant_audio_tokens_gt) == 0:
        print(f"⚠️  No GT assistant audio tokens available, skipping GT audio test")
        gt_generated_motion_tokens = None
    else:
        assistant_audio_tokens = torch.tensor([assistant_audio_tokens_gt], dtype=torch.long, device=args.device)
        print(f"✅ GT assistant audio tokens: {len(assistant_audio_tokens_gt)} tokens")
        
        # 从 GT audio tokens 生成 motion tokens
        print(f"\n🚀 [方案B] Generating motion tokens from GT audio...")
        gt_generated_motion_tokens = model.generate_motion_from_audio_tokens(
            audio_tokens=assistant_audio_tokens,
            max_motion_tokens=args.max_motion_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"✅ Generated {gt_generated_motion_tokens.shape[1]} motion tokens (from GT audio)")
    
    # 7. 计算准确率
    motion_tokens_gt = tokens['motion_tokens_gt']
    
    # 方案A准确率（Kimi audio）
    kimi_motion_list = kimi_generated_motion_tokens[0].cpu().tolist()
    kimi_accuracy = 0.0
    if len(motion_tokens_gt) > 0 and len(kimi_motion_list) > 0:
        min_len = min(len(motion_tokens_gt), len(kimi_motion_list))
        matches = sum([1 for i in range(min_len) if motion_tokens_gt[i] == kimi_motion_list[i]])
        kimi_accuracy = matches / min_len * 100
    
    # 方案B准确率（GT audio）
    gt_audio_accuracy = 0.0
    gt_motion_list = []
    if gt_generated_motion_tokens is not None:
        gt_motion_list = gt_generated_motion_tokens[0].cpu().tolist()
        if len(motion_tokens_gt) > 0 and len(gt_motion_list) > 0:
            min_len = min(len(motion_tokens_gt), len(gt_motion_list))
            matches = sum([1 for i in range(min_len) if motion_tokens_gt[i] == gt_motion_list[i]])
            gt_audio_accuracy = matches / min_len * 100
    
    print(f"\n📈 Results:")
    print(f"   GT motion tokens: {len(motion_tokens_gt)}")
    print(f"   [方案A] Kimi audio → motion: {len(kimi_motion_list)} tokens, accuracy: {kimi_accuracy:.2f}%")
    if gt_generated_motion_tokens is not None:
        print(f"   [方案B] GT audio → motion: {len(gt_motion_list)} tokens, accuracy: {gt_audio_accuracy:.2f}%")
    
    # 8. 保存结果
    report = {
        'sample_idx': args.sample_idx,
        'checkpoint': args.checkpoint,
        'text_prompt': tokens['user_text_prompt'],
        'kimi_audio_motion': {
            'audio_tokens': kimi_generated_audio_tokens.shape[1],
            'motion_tokens': len(kimi_motion_list),
            'motion_accuracy': f"{kimi_accuracy:.2f}%",
        },
        'gt_audio_motion': {
            'audio_tokens': len(assistant_audio_tokens_gt),
            'motion_tokens': len(gt_motion_list) if gt_generated_motion_tokens is not None else 0,
            'motion_accuracy': f"{gt_audio_accuracy:.2f}%" if gt_generated_motion_tokens is not None else "N/A",
        },
        'ground_truth': {
            'audio_tokens': len(assistant_audio_tokens_gt),
            'motion_tokens': len(motion_tokens_gt),
        },
    }
    
    report_path = os.path.join(args.output_dir, 'test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    # 保存tokens
    tokens_path = os.path.join(args.output_dir, 'generated_tokens.pkl')
    with open(tokens_path, 'wb') as f:
        pickle.dump({
            'kimi_audio_tokens': kimi_generated_audio_tokens.cpu().numpy(),
            'kimi_motion_tokens': kimi_motion_list,
            'gt_audio_tokens': assistant_audio_tokens_gt,
            'gt_audio_motion_tokens': gt_motion_list if gt_generated_motion_tokens is not None else [],
            'gt_motion_tokens': motion_tokens_gt,
        }, f)
    
    print(f"✅ Tokens saved to: {tokens_path}")
    
    # 8. 拷贝原始音频文件（user 和 assistant）
    print(f"\n📁 Copying original audio files...")
    user_audio_path = None
    assistant_audio_path = None
    
    for msg in sample['conversation']:
        if msg['role'] == 'user' and msg.get('message_type') == 'audio':
            audio_src = msg.get('content')
            if audio_src and os.path.exists(audio_src):
                user_audio_dst = os.path.join(args.output_dir, 'user_audio.wav')
                shutil.copy2(audio_src, user_audio_dst)
                user_audio_path = user_audio_dst
                print(f"✅ User audio copied to: {user_audio_dst}")
        
        elif msg['role'] == 'assistant' and msg.get('message_type') in ['audio', 'audio_motion']:
            audio_src = msg.get('content')
            if audio_src and os.path.exists(audio_src):
                assistant_audio_dst = os.path.join(args.output_dir, 'assistant_audio.wav')
                shutil.copy2(audio_src, assistant_audio_dst)
                assistant_audio_path = assistant_audio_dst
                print(f"✅ Assistant audio copied to: {assistant_audio_dst}")
    
    # 使用 assistant audio 作为主音频（用于视频）
    audio_path = assistant_audio_path if assistant_audio_path else user_audio_path
    
    # 9. 加载 VQ-VAE 并解码 motion tokens
    print(f"\n🔄 Loading VQ-VAE for decoding...")
    vqvae, mean, std = load_vqvae_model(
        vqvae_config_name=args.vqvae_config,
        vqvae_checkpoint=args.vqvae_checkpoint,
        device=args.device
    )
    
    if vqvae is not None:
        # 解码 GT motion
        print(f"\n🔄 Decoding GT motion tokens...")
        gt_motion_features = decode_motion_tokens(motion_tokens_gt, vqvae, mean, std, args.device)
        
        # 解码方案A：从Kimi audio生成的motion
        print(f"\n🔄 [方案A] Decoding Kimi generated motion tokens...")
        kimi_motion_features = decode_motion_tokens(kimi_motion_list, vqvae, mean, std, args.device)
        
        # 解码方案B：从GT audio生成的motion
        gt_audio_motion_features = None
        if gt_generated_motion_tokens is not None:
            print(f"\n🔄 [方案B] Decoding GT audio generated motion tokens...")
            gt_audio_motion_features = decode_motion_tokens(gt_motion_list, vqvae, mean, std, args.device)
        
        if gt_motion_features is not None and kimi_motion_features is not None:
            # 转换为 pkl 和 csv
            print(f"\n📊 Converting to pkl and csv...")
            
            # 1. GT motion
            gt_motion_tensor = torch.from_numpy(gt_motion_features).unsqueeze(0).to(args.device).float()
            gt_motion_pkl = feats2datapkl(gt_motion_tensor, mean=mean, std=std)
            gt_pkl_path = os.path.join(args.output_dir, 'gt_motion.pkl')
            with open(gt_pkl_path, 'wb') as f:
                pickle.dump(gt_motion_pkl, f)
            
            gt_motion_csv = load_motion_pkl_as_csv_data(gt_pkl_path)
            gt_csv_path = os.path.join(args.output_dir, 'gt_motion.csv')
            np.savetxt(gt_csv_path, gt_motion_csv, delimiter=',', fmt='%.8f')
            print(f"✅ Saved GT motion: {gt_csv_path}")
            
            # 2. 方案A: 从Kimi audio生成的motion
            kimi_motion_tensor = torch.from_numpy(kimi_motion_features).unsqueeze(0).to(args.device).float()
            kimi_motion_pkl = feats2datapkl(kimi_motion_tensor, mean=mean, std=std)
            kimi_pkl_path = os.path.join(args.output_dir, 'kimi_audio_motion.pkl')
            with open(kimi_pkl_path, 'wb') as f:
                pickle.dump(kimi_motion_pkl, f)
            
            kimi_motion_csv = load_motion_pkl_as_csv_data(kimi_pkl_path)
            kimi_csv_path = os.path.join(args.output_dir, 'kimi_audio_motion.csv')
            np.savetxt(kimi_csv_path, kimi_motion_csv, delimiter=',', fmt='%.8f')
            print(f"✅ Saved Kimi audio motion: {kimi_csv_path}")
            
            # 3. 方案B: 从GT audio生成的motion（如果有）
            if gt_audio_motion_features is not None:
                gt_audio_motion_tensor = torch.from_numpy(gt_audio_motion_features).unsqueeze(0).to(args.device).float()
                gt_audio_motion_pkl = feats2datapkl(gt_audio_motion_tensor, mean=mean, std=std)
                gt_audio_pkl_path = os.path.join(args.output_dir, 'gt_audio_motion.pkl')
                with open(gt_audio_pkl_path, 'wb') as f:
                    pickle.dump(gt_audio_motion_pkl, f)
                
                gt_audio_motion_csv = load_motion_pkl_as_csv_data(gt_audio_pkl_path)
                gt_audio_csv_path = os.path.join(args.output_dir, 'gt_audio_motion.csv')
                np.savetxt(gt_audio_csv_path, gt_audio_motion_csv, delimiter=',', fmt='%.8f')
                print(f"✅ Saved GT audio motion: {gt_audio_csv_path}")
            
            # 生成可视化视频
            print(f"\n🎬 Generating visualization videos...")
            
            # 1. GT motion 视频（GT audio）
            gt_video_path = os.path.join(args.output_dir, 'gt_motion.mp4')
            vis_audio_motion(
                gt_csv_path,
                output_path=gt_video_path,
                audio_path=assistant_audio_path,  # 使用 GT assistant audio
                robot_type="g1_brainco",
                rate_limit=False,
                motion_fps=25
            )
            print(f"✅ GT motion video: {gt_video_path}")
            
            # 2. 方案A: Kimi audio → motion 视频（配 Kimi 生成的音频）
            kimi_video_path = os.path.join(args.output_dir, 'kimi_audio_motion.mp4')
            vis_audio_motion(
                kimi_csv_path,
                output_path=kimi_video_path,
                audio_path=kimi_audio_wav_path if kimi_audio_wav_path else user_audio_path,  # 使用 Kimi 生成的音频
                robot_type="g1_brainco",
                rate_limit=False,
                motion_fps=25
            )
            print(f"✅ Kimi audio motion video: {kimi_video_path} (配音: {'Kimi生成' if kimi_audio_wav_path else 'User'})")
            
            # 3. 方案B: GT audio → motion 视频（如果有）
            if gt_audio_motion_features is not None:
                gt_audio_video_path = os.path.join(args.output_dir, 'gt_audio_motion.mp4')
                vis_audio_motion(
                    gt_audio_csv_path,
                    output_path=gt_audio_video_path,
                    audio_path=assistant_audio_path,  # 使用 GT assistant audio
                    robot_type="g1_brainco",
                    rate_limit=False,
                    motion_fps=25
                )
                print(f"✅ GT audio motion video: {gt_audio_video_path}")
            
            print(f"\n🎉 Testing completed!")
            print(f"\n📁 Output files:")
            print(f"   - Report: {report_path}")
            print(f"   - Tokens: {tokens_path}")
            print(f"\n🔊 Audio files:")
            print(f"   - User audio (问题): {user_audio_path}")
            print(f"   - Assistant audio (GT回答): {assistant_audio_path}")
            if kimi_audio_wav_path:
                print(f"   - Kimi generated audio (Kimi回答): {kimi_audio_wav_path}")
            print(f"\n🎬 Videos:")
            print(f"   1. GT motion (配GT assistant audio): {gt_video_path}")
            print(f"   2. Kimi audio → motion (配Kimi生成audio): {kimi_video_path}")
            if gt_audio_motion_features is not None:
                print(f"   3. GT audio → motion (配GT assistant audio): {gt_audio_video_path}")
            print(f"\n💡 对比说明:")
            print(f"   - 视频1 (GT): Ground Truth - GT音频+GT动作")
            print(f"   - 视频2 (Kimi): 完整端到端 - Kimi生成音频+生成动作")
            if gt_audio_motion_features is not None:
                print(f"   - 视频3 (GT audio): 仅测试motion - GT音频+生成动作")
            print(f"\n🎮 播放命令:")
            print(f"   vlc {gt_video_path}")
            print(f"   vlc {kimi_video_path}")
            if gt_audio_motion_features is not None:
                print(f"   vlc {gt_audio_video_path}")
    else:
        print(f"\n⚠️  VQ-VAE not available, skipping video generation")
        print(f"\n🎉 Testing completed (tokens only)")


if __name__ == '__main__':
    import types
    main()
