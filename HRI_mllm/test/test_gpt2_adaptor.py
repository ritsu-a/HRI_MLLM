#!/usr/bin/env python3
"""
测试GPT2 Motion Adaptor模型
支持从checkpoint文件(.pt)或transformers格式目录加载模型

VQ-VAE支持：
- 预训练模型: output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt
- Finetune模型: output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt
  使用 --vqvae_checkpoint 参数指定checkpoint路径
  使用基础配置文件 g1_vqvae_arbitrary_length_balanced.yaml 即可（finetune模型使用相同配置）
"""

import os
import argparse
import json
import pickle
import numpy as np
import torch
import yaml
from pathlib import Path
import subprocess

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from huggingface_hub import snapshot_download
from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT

from HRI_mllm.utils.motion_utils.g1ml3d_final import vec_to_data_pkl, feats2datapkl
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats
from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand

from kimia_infer.api.kimia import KimiAudio
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion

from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from types import SimpleNamespace
import soundfile as sf
from kimia_infer.api.prompt_manager import KimiAPromptManager

torch.cuda.set_device(0)


def generate_motion_tokens(model, audio_tokens, device="cuda", max_new_tokens=256, 
                         temperature=0.8, top_k=50, repetition_penalty=1.1):
    """
    使用GPT2 adaptor模型生成motion tokens
    支持special token的处理，确保训练与测试结构一致
    自动适配包含或不包含special token的输入数据
    
    Args:
        model: GPT2 adaptor模型
        audio_tokens: 音频token序列（可能包含或不包含special token）
        device: 设备
        max_new_tokens: 最大新生成token数量
        temperature: 温度参数
        top_k: top-k采样
        repetition_penalty: 重复惩罚
    
    Returns:
        generated_motion_tokens: 生成的motion token序列（已过滤special token，可直接用于VQ-VAE）
    """
    model.eval()
    
    # 获取special token ID（从模型配置中）
    gesture_start_token_id = getattr(model.config, 'gesture_start_token_id', 512*2 + 2)
    audio_gesture_start_token_id = getattr(model.config, 'audio_gesture_start_token_id', 512*2 + 3)
    gesture_end_token_id = getattr(model.config, 'gesture_end_token_id', 512*2 + 4)
    audio_gesture_end_token_id = getattr(model.config, 'audio_gesture_end_token_id', 512*2 + 5)
    
    # 定义所有special token ID
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    # 检测输入是否包含special token
    has_special_tokens = any(token in special_token_ids for token in audio_tokens)
    
    print(f"Input audio tokens length: {len(audio_tokens)}")
    print(f"Audio tokens range: [{min(audio_tokens)}, {max(audio_tokens)}]")
    print(f"Special token IDs: {special_token_ids}")
    print(f"Input contains special tokens: {has_special_tokens}")
    
    # 如果输入包含special token，过滤掉它们（因为测试时应该使用纯audio tokens）
    if has_special_tokens:
        print(f"⚠️  检测到输入包含special token，过滤掉special token用于推理")
        audio_tokens = [t for t in audio_tokens if t not in special_token_ids]
        print(f"   过滤后audio tokens长度: {len(audio_tokens)}")
    
    generated_motion_tokens = []  # 用于VQ-VAE的纯motion tokens（已过滤special token）
    current_seq = []  # 当前序列（包含special token，用于模型推理）
    token_labels = []  # token类型标签
    generated_history = []  # 生成历史（用于重复惩罚）
    
    interleave_audios, interleave_motions = 1, 1  # 从训练配置中获取
    motion_token_count = 0  # 总的motion token计数（不包括special token）
    
    with torch.no_grad():
        for i, audio_token in enumerate(audio_tokens):
            # 添加audio token（teacher-forcing）
            current_seq.append(audio_token)
            token_labels.append(-100)  # audio token对应的label为-100
            
            # 每interleave_audios个audio token后，生成interleave_motions个motion token
            if (i + 1) % interleave_audios == 0:
                motion_idx = i // interleave_audios * interleave_motions
                
                for j in range(interleave_motions):
                    # 准备输入
                    inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
                    attn_mask = torch.ones_like(inputs)
                    labels = torch.tensor(token_labels).unsqueeze(0).to(device)
                    
                    # 调用模型forward方法
                    output = model(input_data=inputs, attention_mask=attn_mask, labels=labels)
                    next_token_logits = output.logits[0, -1, :]
                    
                    # 应用重复惩罚（只对非special token应用）
                    if repetition_penalty != 1.0 and generated_history:
                        for token_id in set(generated_history):
                            if token_id not in special_token_ids and token_id < next_token_logits.size(-1):
                                next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                    
                    # 稳健采样逻辑：支持temperature=0贪心与top_k屏蔽
                    if temperature is not None and temperature > 0:
                        next_token_logits = next_token_logits / temperature
                        if top_k is not None and top_k > 0:
                            k = min(int(top_k), next_token_logits.size(-1))
                            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                            masked = torch.full_like(next_token_logits, float('-inf'))
                            masked[top_k_indices] = top_k_logits
                            next_token_logits = masked
                        probs = torch.softmax(next_token_logits, dim=-1)
                        if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() <= 0:
                            next_token = torch.argmax(next_token_logits, dim=-1).item()
                        else:
                            next_token = torch.multinomial(probs, 1).item()
                    else:
                        if top_k is not None and top_k > 0:
                            k = min(int(top_k), next_token_logits.size(-1))
                            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                            masked = torch.full_like(next_token_logits, float('-inf'))
                            masked[top_k_indices] = top_k_logits
                            next_token_logits = masked
                        next_token = torch.argmax(next_token_logits, dim=-1).item()
                    
                    # 处理special token：根据训练时的规则补充对应的token
                    tokens_to_add = []  # 要添加到序列的tokens（包括补充的token）
                    tokens_labels_to_add = []  # 对应的labels
                    
                    if next_token == gesture_start_token_id:
                        # 模型预测了gesture_start，按照训练规则补充audio_gesture_start
                        print(f"  🔵 检测到gesture_start token ({gesture_start_token_id})，补充audio_gesture_start")
                        tokens_to_add.append(gesture_start_token_id)
                        tokens_labels_to_add.append(gesture_start_token_id)  # motion类型
                        tokens_to_add.append(audio_gesture_start_token_id)
                        tokens_labels_to_add.append(-100)  # audio类型
                        # 注意：gesture_start和audio_gesture_start都不添加到generated_motion_tokens（VQ-VAE不需要）
                        
                    elif next_token == gesture_end_token_id:
                        # 模型预测了gesture_end，按照训练规则补充audio_gesture_end
                        print(f"  🔴 检测到gesture_end token ({gesture_end_token_id})，补充audio_gesture_end")
                        tokens_to_add.append(gesture_end_token_id)
                        tokens_labels_to_add.append(gesture_end_token_id)  # motion类型
                        tokens_to_add.append(audio_gesture_end_token_id)
                        tokens_labels_to_add.append(-100)  # audio类型
                        # 注意：gesture_end和audio_gesture_end都不添加到generated_motion_tokens（VQ-VAE不需要）
                        
                    elif next_token in special_token_ids:
                        # 其他special token（audio_gesture_start或audio_gesture_end）
                        # 这些应该是模型自动生成的补充token，直接添加
                        tokens_to_add.append(next_token)
                        tokens_labels_to_add.append(-100)  # audio类型
                        # 不添加到generated_motion_tokens
                        
                    else:
                        # 普通motion token
                        # 确保motion token在正确范围内
                        vocab_size = model.config.vocab_size
                        if next_token >= vocab_size:
                            print(f"⚠️  警告: motion token {next_token} >= vocab_size {vocab_size}, 调整为 {vocab_size - 1}")
                            next_token = vocab_size - 1
                        
                        # 对于motion token，根据奇偶性调整范围（可选，如果模型已经输出正确范围）
                        # 这里保持原样，因为模型应该已经学会了正确的范围
                        
                        tokens_to_add.append(next_token)
                        tokens_labels_to_add.append(next_token)  # motion token对应的label为自身
                        
                        # 添加到generated_motion_tokens（用于VQ-VAE）
                        generated_motion_tokens.append(next_token)
                        motion_token_count += 1
                        
                        # 添加到生成历史（用于重复惩罚）
                        generated_history.append(next_token)
                    
                    # 将生成的token添加到当前序列（用于后续推理）
                    for token, label in zip(tokens_to_add, tokens_labels_to_add):
                        current_seq.append(token)
                        token_labels.append(label)
                    
                    # 保持历史记录长度
                    if len(generated_history) > 100:
                        generated_history = generated_history[-100:]
                    
                    # 检查是否达到最大motion token数量（只计算实际motion token）
                    if len(generated_motion_tokens) >= max_new_tokens:
                        break
                
                # 检查是否达到最大motion token数量
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    print(f"\n📊 生成统计:")
    print(f"   Generated motion tokens (for VQ-VAE): {len(generated_motion_tokens)}")
    if generated_motion_tokens:
        print(f"   Motion tokens range: [{min(generated_motion_tokens)}, {max(generated_motion_tokens)}]")
    
    # 过滤掉所有special token（额外安全检查）
    filtered_tokens = [t for t in generated_motion_tokens if t not in special_token_ids]
    if len(filtered_tokens) != len(generated_motion_tokens):
        print(f"⚠️  过滤了 {len(generated_motion_tokens) - len(filtered_tokens)} 个special token")
        generated_motion_tokens = filtered_tokens
    
    return generated_motion_tokens


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """
    解码motion tokens为motion features
    注意：motion_tokens应该已经过滤掉所有special token（由generate_motion_tokens处理）
    
    Args:
        motion_tokens: List of motion tokens [body1, hand1, body2, hand2, ...]（已过滤special token）
        motion_vae: The motion VQ-VAE model
        mean_t: Mean tensor for denormalization
        std_t: Std tensor for denormalization
        expected_frames: Expected number of motion frames (optional, used for truncation)
    
    Returns:
        data_dict: Decoded motion data in pkl format
    """
    # 额外安全检查：过滤掉任何可能的special token（以防万一）
    special_token_ids = {512*2 + 2, 512*2 + 3, 512*2 + 4, 512*2 + 5}  # gesture_start, audio_gesture_start, gesture_end, audio_gesture_end
    filtered_tokens = [t for t in motion_tokens if t not in special_token_ids]
    if len(filtered_tokens) != len(motion_tokens):
        print(f"⚠️  decode_motion_tokens: 过滤了 {len(motion_tokens) - len(filtered_tokens)} 个special token")
        motion_tokens = filtered_tokens
    
    # Separate body and hand tokens
    # motion_tokens are alternating: [body, hand, body, hand, ...]
    if len(motion_tokens) % 2 != 0:
        print(f"Warning: motion_tokens length ({len(motion_tokens)}) is odd, dropping last token")
        motion_tokens = motion_tokens[:-1]
    
    # 检查是否有足够的tokens
    if len(motion_tokens) == 0:
        print("Error: No motion tokens after processing")
        return None
    
    body_tokens = motion_tokens[0::2]  # even indices
    hand_tokens = motion_tokens[1::2]  # odd indices
    
    # 验证和修正token范围
    # Body tokens应该在[0, 511]范围内
    body_tokens = [max(0, min(511, int(t))) for t in body_tokens]
    
    # Hand tokens应该在[512, 1023]范围内（在减去512之前）
    # 修正超出范围的hand tokens
    corrected_hand_tokens = []
    for t in hand_tokens:
        t_int = int(t)
        if t_int < 512:
            # 如果小于512，假设模型输出的是[0, 511]范围，需要加上512
            corrected_hand_tokens.append(512 + (t_int % 512))
        elif t_int > 1023:
            # 如果大于1023，限制到[512, 1023]
            corrected_hand_tokens.append(512 + (t_int % 512))
        else:
            corrected_hand_tokens.append(t_int)
    hand_tokens = corrected_hand_tokens
    
    # Convert to tensors
    body_tokens_tensor = torch.tensor(body_tokens).unsqueeze(0).to("cuda")
    # hand tokens need to subtract 512 offset
    hand_tokens_tensor = torch.tensor(hand_tokens).unsqueeze(0).to("cuda") - 512
    
    print(f"Body tokens shape: {body_tokens_tensor.shape}, range: [{body_tokens_tensor.min()}, {body_tokens_tensor.max()}]")
    print(f"Hand tokens shape: {hand_tokens_tensor.shape}, range: [{hand_tokens_tensor.min()}, {hand_tokens_tensor.max()}]")
    print(f"Hand tokens (before offset): {hand_tokens[:5]}... (should be in [512, 1023])")
    
    # Decode using VAE - this returns normalized features
    decoded = motion_vae.decode((body_tokens_tensor, hand_tokens_tensor))
    
    print(f"Decoded shape before truncation: {decoded.shape}")
    
    # If expected_frames is provided and decoded is longer, truncate to expected length
    if expected_frames is not None and decoded.shape[1] > expected_frames:
        decoded = decoded[:, :expected_frames, :]
        print(f"Truncated to expected length: {decoded.shape}")
    
    # Convert to data format (feats2datapkl will handle denormalization with provided stats)
    data_dict = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())
    
    return data_dict


def audioToken2motionPkl(audio_codes, motion_tokens_gt):
    """
    Convert audio codes to motion codes using the updated generation approach.
    """
    print("🔄 使用free generation模式生成motion tokens...")
    
    # 生成motion tokens
    motion_tokens = generate_motion_tokens(
        motion_adaptor, 
        audio_codes.squeeze(0), 
        device="cuda",
        max_new_tokens=4096,
        temperature=1.2,
        top_k=100,
        repetition_penalty=1.6
    )
    
    # 解码motion tokens为motion data
    motion_pkl = decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t)
    
    return motion_pkl, motion_tokens


# 解析命令行参数
parser = argparse.ArgumentParser(description='Test GPT2 Motion Adaptor')
parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                   help='VQ-VAE config file name. Can use base config with finetuned checkpoint.')
parser.add_argument('--vqvae_checkpoint', type=str, default='output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt',
                   help='VQ-VAE checkpoint path. Supports both pretrained and finetuned checkpoints. '
                        'If not provided, will use the one in config file. '
                        'Examples: output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt '
                        'or output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt')
parser.add_argument('--audio_path', type=str, default="/root/workspace/HRI_MLLM/data/beat_english_v0.2.1/1/1_wayne_0_1_1_qwen1.wav",
                   help='Input audio file path')
parser.add_argument('--motion_adaptor_path', type=str, 
                   default="/root/workspace/HRI_MLLM/output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1200.pt",
                   help='Motion adaptor model path. Supports two formats: 1) checkpoint file (.pt format), 2) transformers format directory (with config.json and pytorch_model.bin)')
args = parser.parse_args()

def load_gpt2_model(model_path, device="cuda"):
    """加载训练完成的GPT2 adaptor模型
    
    支持两种格式：
    1. transformers格式目录（包含config.json和pytorch_model.bin）
    2. checkpoint文件（.pt格式，包含model_state）
    """
    print(f"Loading GPT2 adaptor model from: {model_path}")
    
    # 检查是否是.pt checkpoint文件
    if model_path.endswith('.pt'):
        return load_gpt2_from_checkpoint(model_path, device)
    else:
        return load_gpt2_from_transformers(model_path, device)


def load_gpt2_from_checkpoint(checkpoint_path, device="cuda"):
    """从.pt checkpoint文件加载模型"""
    print(f"🔄 Loading from checkpoint: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 从checkpoint中获取epoch信息
    epoch = checkpoint.get('epoch', 0)
    print(f"📊 Checkpoint epoch: {epoch}")
    
    # 创建模型配置（使用训练时的默认配置）
    model_config = GPT2Config(
        vocab_size=1034,  # 512*2 + 10 (motion_vocab_size + pad_token)
        n_positions=4096,  # max_seq_length
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    # 添加special token ID到模型配置（与训练时保持一致）
    model_config.gesture_start_token_id = 512*2 + 2
    model_config.audio_gesture_start_token_id = 512*2 + 3
    model_config.gesture_end_token_id = 512*2 + 4
    model_config.audio_gesture_end_token_id = 512*2 + 5
    
    # 创建模型实例
    model = MixedInputGPT2(model_config, audio_hidden_size=3584)
    
    # 加载模型权重
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print("✅ Loaded model_state from checkpoint")
    else:
        print("⚠️  model_state not found in checkpoint, using random weights")
    
    model.eval()
    model.to(device)
    print(f"✅ GPT2 adaptor model loaded successfully from checkpoint!")
    print(f"   Special token IDs: gesture_start={model_config.gesture_start_token_id}, "
          f"audio_gesture_start={model_config.audio_gesture_start_token_id}, "
          f"gesture_end={model_config.gesture_end_token_id}, "
          f"audio_gesture_end={model_config.audio_gesture_end_token_id}")
    return model


def load_gpt2_from_transformers(model_path, device="cuda"):
    """从transformers格式目录加载模型"""
    print(f"🔄 Loading from transformers format: {model_path}")
    
    # 加载配置
    config_path = os.path.join(model_path, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    
    # 确保special token ID存在（如果config中没有，使用默认值）
    if not hasattr(config, 'gesture_start_token_id'):
        config.gesture_start_token_id = 512*2 + 2
        config.audio_gesture_start_token_id = 512*2 + 3
        config.gesture_end_token_id = 512*2 + 4
        config.audio_gesture_end_token_id = 512*2 + 5
        print("⚠️  Special token IDs not found in config, using default values")
    
    # 创建模型实例
    model = MixedInputGPT2(config, audio_hidden_size=3584)
    
    # 加载模型权重
    model_path_pytorch = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_path_pytorch):
        state_dict = torch.load(model_path_pytorch, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pytorch_model.bin")
    else:
        print("⚠️  pytorch_model.bin not found, using random weights")
    
    model.eval()
    model.to(device)
    print(f"✅ GPT2 adaptor model loaded successfully from transformers format!")
    print(f"   Special token IDs: gesture_start={config.gesture_start_token_id}, "
          f"audio_gesture_start={config.audio_gesture_start_token_id}, "
          f"gesture_end={config.gesture_end_token_id}, "
          f"audio_gesture_end={config.audio_gesture_end_token_id}")
    return model

# 加载motion adaptor模型
motion_adaptor = load_gpt2_model(args.motion_adaptor_path)

### Loading motion VQ-VAE (Semantic Enhanced)
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

# 加载配置文件
config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
print(f"Loading VQ-VAE config from: {config_path}")
motion_config = open_yaml(config_path)

# 确定checkpoint路径
if args.vqvae_checkpoint:
    checkpoint_path = args.vqvae_checkpoint
elif "ckpt" in motion_config and motion_config["ckpt"]:
    checkpoint_path = motion_config["ckpt"]
else:
    # 尝试使用finetuned checkpoint作为默认值（如果存在）
    finetuned_checkpoint = "output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt"
    pretrained_checkpoint = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
    
    if os.path.exists(finetuned_checkpoint):
        checkpoint_path = finetuned_checkpoint
        print(f"⚠️  No checkpoint specified, using finetuned checkpoint: {checkpoint_path}")
    elif os.path.exists(pretrained_checkpoint):
        checkpoint_path = pretrained_checkpoint
        print(f"⚠️  No checkpoint specified, using pretrained checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = pretrained_checkpoint
        print(f"⚠️  No checkpoint specified, using default: {checkpoint_path}")

# 检查checkpoint是否存在
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print(f"\n可用的checkpoint路径示例:")
    print(f"  - 预训练模型: output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt")
    print(f"  - Finetune模型: output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt")
    print(f"\n请使用 --vqvae_checkpoint 参数指定正确的路径")
    exit(1)

print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")

# 检测是否是finetuned checkpoint
is_finetuned = "finetune" in checkpoint_path.lower() or "vqvae_finetune" in checkpoint_path
if is_finetuned:
    print(f"📦 Detected finetuned VQ-VAE checkpoint")

# 加载归一化统计量（与训练时保持一致）
# 注意：finetuned模型使用与pretrained模型相同的归一化统计量（都使用mixed statistics）
test_mean, test_std = load_normalization_stats(motion_config)
mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")

# 加载模型
motion_vae = VQVaeBodyHand(**motion_config)
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")
print(f"✅ VQ-VAE model loaded successfully!")
if is_finetuned:
    print(f"   Using finetuned model (better performance on BEAT and seg_finger datasets)")





# 使用命令行指定的音频路径
audio_path = args.audio_path
print(f"Using audio file: {audio_path}")


# audio_tokens = torch.load(audio_token_path).squeeze(0)
cache_path = snapshot_download("moonshotai/Kimi-Audio-7B")
model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)

prompt_manager = KimiAPromptManager(
        model_path=cache_path, kimia_token_offset=model_config.kimia_token_offset, kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
    )
audio_tokens = torch.from_numpy(np.array(prompt_manager._tokenize_audio(audio_path))).unsqueeze(0)

## Use a local HuggingFace model to inference.




# motion_tokens = motion_vae.encode(normalize_vec(torch.from_numpy(train_data_feature).unsqueeze(0).to("cuda:0")))[0]

# 生成motion并保存结果
motion_pkl, llm_motion_tokens = audioToken2motionPkl(audio_tokens, None)

# 添加调试信息
print(f"\n📊 生成统计信息:")
print(f"   - 生成的motion tokens数量: {len(llm_motion_tokens)}")
print(f"   - Body tokens: {len(llm_motion_tokens[0::2])}")
print(f"   - Hand tokens: {len(llm_motion_tokens[1::2])}")
print(f"   - Body token范围: [{min(llm_motion_tokens[0::2])}, {max(llm_motion_tokens[0::2])}]")
print(f"   - Hand token范围: [{min(llm_motion_tokens[1::2])}, {max(llm_motion_tokens[1::2])}]")

print("\n✅ 使用统一的free generation模式生成motion")
print("💡 建议:")
print("   1. 检查生成的motion质量")
print("   2. 如果效果不好，可能需要调整temperature、top_k等参数")
print("   3. 确保VQ-VAE解码和数据预处理正确")

# 保存结果
with open("output.pkl", 'wb') as f:
    pickle.dump(motion_pkl, f)
motion_csv = load_motion_pkl_as_csv_data("output.pkl")
np.savetxt("llm.csv", motion_csv, delimiter=',', fmt='%.8f')

# 复制音频文件
import shutil
shutil.copyfile(audio_path, "audio.wav")

# 生成可视化视频
vis_audio_motion("llm.csv", output_path="final_output_llm.mp4", audio_path="audio.wav", robot_type="g1_brainco", rate_limit=False, motion_fps=25)

print(f"\n🎉 处理完成!")
print(f"   - Motion pkl: output.pkl")
print(f"   - Motion csv: llm.csv") 
print(f"   - 可视化视频: final_output_llm.mp4")
    
