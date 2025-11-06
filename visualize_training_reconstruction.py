#!/usr/bin/env python3
"""
可视化训练json里动作序列重建效果的代码
使用训练完成的GPT2 adaptor模型生成motion tokens，然后通过VQ-VAE解码为动作序列
"""

import os
import argparse
import json
import pickle
import numpy as np
import torch
import shutil
import yaml
from pathlib import Path
import subprocess
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats, feats2datapkl
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion
from HRI_mllm import ROOT, DATA_ROOT

# GPT2 adaptor相关导入
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from types import SimpleNamespace


def open_yaml(path):
    """打开YAML配置文件"""
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


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


def generate_motion_tokens(model, audio_tokens, device="cuda", max_new_tokens=None, 
                         temperature=0.8, top_k=50, repetition_penalty=1.1, 
                         teacher_forcing=False, ground_truth_tokens=None):
    """
    使用GPT2 adaptor模型生成motion tokens
    参考test_gpt2_adaptor.py的实现方式
    
    Args:
        model: GPT2 adaptor模型
        audio_tokens: 音频token序列
        device: 设备
        max_new_tokens: 最大新生成token数量
        temperature: 温度参数
        top_k: top-k采样
        repetition_penalty: 重复惩罚
        teacher_forcing: 是否使用teacher forcing模式
        ground_truth_tokens: ground truth motion tokens（teacher forcing时使用）
    
    Returns:
        generated_motion_tokens: 生成的motion token序列
        loss: 计算的loss值（teacher forcing模式）
    """
    model.eval()
    
    print(f"Input audio tokens length: {len(audio_tokens)}")
    print(f"Audio tokens range: [{min(audio_tokens)}, {max(audio_tokens)}]")
    
    if teacher_forcing and ground_truth_tokens is not None:
        print(f"🎓 Using teacher forcing mode with {len(ground_truth_tokens)} ground truth tokens")
        return _teacher_forcing_generation(model, audio_tokens, ground_truth_tokens, device)
    else:
        print(f"🎲 Using free generation mode")
        return _free_generation(model, audio_tokens, max_new_tokens, temperature, top_k, repetition_penalty, device)


def _teacher_forcing_generation(model, audio_tokens, ground_truth_tokens, device):
    """
    正确的Teacher Forcing模式：
    1. 使用与训练时一致的方式：一次性处理完整序列
    2. 使用模型的内部loss计算
    3. 只对motion token计算loss
    4. 支持special token的处理
    """
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
    has_special_tokens_in_audio = any(token in special_token_ids for token in audio_tokens)
    has_special_tokens_in_motion = any(token in special_token_ids for token in ground_truth_tokens)
    
    if has_special_tokens_in_audio:
        print(f"⚠️  检测到audio tokens包含special token，过滤掉special token用于推理")
        audio_tokens = [t for t in audio_tokens if t not in special_token_ids]
        print(f"   过滤后audio tokens长度: {len(audio_tokens)}")
    
    # 过滤ground truth motion tokens中的special token（因为它们不应该影响loss计算）
    filtered_gt_tokens = [t for t in ground_truth_tokens if t not in special_token_ids]
    if has_special_tokens_in_motion:
        print(f"⚠️  检测到ground truth motion tokens包含special token，过滤掉special token")
        print(f"   过滤前motion tokens长度: {len(ground_truth_tokens)}, 过滤后: {len(filtered_gt_tokens)}")
        ground_truth_tokens = filtered_gt_tokens
    
    # 按照训练时的穿插模式构建完整序列：audio[0], motion[0], audio[1], motion[1], ...
    full_sequence = []
    labels = []
    masks = []  # 1表示motion token，0表示audio token
    
    interleave_audios, interleave_motions = 1, 1  # 从训练配置中获取
    
    for i in range(len(audio_tokens)):
        # 添加audio token
        full_sequence.append(audio_tokens[i])
        labels.append(-100)  # audio token的label为-100
        masks.append(0)  # audio token的mask为0
        
        # 每interleave_audios个audio token后，添加interleave_motions个motion token
        if (i + 1) % interleave_audios == 0:
            motion_idx = i // interleave_audios * interleave_motions
            for j in range(interleave_motions):
                if motion_idx + j < len(ground_truth_tokens):
                    # 确保motion token在模型vocab_size范围内 [0, 1025]
                    motion_token = ground_truth_tokens[motion_idx + j]
                    # 过滤special token（不应该在ground truth中出现，但为了安全起见）
                    if motion_token not in special_token_ids:
                        motion_token = min(motion_token, 1025)  # 限制在vocab_size范围内
                        full_sequence.append(motion_token)
                        labels.append(motion_token)  # motion token的label为自身
                        masks.append(1)  # motion token的mask为1
    
    # 转换为tensor
    inputs = torch.tensor(full_sequence).unsqueeze(0).to(device)
    labels_tensor = torch.tensor(labels).unsqueeze(0).to(device)
    masks_tensor = torch.tensor(masks).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(inputs)
    
    print(f"🎓 Teacher forcing input shape: {inputs.shape}")
    print(f"🎓 Teacher forcing labels shape: {labels_tensor.shape}")
    print(f"🎓 Teacher forcing masks shape: {masks_tensor.shape}")
    print(f"🎓 Motion tokens: {masks_tensor.sum().item()}/{masks_tensor.numel()}")
    
    # 使用与训练时一致的方式计算loss
    with torch.no_grad():
        # 调用模型，使用模型的内部loss计算
        output = model(input_data=inputs, attention_mask=attention_mask, labels=labels_tensor)
        
        # 使用模型计算的loss（与训练时一致）
        # 模型的loss计算会自动忽略label为-100的位置，只计算motion token的loss
        loss = output.loss.item() if output.loss is not None else 0.0
    
    print(f"🎓 Teacher forcing loss (motion tokens only): {loss:.6f}")
    
    # 提取生成的motion tokens（模型预测的）
    generated_motion_tokens = []
    motion_token_count = 0
    
    for i, token in enumerate(full_sequence):
        if masks[i] == 1:  # 如果是motion token
            # 获取模型对该位置的预测
            with torch.no_grad():
                # 使用到当前位置的序列作为输入
                partial_inputs = torch.tensor(full_sequence[:i+1]).unsqueeze(0).to(device)
                partial_attn = torch.ones_like(partial_inputs)
                partial_labels = torch.tensor(labels[:i+1]).unsqueeze(0).to(device)
                
                output = model(input_data=partial_inputs, attention_mask=partial_attn, labels=partial_labels)
                predicted_token = torch.argmax(output.logits[0, -1, :]).item()
                
                # 过滤special token
                if predicted_token not in special_token_ids:
                    # 确保motion token在正确范围内
                    if motion_token_count % 2 == 0:
                        # body token [0, 511]
                        predicted_token = int(predicted_token % 512)
                    else:
                        # hand token [512, 1023]
                        predicted_token = int(512 + (predicted_token % 512))
                    
                    generated_motion_tokens.append(predicted_token)
                    motion_token_count += 1
    
    print(f"🎓 Generated {len(generated_motion_tokens)} motion tokens (filtered special tokens)")
    if generated_motion_tokens:
        print(f"🎓 Generated tokens range: [{min(generated_motion_tokens)}, {max(generated_motion_tokens)}]")
    
    # 额外安全检查：过滤掉所有special token
    filtered_tokens = [t for t in generated_motion_tokens if t not in special_token_ids]
    if len(filtered_tokens) != len(generated_motion_tokens):
        print(f"⚠️  额外过滤了 {len(generated_motion_tokens) - len(filtered_tokens)} 个special token")
        generated_motion_tokens = filtered_tokens
    
    return generated_motion_tokens, loss


def _free_generation(model, audio_tokens, max_new_tokens, temperature, top_k, repetition_penalty, device):
    """
    自由生成模式：audio tokens作为teacher-forcing，只生成motion tokens
    支持special token的处理，确保训练与测试结构一致
    自动适配包含或不包含special token的输入数据
    """
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
                    
                    # 应用temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k采样
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits[top_k_indices] = top_k_logits
                    
                    # 采样
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
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
                        
                        # 根据奇偶性限制token范围
                        # 偶数次 motion（body）∈ [0, 511]；奇数次 motion（hand）∈ [512, 1023]
                        if motion_token_count % 2 == 0:
                            # body token [0, 511]
                            next_token = int(next_token % 512)
                        else:
                            # hand token [512, 1023]
                            # 确保在正确范围内
                            if next_token < 512:
                                next_token = 512 + (next_token % 512)
                            elif next_token > 1023:
                                next_token = 512 + (next_token % 512)
                            else:
                                next_token = int(next_token)
                        
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
                    
                    # 不再限制motion token数量，继续生成直到所有audio tokens处理完
    
    print(f"\n📊 生成统计:")
    print(f"   Generated motion tokens (for VQ-VAE): {len(generated_motion_tokens)}")
    if generated_motion_tokens:
        print(f"   Motion tokens range: [{min(generated_motion_tokens)}, {max(generated_motion_tokens)}]")
    
    # 过滤掉所有special token（额外安全检查）
    filtered_tokens = [t for t in generated_motion_tokens if t not in special_token_ids]
    if len(filtered_tokens) != len(generated_motion_tokens):
        print(f"⚠️  过滤了 {len(generated_motion_tokens) - len(filtered_tokens)} 个special token")
        generated_motion_tokens = filtered_tokens
    
    return generated_motion_tokens, None


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


def concat_side_by_side(video_left: str, video_right: str, output_path: str, target_height: int = 720):
    """使用 ffmpeg 横向拼接两个视频，统一高度为 target_height，保持宽高比。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-i", video_left,
        "-i", video_right,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1[left];[1:v]scale=-2:{target_height},setsar=1[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",  # 使用左视频音频（若存在）
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def concat_three_side_by_side(video1: str, video2: str, video3: str, output_path: str, target_height: int = 720):
    """使用 ffmpeg 横向拼接三个视频，统一高度为 target_height，保持宽高比。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-i", video1,
        "-i", video2,
        "-i", video3,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1[v1];[1:v]scale=-2:{target_height},setsar=1[v2];[2:v]scale=-2:{target_height},setsar=1[v3];[v1][v2]hstack=inputs=2[tmp];[tmp][v3]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",  # 使用第一个视频的音频（若存在）
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def add_text_to_video(video_path: str, output_path: str, text: str, position: str = "top-left", font_size: int = 30):
    """
    使用ffmpeg在视频上添加文本标注
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        text: 要添加的文本
        position: 文本位置 ("top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right")
        font_size: 字体大小
    """
    # 转义文本中的特殊字符（ffmpeg drawtext需要的转义）
    text_escaped = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace("[", "\\[").replace("]", "\\]")
    
    # 字体文件路径
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_file):
        # 如果字体文件不存在，尝试使用系统默认字体
        font_file_param = ""
    else:
        font_file_param = f"fontfile={font_file}:"
    
    # 根据位置设置坐标
    position_map = {
        "top-left": f"x={font_size//2}:y={font_size//2}",
        "top-center": f"x=(w-text_w)/2:y={font_size//2}",
        "top-right": f"x=w-text_w-{font_size//2}:y={font_size//2}",
        "bottom-left": f"x={font_size//2}:y=h-text_h-{font_size//2}",
        "bottom-center": f"x=(w-text_w)/2:y=h-text_h-{font_size//2}",
        "bottom-right": f"x=w-text_w-{font_size//2}:y=h-text_h-{font_size//2}",
    }
    
    pos = position_map.get(position, position_map["top-left"])
    
    # 构建drawtext filter
    drawtext_filter = f"drawtext=text='{text_escaped}':{font_file_param}fontsize={font_size}:fontcolor=white:bordercolor=black:borderw=2:{pos}"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-i", video_path,
        "-vf", drawtext_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "copy",  # 复制音频流
        output_path,
    ]
    subprocess.run(cmd, check=True)


def concat_side_by_side_with_labels(video_left: str, video_right: str, output_path: str, 
                                     label_left: str, label_right: str, target_height: int = 720):
    """
    横向拼接两个视频并添加文本标注
    
    Args:
        video_left: 左侧视频路径
        video_right: 右侧视频路径
        output_path: 输出视频路径
        label_left: 左侧视频的标注文本
        label_right: 右侧视频的标注文本
        target_height: 目标高度
    """
    # 转义文本中的特殊字符
    label_left_escaped = label_left.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace("[", "\\[").replace("]", "\\]")
    label_right_escaped = label_right.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace("[", "\\[").replace("]", "\\]")
    
    font_size = 30
    # 字体文件路径
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_file):
        font_param = f"fontfile={font_file}:"
    else:
        font_param = ""
    
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-i", video_left,
        "-i", video_right,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1,drawtext=text='{label_left_escaped}':{font_param}fontsize={font_size}:fontcolor=white:bordercolor=black:borderw=2:x={font_size//2}:y={font_size//2}[left];"
        f"[1:v]scale=-2:{target_height},setsar=1,drawtext=text='{label_right_escaped}':{font_param}fontsize={font_size}:fontcolor=white:bordercolor=black:borderw=2:x={font_size//2}:y={font_size//2}[right];"
        f"[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",  # 使用左视频音频（若存在）
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def concat_three_side_by_side_with_labels(video1: str, video2: str, video3: str, output_path: str,
                                          label1: str, label2: str, label3: str, target_height: int = 720):
    """
    横向拼接三个视频并添加文本标注
    
    Args:
        video1: 第一个视频路径
        video2: 第二个视频路径
        video3: 第三个视频路径
        output_path: 输出视频路径
        label1: 第一个视频的标注文本
        label2: 第二个视频的标注文本
        label3: 第三个视频的标注文本
        target_height: 目标高度
    """
    # 转义文本中的特殊字符
    label1_escaped = label1.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace("[", "\\[").replace("]", "\\]")
    label2_escaped = label2.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace("[", "\\[").replace("]", "\\]")
    label3_escaped = label3.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace("[", "\\[").replace("]", "\\]")
    
    font_size = 30
    # 字体文件路径
    font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_file):
        font_param = f"fontfile={font_file}:"
    else:
        font_param = ""
    
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-i", video1,
        "-i", video2,
        "-i", video3,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1,drawtext=text='{label1_escaped}':{font_param}fontsize={font_size}:fontcolor=white:bordercolor=black:borderw=2:x={font_size//2}:y={font_size//2}[v1];"
        f"[1:v]scale=-2:{target_height},setsar=1,drawtext=text='{label2_escaped}':{font_param}fontsize={font_size}:fontcolor=white:bordercolor=black:borderw=2:x={font_size//2}:y={font_size//2}[v2];"
        f"[2:v]scale=-2:{target_height},setsar=1,drawtext=text='{label3_escaped}':{font_param}fontsize={font_size}:fontcolor=white:bordercolor=black:borderw=2:x={font_size//2}:y={font_size//2}[v3];"
        f"[v1][v2]hstack=inputs=2[tmp];[tmp][v3]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",  # 使用第一个视频的音频（若存在）
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def smooth_qpos_data(data, max_velocity_change=0.5, oscillation_threshold=0.3, smoothing_window=5):
    """
    平滑处理qpos数据，处理突变和振荡问题
    
    Args:
        data: numpy array, shape (num_frames, num_dofs)，CSV格式的qpos数据
        max_velocity_change: float, 相邻帧之间允许的最大变化量（用于检测突变）
        oscillation_threshold: float, 振荡检测阈值（相邻帧变化方向反转的次数）
        smoothing_window: int, 平滑窗口大小（用于移动平均）
    
    Returns:
        smoothed_data: numpy array, 平滑后的数据
    """
    if data is None or data.shape[0] < 2:
        return data
    
    smoothed_data = data.copy().astype(np.float64)
    num_frames, num_dofs = smoothed_data.shape
    
    print(f"🔄 开始平滑处理: {num_frames}帧, {num_dofs}个自由度")
    
    # 第一步：限制速度突变（velocity limiting）
    # 检测并限制相邻帧之间的过大变化
    mutation_count = 0
    mutation_details = []
    for dof_idx in range(num_dofs):
        for frame_idx in range(1, num_frames):
            prev_val = smoothed_data[frame_idx - 1, dof_idx]
            curr_val = smoothed_data[frame_idx, dof_idx]
            change = curr_val - prev_val
            
            # 如果变化过大，限制到最大允许变化
            if abs(change) > max_velocity_change:
                smoothed_data[frame_idx, dof_idx] = prev_val + np.sign(change) * max_velocity_change
                mutation_count += 1
                if len(mutation_details) < 5:  # 只记录前5个突变详情
                    mutation_details.append(f"DOF {dof_idx}, 帧 {frame_idx}, 变化 {change:.4f}")
    
    if mutation_count > 0:
        print(f"  ⚠️  检测到 {mutation_count} 个突变，已限制速度")
        if mutation_details:
            for detail in mutation_details[:3]:  # 只显示前3个
                print(f"     - {detail}")
    
    # 第二步：检测和处理振荡
    # 振荡的特征：相邻帧之间的变化方向频繁反转
    oscillation_dofs = []
    for dof_idx in range(num_dofs):
        velocities = np.diff(smoothed_data[:, dof_idx])
        if len(velocities) < 3:
            continue
        
        # 检测方向反转
        sign_changes = np.diff(np.sign(velocities))
        oscillation_count = np.sum(np.abs(sign_changes) > 0)
        
        # 如果振荡次数过多，应用更强的平滑
        if oscillation_count > oscillation_threshold * (num_frames - 2):
            oscillation_dofs.append((dof_idx, oscillation_count))
            # 使用移动平均平滑
            window = min(smoothing_window, num_frames)
            for i in range(num_frames):
                start_idx = max(0, i - window // 2)
                end_idx = min(num_frames, i + window // 2 + 1)
                smoothed_data[i, dof_idx] = np.mean(smoothed_data[start_idx:end_idx, dof_idx])
    
    if oscillation_dofs:
        print(f"  ⚠️  检测到 {len(oscillation_dofs)} 个自由度存在振荡，已应用平滑处理")
        for dof_idx, count in oscillation_dofs[:5]:  # 只显示前5个
            print(f"     - DOF {dof_idx}: {count} 次振荡")
    
    # 第三步：应用低通滤波器（去除高频噪声）
    # 使用Butterworth低通滤波器
    cutoff_freq = 0.3  # 归一化截止频率（0~1）
    order = 4
    try:
        b, a = signal.butter(order, cutoff_freq, 'low')
        for dof_idx in range(num_dofs):
            smoothed_data[:, dof_idx] = signal.filtfilt(b, a, smoothed_data[:, dof_idx])
    except Exception as e:
        print(f"  ⚠️  低通滤波失败: {e}, 跳过滤波步骤")
    
    # 第四步：额外的平滑处理 - 使用高斯平滑
    # 对每个自由度应用高斯平滑
    try:
        sigma = 1.0  # 高斯核的标准差
        for dof_idx in range(num_dofs):
            smoothed_data[:, dof_idx] = gaussian_filter1d(
                smoothed_data[:, dof_idx], 
                sigma=sigma, 
                mode='nearest'
            )
    except Exception as e:
        print(f"  ⚠️  高斯平滑失败: {e}, 跳过此步骤")
    
    print(f"✅ 平滑处理完成")
    return smoothed_data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='可视化训练json里动作序列重建效果')
    parser.add_argument('--jsonl_path', type=str, default='/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl',
                       help='Path to the jsonl file containing tokens')
    parser.add_argument('--gpt2_model_path', type=str, 
                       default='/root/workspace/HRI_MLLM/output/motion_adaptor_v3/kimi_audio_motion_gpt2_brainco_30_100/transformers_format',
                       help='Path to the trained GPT2 adaptor model. Supports two formats: 1) transformers format directory (with config.json and pytorch_model.bin), 2) checkpoint file (.pt format). Not required if --compare_models is set.')
    parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='VQ-VAE config file name')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                       help='VQ-VAE checkpoint path. If not provided, will use the one in config file')
    parser.add_argument('--output_dir', type=str, default='./training_reconstruction_visualization',
                       help='Output directory for visualization files')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to process')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                       help='Maximum number of new tokens to generate for motion tokens (None = no limit, generate until all audio tokens processed)')
    parser.add_argument('--temperature', type=float, default=1.8,
                       help='Temperature for generation')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.8,
                       help='Repetition penalty parameter')
    parser.add_argument('--teacher_forcing', action='store_true',
                       help='Use teacher forcing mode to calculate loss')
    parser.add_argument('--compare_modes', action='store_true',
                       help='Compare both teacher forcing and free generation modes')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare two different models in free-running mode')
    parser.add_argument('--model1_path', type=str, default=None,
                       help='Path to first model for comparison (required if --compare_models is set)')
    parser.add_argument('--model2_path', type=str, default=None,
                       help='Path to second model for comparison (required if --compare_models is set)')

    args = parser.parse_args()

    # Validate arguments
    if args.compare_models:
        if args.model1_path is None or args.model2_path is None:
            print("❌ Error: --model1_path and --model2_path are required when --compare_models is set")
            exit(1)
        if not os.path.exists(args.model1_path):
            print(f"❌ Error: Model 1 not found: {args.model1_path}")
            exit(1)
        if not os.path.exists(args.model2_path):
            print(f"❌ Error: Model 2 not found: {args.model2_path}")
            exit(1)
    else:
        # Validate single model path if not comparing models
        if args.gpt2_model_path and not os.path.exists(args.gpt2_model_path):
            print(f"❌ Error: Model not found: {args.gpt2_model_path}")
            exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GPT2 adaptor model(s)
    if args.compare_models:
        print(f"\n{'='*60}")
        print("🔄 Loading two models for comparison...")
        print(f"{'='*60}")
        gpt2_model1 = load_gpt2_model(args.model1_path)
        gpt2_model2 = load_gpt2_model(args.model2_path)
        gpt2_model = gpt2_model1  # Use model1 as default for backward compatibility
        print(f"✅ Both models loaded successfully!")
    else:
        gpt2_model = load_gpt2_model(args.gpt2_model_path)

    # Load VQ-VAE configuration
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
    print(f"Loading VQ-VAE config from: {config_path}")
    motion_config = open_yaml(config_path)

    # Determine checkpoint path
    if args.vqvae_checkpoint:
        checkpoint_path = args.vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
        print(f"⚠️  No checkpoint specified in config, using default: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)

    print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")

    # Load normalization statistics
    test_mean, test_std = load_normalization_stats(motion_config)
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
    std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
    print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")

    # Load the VAE model
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device="cuda")
    print(f"✅ VQ-VAE model loaded successfully!")

    # Load and process samples from jsonl
    print(f"Reading samples from: {args.jsonl_path}")
    with open(args.jsonl_path, 'r') as f:
        lines = f.readlines()

    processed_count = 0
    for i, line in enumerate(lines[:args.num_samples]):
        try:
            data = json.loads(line.strip())
            
            # Extract audio path, audio tokens, and ground truth motion tokens
            audio_path = None
            motion_tokens_gt = None
            audio_tokens = None

            for msg in data['conversation']:
                if msg.get('message_type') == 'audio' and msg.get('audio_tokens'):
                    audio_path = msg['content']
                    audio_tokens = msg['audio_tokens']
                
                if msg.get('message_type') == 'audio_motion' and msg.get('motion_tokens'):
                    motion_tokens_gt = msg['motion_tokens']

            if motion_tokens_gt is None or audio_tokens is None:
                print(f"Skipping sample {i+1}: Missing tokens")
                continue

            print(f"\n{'='*60}")
            print(f"Processing sample {i+1}/{min(len(lines), args.num_samples)}")
            print(f"Audio path: {audio_path}")
            print(f"Audio tokens length: {len(audio_tokens)}")
            print(f"Ground truth motion tokens length: {len(motion_tokens_gt)}")

            # Create sample output directory
            sample_dir = os.path.join(args.output_dir, f"sample_{i+1}")
            os.makedirs(sample_dir, exist_ok=True)

            # Generate motion tokens using GPT2 adaptor
            print("\nGenerating motion tokens with GPT2 adaptor...")
            
            if args.compare_models:
                print("\n🔄 Comparing two models in free-running mode...")
                
                # Get model names for labeling
                # Extract version from path (e.g., motion_adaptor_v6 from .../motion_adaptor_v6/...)
                model1_path_parts = args.model1_path.split('/')
                model2_path_parts = args.model2_path.split('/')
                
                # Try to find motion_adaptor_v* in the path
                model1_name = "model1"
                model2_name = "model2"
                for part in model1_path_parts:
                    if 'motion_adaptor_v' in part:
                        model1_name = part
                        break
                for part in model2_path_parts:
                    if 'motion_adaptor_v' in part:
                        model2_name = part
                        break
                
                # Fallback: use directory name if not found
                if model1_name == "model1":
                    model1_name = os.path.basename(os.path.dirname(os.path.dirname(args.model1_path)))
                if model2_name == "model2":
                    model2_name = os.path.basename(os.path.dirname(os.path.dirname(args.model2_path)))
                
                print(f"   Model 1: {model1_name}")
                print(f"   Model 2: {model2_name}")
                
                # Free Generation with Model 1 (for visualization)
                print(f"\n🎲 Free Generation Mode - Model 1 ({model1_name}):")
                model1_tokens, _ = generate_motion_tokens(
                    gpt2_model1, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=False
                )
                
                # Teacher Forcing with Model 1 (for loss calculation)
                print(f"\n🎓 Teacher Forcing Mode - Model 1 ({model1_name}):")
                _, model1_tf_loss = generate_motion_tokens(
                    gpt2_model1, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=True,
                    ground_truth_tokens=motion_tokens_gt
                )
                
                # Free Generation with Model 2 (for visualization)
                print(f"\n🎲 Free Generation Mode - Model 2 ({model2_name}):")
                model2_tokens, _ = generate_motion_tokens(
                    gpt2_model2, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=False
                )
                
                # Teacher Forcing with Model 2 (for loss calculation)
                print(f"\n🎓 Teacher Forcing Mode - Model 2 ({model2_name}):")
                _, model2_tf_loss = generate_motion_tokens(
                    gpt2_model2, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=True,
                    ground_truth_tokens=motion_tokens_gt
                )
                
                print(f"\n📊 Model Comparison Results:")
                print(f"   Free Generation:")
                print(f"      Model 1 tokens: {len(model1_tokens)}")
                print(f"      Model 2 tokens: {len(model2_tokens)}")
                print(f"   Teacher Forcing Loss:")
                print(f"      Model 1 ({model1_name}): {model1_tf_loss:.6f}")
                print(f"      Model 2 ({model2_name}): {model2_tf_loss:.6f}")
                if model1_tf_loss is not None and model2_tf_loss is not None:
                    loss_diff = model1_tf_loss - model2_tf_loss
                    better_model = model1_name if model1_tf_loss < model2_tf_loss else model2_name
                    print(f"      Difference: {abs(loss_diff):.6f} ({better_model} is better)")
                
                # Decode Model 1 motion tokens
                print(f"\nDecoding Model 1 motion tokens...")
                model1_motion_pkl = decode_motion_tokens(
                    model1_tokens, motion_vae, mean_t, std_t
                )
                
                # Save Model 1 motion
                model1_pkl_path = os.path.join(sample_dir, f"model1_{model1_name}_motion.pkl")
                with open(model1_pkl_path, 'wb') as f:
                    pickle.dump(model1_motion_pkl, f)
                
                model1_csv = load_motion_pkl_as_csv_data(model1_pkl_path)
                # 应用平滑处理，处理突变和振荡
                model1_csv = smooth_qpos_data(model1_csv)
                model1_csv_path = os.path.join(sample_dir, f"model1_{model1_name}_motion.csv")
                np.savetxt(model1_csv_path, model1_csv, delimiter=',', fmt='%.8f')
                print(f"Saved Model 1 motion CSV to: {model1_csv_path}")
                
                # Decode Model 2 motion tokens
                print(f"\nDecoding Model 2 motion tokens...")
                model2_motion_pkl = decode_motion_tokens(
                    model2_tokens, motion_vae, mean_t, std_t
                )
                
                # Save Model 2 motion
                model2_pkl_path = os.path.join(sample_dir, f"model2_{model2_name}_motion.pkl")
                with open(model2_pkl_path, 'wb') as f:
                    pickle.dump(model2_motion_pkl, f)
                
                model2_csv = load_motion_pkl_as_csv_data(model2_pkl_path)
                # 应用平滑处理，处理突变和振荡
                model2_csv = smooth_qpos_data(model2_csv)
                model2_csv_path = os.path.join(sample_dir, f"model2_{model2_name}_motion.csv")
                np.savetxt(model2_csv_path, model2_csv, delimiter=',', fmt='%.8f')
                print(f"Saved Model 2 motion CSV to: {model2_csv_path}")
                
                # Visualize Model 1 motion
                print(f"\nCreating Model 1 visualization...")
                model1_video_path_temp = os.path.join(sample_dir, f"model1_{model1_name}_motion_temp.mp4")
                model1_video_path = os.path.join(sample_dir, f"model1_{model1_name}_motion.mp4")
                vis_audio_motion(
                    model1_csv_path,
                    output_path=model1_video_path_temp,
                    audio_path=audio_path,
                    robot_type="g1_brainco",
                    rate_limit=False
                )
                # Add text label to Model 1 video
                add_text_to_video(model1_video_path_temp, model1_video_path, f"Model 1: {model1_name}", position="top-left")
                os.remove(model1_video_path_temp)  # Remove temp file
                print(f"✅ Model 1 motion visualization saved to: {model1_video_path}")
                
                # Visualize Model 2 motion
                print(f"\nCreating Model 2 visualization...")
                model2_video_path_temp = os.path.join(sample_dir, f"model2_{model2_name}_motion_temp.mp4")
                model2_video_path = os.path.join(sample_dir, f"model2_{model2_name}_motion.mp4")
                vis_audio_motion(
                    model2_csv_path,
                    output_path=model2_video_path_temp,
                    audio_path=audio_path,
                    robot_type="g1_brainco",
                    rate_limit=False
                )
                # Add text label to Model 2 video
                add_text_to_video(model2_video_path_temp, model2_video_path, f"Model 2: {model2_name}", position="top-left")
                os.remove(model2_video_path_temp)  # Remove temp file
                print(f"✅ Model 2 motion visualization saved to: {model2_video_path}")
                
                # Create side-by-side comparison of two models with labels
                models_comparison_path = os.path.join(sample_dir, f"models_comparison_{model1_name}_vs_{model2_name}.mp4")
                concat_side_by_side_with_labels(
                    model1_video_path, 
                    model2_video_path, 
                    models_comparison_path, 
                    label_left=f"Model 1: {model1_name}",
                    label_right=f"Model 2: {model2_name}",
                    target_height=720
                )
                print(f"✅ Models comparison video saved to: {models_comparison_path}")
                
                # Also decode ground truth motion tokens for reference
                print(f"\nDecoding ground truth motion tokens...")
                gt_motion_pkl = decode_motion_tokens(
                    motion_tokens_gt, motion_vae, mean_t, std_t
                )
                
                # Save ground truth motion
                gt_pkl_path = os.path.join(sample_dir, "ground_truth_motion.pkl")
                with open(gt_pkl_path, 'wb') as f:
                    pickle.dump(gt_motion_pkl, f)
                
                gt_csv_path = os.path.join(sample_dir, "ground_truth_motion.csv")
                gt_csv = load_motion_pkl_as_csv_data(gt_pkl_path)
                np.savetxt(gt_csv_path, gt_csv, delimiter=',', fmt='%.8f')
                
                # Visualize ground truth motion
                gt_video_path = os.path.join(sample_dir, "ground_truth_motion.mp4")
                vis_audio_motion(
                    gt_csv_path,
                    output_path=gt_video_path,
                    audio_path=audio_path,
                    robot_type="g1_brainco",
                    rate_limit=False
                )
                print(f"✅ Ground truth motion visualization saved to: {gt_video_path}")
                
                # Create three-way comparison (GT, Model1, Model2) with labels
                three_way_comparison_path = os.path.join(sample_dir, f"three_way_comparison_GT_{model1_name}_{model2_name}.mp4")
                concat_three_side_by_side_with_labels(
                    gt_video_path, 
                    model1_video_path, 
                    model2_video_path, 
                    three_way_comparison_path,
                    label1="Ground Truth",
                    label2=f"Model 1: {model1_name}",
                    label3=f"Model 2: {model2_name}",
                    target_height=720
                )
                print(f"✅ Three-way comparison (GT, Model1, Model2) saved to: {three_way_comparison_path}")
                
                processed_count += 1
                print(f"✅ Sample {i+1} processed successfully!")
                continue
            
            elif args.compare_modes:
                print("\n🔄 Comparing teacher forcing vs free generation modes...")
                
                # Teacher forcing mode
                print("\n🎓 Teacher Forcing Mode:")
                tf_tokens, tf_loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=True,
                    ground_truth_tokens=motion_tokens_gt
                )
                
                # Free generation mode
                print("\n🎲 Free Generation Mode:")
                fg_tokens, fg_loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=False
                )
                
                print(f"\n📊 Results Comparison:")
                print(f"   Teacher Forcing Loss: {tf_loss:.6f}")
                print(f"   Free Generation Loss: {fg_loss if fg_loss else 'N/A'}")
                print(f"   Expected Training Loss: ~0.01")
                
                if tf_loss is not None:
                    if abs(tf_loss - 0.01) < 0.005:
                        print(f"   ✅ Teacher forcing loss is close to training loss!")
                    else:
                        print(f"   ⚠️  Teacher forcing loss differs from training loss")
                
                # Use free generation tokens for visualization
                generated_motion_tokens = fg_tokens
                
            elif args.teacher_forcing:
                print("\n🎓 Teacher Forcing Mode:")
                generated_motion_tokens, loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=True,
                    ground_truth_tokens=motion_tokens_gt
                )
                
                print(f"\n📊 Teacher Forcing Results:")
                print(f"   Loss: {loss:.6f}")
                print(f"   Expected Training Loss: ~0.01")
                
                if loss is not None:
                    if abs(loss - 0.01) < 0.005:
                        print(f"   ✅ Loss is close to training loss!")
                    else:
                        print(f"   ⚠️  Loss differs from training loss")
                
            else:
                print("\n🎲 Free Generation Mode:")
                generated_motion_tokens, loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=False
                )

            # Decode generated motion tokens
            print("\nDecoding generated motion tokens...")
            generated_motion_pkl = decode_motion_tokens(
                generated_motion_tokens, motion_vae, mean_t, std_t
            )

            # Save generated motion
            generated_pkl_path = os.path.join(sample_dir, "generated_motion.pkl")
            with open(generated_pkl_path, 'wb') as f:
                pickle.dump(generated_motion_pkl, f)
            print(f"Saved generated motion pkl to: {generated_pkl_path}")

            # Convert to CSV
            generated_csv = load_motion_pkl_as_csv_data(generated_pkl_path)
            # 应用平滑处理，处理突变和振荡
            generated_csv = smooth_qpos_data(generated_csv)
            generated_csv_path = os.path.join(sample_dir, "generated_motion.csv")
            np.savetxt(generated_csv_path, generated_csv, delimiter=',', fmt='%.8f')
            print(f"Saved generated motion CSV to: {generated_csv_path}")

            # Decode ground truth motion tokens
            print("\nDecoding ground truth motion tokens...")
            gt_motion_pkl = decode_motion_tokens(
                motion_tokens_gt, motion_vae, mean_t, std_t
            )

            # Save ground truth motion
            gt_pkl_path = os.path.join(sample_dir, "ground_truth_motion.pkl")
            with open(gt_pkl_path, 'wb') as f:
                pickle.dump(gt_motion_pkl, f)

            # Convert to CSV
            gt_csv_path = os.path.join(sample_dir, "ground_truth_motion.csv")
            gt_csv = load_motion_pkl_as_csv_data(gt_pkl_path)
            np.savetxt(gt_csv_path, gt_csv, delimiter=',', fmt='%.8f')
            print(f"Saved ground truth motion CSV to: {gt_csv_path}")

            # Visualize generated motion
            print("\nCreating visualization...")
            generated_video_path = os.path.join(sample_dir, "generated_motion.mp4")
            vis_audio_motion(
                generated_csv_path,
                output_path=generated_video_path,
                audio_path=audio_path,
                robot_type="g1_brainco",
                rate_limit=False
            )
            print(f"✅ Generated motion visualization saved to: {generated_video_path}")

            # Visualize ground truth motion
            gt_video_path = os.path.join(sample_dir, "ground_truth_motion.mp4")
            vis_audio_motion(
                gt_csv_path,
                output_path=gt_video_path,
                audio_path=audio_path,
                robot_type="g1_brainco",
                rate_limit=False
            )
            print(f"✅ Ground truth motion visualization saved to: {gt_video_path}")

            # Create side-by-side comparison
            comparison_video_path = os.path.join(sample_dir, "comparison.mp4")
            concat_side_by_side(gt_video_path, generated_video_path, comparison_video_path, target_height=720)
            print(f"✅ Comparison video saved to: {comparison_video_path}")

            processed_count += 1
            print(f"✅ Sample {i+1} processed successfully!")

        except Exception as e:
            print(f"❌ Error processing sample {i+1}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print(f"🎉 Processing completed!")
    print(f"Successfully processed: {processed_count}/{min(len(lines), args.num_samples)} samples")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
