#!/usr/bin/env python3
"""
测试GPT2 Motion Adaptor模型在JSONL数据上的表现
对比GT、Teacher-forcing和Free-running三种模式的效果

支持：
- 预训练VQ-VAE模型
- Finetune VQ-VAE模型
- 从JSONL文件加载数据
- 三种模式的对比评估

使用示例：
    python HRI_mllm/test/test_adaptor_jsonl_comparison.py \
        --jsonl_path data/BEAT_v2_1110_tokens.jsonl \
        --motion_adaptor_path output/motion_adaptor_10_v4/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_500.pt \
        --vqvae_config g1_vqvae_arbitrary_length_balanced.yaml \
        --vqvae_checkpoint output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt \
        --output_dir ./adaptor_comparison_results \
        --num_samples 5 \
        --temperature 1.8 \
        --top_k 50 \
        --repetition_penalty 1.8

三种模式说明：
1. GT模式：直接使用JSONL中的ground truth motion tokens进行解码，作为参考基准
2. Teacher-forcing模式：使用GT motion tokens构建完整序列，提取模型在每个motion位置的预测，计算准确率
3. Free-running模式：完全自回归生成motion tokens，不依赖GT motion tokens

输出：
- 每个样本的motion pkl文件和csv文件
- 可视化视频（如果音频文件可用）
- 评估结果JSON文件，包含准确率等指标
"""

import os
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import yaml
import random
import shutil
import subprocess
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT
from HRI_mllm.utils.motion_utils.g1ml3d_final import vec_to_data_pkl, feats2datapkl
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion

from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

torch.cuda.set_device(0)


def load_gpt2_model(model_path, device="cuda"):
    """加载训练完成的GPT2 adaptor模型"""
    print(f"Loading GPT2 adaptor model from: {model_path}")
    
    if model_path.endswith('.pt'):
        model = load_gpt2_from_checkpoint(model_path, device)
    else:
        model = load_gpt2_from_transformers(model_path, device)
    
    return model


def load_gpt2_from_checkpoint(checkpoint_path, device="cuda"):
    """从.pt checkpoint文件加载模型"""
    print(f"🔄 Loading from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get('epoch', 0)
    print(f"📊 Checkpoint epoch: {epoch}")
    
    model_config = GPT2Config(
        vocab_size=1034,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    model_config.gesture_start_token_id = 512*2 + 2
    model_config.audio_gesture_start_token_id = 512*2 + 3
    model_config.gesture_end_token_id = 512*2 + 4
    model_config.audio_gesture_end_token_id = 512*2 + 5
    
    model = MixedInputGPT2(model_config, audio_hidden_size=3584)
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print("✅ Loaded model_state from checkpoint")
    else:
        print("⚠️  model_state not found in checkpoint, using random weights")
    
    model.eval()
    model.to(device)
    print(f"✅ GPT2 adaptor model loaded successfully!")
    return model


def load_gpt2_from_transformers(model_path, device="cuda"):
    """从transformers格式目录加载模型"""
    print(f"🔄 Loading from transformers format: {model_path}")
    
    config_path = os.path.join(model_path, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    
    if not hasattr(config, 'gesture_start_token_id'):
        config.gesture_start_token_id = 512*2 + 2
        config.audio_gesture_start_token_id = 512*2 + 3
        config.gesture_end_token_id = 512*2 + 4
        config.audio_gesture_end_token_id = 512*2 + 5
    
    model = MixedInputGPT2(config, audio_hidden_size=3584)
    
    model_path_pytorch = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_path_pytorch):
        state_dict = torch.load(model_path_pytorch, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pytorch_model.bin")
    
    model.eval()
    model.to(device)
    print(f"✅ GPT2 adaptor model loaded successfully!")
    return model


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """解码motion tokens为motion features"""
    # 过滤所有special token IDs，包括motion_empty_token_id
    special_token_ids = {512*2 + 2, 512*2 + 3, 512*2 + 4, 512*2 + 5}  # gesture tokens
    motion_empty_token_id = 512*2 + 7  # motion empty/padding token
    # 只保留有效的motion tokens: [0, 1023]
    filtered_tokens = [t for t in motion_tokens 
                       if t not in special_token_ids 
                       and t != motion_empty_token_id
                       and 0 <= int(t) < 1024]  # 确保token在有效范围内
    if len(filtered_tokens) != len(motion_tokens):
        print(f"⚠️  过滤了 {len(motion_tokens) - len(filtered_tokens)} 个special/invalid token")
        motion_tokens = filtered_tokens
    
    if len(motion_tokens) % 2 != 0:
        motion_tokens = motion_tokens[:-1]
    
    if len(motion_tokens) == 0:
        print("Error: No motion tokens after processing")
        return None
    
    body_tokens = motion_tokens[0::2]
    hand_tokens = motion_tokens[1::2]
    
    body_tokens = [max(0, min(511, int(t))) for t in body_tokens]
    
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
    
    body_tokens_tensor = torch.tensor(body_tokens).unsqueeze(0).to("cuda")
    hand_tokens_tensor = torch.tensor(hand_tokens).unsqueeze(0).to("cuda") - 512
    
    decoded = motion_vae.decode((body_tokens_tensor, hand_tokens_tensor))
    
    if expected_frames is not None and decoded.shape[1] > expected_frames:
        decoded = decoded[:, :expected_frames, :]
    
    data_dict = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())
    
    return data_dict


def generate_motion_tokens_free_running(model, audio_tokens, device="cuda", max_new_tokens=512, 
                                       temperature=0.8, top_k=50, repetition_penalty=1.1, enable_model_debug=False,
                                       save_attention_weights=False):
    """Free-running模式：完全自回归生成motion tokens
    
    返回:
        generated_motion_tokens: 生成的motion tokens列表
        attention_weights_list: 第一层attention weights列表（如果save_attention_weights=True）
    """
    model.eval()
    
    # 获取模型的最大序列长度（n_positions），应该与训练时的max_seq_length一致
    max_seq_length = getattr(model.config, 'n_positions', 512)
    
    gesture_start_token_id = getattr(model.config, 'gesture_start_token_id', 512*2 + 2)
    audio_gesture_start_token_id = getattr(model.config, 'audio_gesture_start_token_id', 512*2 + 3)
    gesture_end_token_id = getattr(model.config, 'gesture_end_token_id', 512*2 + 4)
    audio_gesture_end_token_id = getattr(model.config, 'audio_gesture_end_token_id', 512*2 + 5)
    
    # 训练时的padding token IDs
    audio_empty_token_id = 152063  # glm-voice-4 audio tokenizer的padding token
    motion_empty_token_id = 512*2 + 7  # motion empty token
    
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    # 过滤输入中的special token
    audio_tokens = [t for t in audio_tokens if t not in special_token_ids]
    
    # 按照训练时的格式：在audio tokens后面添加10个audio_empty_token
    audio_tokens = audio_tokens + [audio_empty_token_id] * 10
    

    
    generated_motion_tokens = []
    current_seq = []
    token_labels = []
    generated_history = []
    attention_weights_list = []  # 收集第一层的attention weights
    
    interleave_audios, interleave_motions = 1, 1
    
    # 按照训练时的格式：前10个audio token对应motion_empty_token
    # 之后的audio token对应真实的motion token
    motion_padding_count = 0
    max_motion_padding = 10
    
    with torch.no_grad():
        for i, audio_token in enumerate(audio_tokens):
            current_seq.append(audio_token)
            token_labels.append(-100)
            
            if (i + 1) % interleave_audios == 0:
                # 前10个audio token位置，添加motion_empty_token
                # 之后的audio token位置，生成真实的motion token
                if motion_padding_count < max_motion_padding:
                    # 添加motion_empty_token
                    current_seq.append(motion_empty_token_id)
                    token_labels.append(-100)  # padding token不计算损失
                    motion_padding_count += 1
                else:
                    # 生成真实的motion token
                    for j in range(interleave_motions):
                        if len(generated_motion_tokens) >= max_new_tokens:
                            break
                        
                        # 检查序列长度是否超过模型最大长度
                        if len(current_seq) >= max_seq_length:
                            print(f"⚠️  Warning: Sequence length ({len(current_seq)}) reached model max_seq_length ({max_seq_length}). Stopping generation.")
                            break
                        
                        inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
                        # 如果序列长度超过max_seq_length，截断到最大长度
                        if inputs.shape[1] > max_seq_length:
                            inputs = inputs[:, -max_seq_length:]
                            attn_mask = torch.ones_like(inputs)
                            labels = torch.tensor(token_labels[-max_seq_length:]).unsqueeze(0).to(device)
                        else:
                            attn_mask = torch.ones_like(inputs)
                            labels = torch.tensor(token_labels).unsqueeze(0).to(device)
                        
                        # 注意：在生成过程中不提取attention weights，因为每次只生成一个token
                        # 我们会在生成完成后用完整序列重新forward一次来获取完整的attention matrix
                        output = model(
                            input_data=inputs, 
                            attention_mask=attn_mask, 
                            labels=labels, 
                            label_tokens=None,
                            use_label_prediction_mode=False
                        )
                        
                        # 检查logits是否存在
                        if output.logits is None:
                            raise ValueError("Model output logits is None. This should not happen in free-running mode.")
                        next_token_logits = output.logits[0, -1, :]
                        
                        # 在采样前排除motion_empty_token_id，避免生成padding token
                        if motion_empty_token_id < next_token_logits.size(-1):
                            next_token_logits[motion_empty_token_id] = float('-inf')
                        
                        if repetition_penalty != 1.0 and generated_history:
                            for token_id in set(generated_history):
                                if token_id not in special_token_ids and token_id < next_token_logits.size(-1):
                                    next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                        # 采样策略（稳健处理 temperature 与 top_k）
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
                            # temperature<=0 时，使用贪心选择，避免除零
                            if top_k is not None and top_k > 0:
                                k = min(int(top_k), next_token_logits.size(-1))
                                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                                masked = torch.full_like(next_token_logits, float('-inf'))
                                masked[top_k_indices] = top_k_logits
                                next_token_logits = masked
                            next_token = torch.argmax(next_token_logits, dim=-1).item()
                        
                        tokens_to_add = []
                        tokens_labels_to_add = []
                        
                        if next_token == gesture_start_token_id:
                            tokens_to_add.append(gesture_start_token_id)
                            tokens_labels_to_add.append(gesture_start_token_id)
                            tokens_to_add.append(audio_gesture_start_token_id)
                            tokens_labels_to_add.append(-100)
                        elif next_token == gesture_end_token_id:
                            tokens_to_add.append(gesture_end_token_id)
                            tokens_labels_to_add.append(gesture_end_token_id)
                            tokens_to_add.append(audio_gesture_end_token_id)
                            tokens_labels_to_add.append(-100)
                        elif next_token in special_token_ids:
                            tokens_to_add.append(next_token)
                            tokens_labels_to_add.append(-100)
                        else:
                            vocab_size = model.config.vocab_size
                            if next_token >= vocab_size:
                                next_token = vocab_size - 1
                            
                            tokens_to_add.append(next_token)
                            tokens_labels_to_add.append(next_token)
                            
                            # 跳过motion_empty_token，因为它是padding token，不应该被当作有效的motion token
                            # 但仍然需要添加到current_seq中，以便模型继续生成
                            if next_token != motion_empty_token_id:
                                generated_motion_tokens.append(next_token)
                                generated_history.append(next_token)
                        
                        for token, label in zip(tokens_to_add, tokens_labels_to_add):
                            current_seq.append(token)
                            token_labels.append(label)
                        
                        if len(generated_history) > 100:
                            generated_history = generated_history[-100:]
                        
                        if len(generated_motion_tokens) >= max_new_tokens:
                            break
                
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    # 如果需要保存attention weights，用完整序列重新forward一次以获取完整的attention matrix
    if save_attention_weights and len(current_seq) > 0:
        print("   [Attention] Forwarding complete sequence to get full attention weights...")
        try:
            full_inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
            full_attn_mask = torch.ones_like(full_inputs)
            full_labels = torch.tensor(token_labels).unsqueeze(0).to(device)
            
            full_output = model(
                input_data=full_inputs,
                attention_mask=full_attn_mask,
                labels=full_labels,
                output_attentions=True
            )
            
            # 提取第一层的attention weights
            if hasattr(full_output, 'attentions') and full_output.attentions is not None:
                if len(full_output.attentions) > 0:
                    first_layer_attn = full_output.attentions[0]  # [batch_size, num_heads, seq_len, seq_len]
                    # 平均所有head的attention weights
                    first_layer_attn_mean = first_layer_attn[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
                    attention_weights_list = [first_layer_attn_mean]  # 替换为完整的attention matrix
                    print(f"   [Attention] Extracted full attention matrix: shape {first_layer_attn_mean.shape}")
        except Exception as e:
            print(f"   ⚠️  Failed to extract full attention weights: {e}")
            attention_weights_list = []
    
    # 过滤special token和motion_empty_token
    filtered_tokens = [t for t in generated_motion_tokens 
                      if t not in special_token_ids and t != motion_empty_token_id]
    if save_attention_weights:
        return filtered_tokens, attention_weights_list
    else:
        return filtered_tokens


def generate_motion_tokens_teacher_forcing(model, audio_tokens, motion_tokens_gt, device="cuda", enable_model_debug=False):
    """Teacher-forcing模式：使用GT motion tokens构建完整序列，提取模型在每个motion位置的预测
    
    支持滑动窗口处理超过512的序列
    
    返回:
        predicted_tokens: 预测的motion tokens列表
        gt_tokens: GT motion tokens列表
    """
    model.eval()
    
    # 获取模型的最大序列长度
    max_seq_length = getattr(model.config, 'n_positions', 512)
    
    gesture_start_token_id = getattr(model.config, 'gesture_start_token_id', 512*2 + 2)
    audio_gesture_start_token_id = getattr(model.config, 'audio_gesture_start_token_id', 512*2 + 3)
    gesture_end_token_id = getattr(model.config, 'gesture_end_token_id', 512*2 + 4)
    audio_gesture_end_token_id = getattr(model.config, 'audio_gesture_end_token_id', 512*2 + 5)
    
    # 训练时的padding token IDs
    audio_empty_token_id = 152063  # glm-voice-4 audio tokenizer的padding token
    motion_empty_token_id = 512*2 + 7  # motion empty token
    
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    # 过滤输入中的special token
    audio_tokens_clean = [t for t in audio_tokens if t not in special_token_ids]
    motion_tokens_gt_clean = [t for t in motion_tokens_gt if t not in special_token_ids]
    
    # 按照训练时的格式：在audio tokens后面添加10个audio_empty_token
    audio_tokens_clean = audio_tokens_clean + [audio_empty_token_id] * 10
    
    # 按照训练时的格式：在motion tokens前面添加10个motion_empty_token
    motion_tokens_gt_clean = [motion_empty_token_id] * 10 + motion_tokens_gt_clean
    
    # 构建完整序列：按照训练时的格式，交替audio和motion tokens
    # 格式：每interleave_audios个audio token后，插入interleave_motions个motion token
    interleave_audios, interleave_motions = 1, 1
    full_sequence = []
    token_labels = []
    motion_positions = []  # 记录motion token在序列中的位置
    
    audio_idx = 0
    motion_idx = 0
    
    # 构建序列，同时记录motion token的位置
    while audio_idx < len(audio_tokens_clean) or motion_idx < len(motion_tokens_gt_clean):
        # 添加interleave_audios个audio token
        for _ in range(interleave_audios):
            if audio_idx < len(audio_tokens_clean):
                full_sequence.append(audio_tokens_clean[audio_idx])
                token_labels.append(-100)  # audio token不计算损失
                audio_idx += 1
            else:
                break
        
        # 每interleave_audios个audio token后，添加interleave_motions个motion token
        for _ in range(interleave_motions):
            if motion_idx < len(motion_tokens_gt_clean):
                full_sequence.append(motion_tokens_gt_clean[motion_idx])
                motion_positions.append(len(full_sequence) - 1)  # 记录motion token位置
                token_labels.append(motion_tokens_gt_clean[motion_idx])  # motion token计算损失
                motion_idx += 1
            else:
                break
        
        # 如果audio和motion tokens都用完了，停止
        if audio_idx >= len(audio_tokens_clean) and motion_idx >= len(motion_tokens_gt_clean):
            break
    
    if len(full_sequence) == 0:
        return [], []
    
    # 检查序列长度，如果超过max_seq_length，使用滑动窗口处理
    if len(full_sequence) <= max_seq_length:
        # 序列长度在限制内，直接处理
        with torch.no_grad():
            inputs = torch.tensor(full_sequence).unsqueeze(0).to(device)
            attn_mask = torch.ones_like(inputs)
            labels = torch.tensor(token_labels).unsqueeze(0).to(device)
            
            # 前向传播
            output = model(
                input_data=inputs, 
                attention_mask=attn_mask, 
                labels=labels, 
                label_tokens=None,
                use_label_prediction_mode=False
            )
            
            # 检查logits是否存在
            if output.logits is None:
                raise ValueError("Model output logits is None. This should not happen in teacher-forcing mode.")
            logits = output.logits[0]  # [seq_len, vocab_size]
            
            # 提取motion token位置的预测
            predicted_tokens = []
            gt_tokens = []
            
            for pos in motion_positions:
                if pos < logits.shape[0]:
                    # 使用位置pos的logits来预测该位置的token
                    pred_logits = logits[pos]
                    predicted_token = torch.argmax(pred_logits, dim=-1).item()
                    gt_token = full_sequence[pos]
                    
                    # 过滤special token和padding token
                    if (predicted_token not in special_token_ids and 
                        gt_token not in special_token_ids and
                        predicted_token != motion_empty_token_id and
                        gt_token != motion_empty_token_id):
                        predicted_tokens.append(predicted_token)
                        gt_tokens.append(gt_token)
            
            return predicted_tokens, gt_tokens
    else:
        # 序列长度超过限制，使用滑动窗口处理
        print(f"⚠️  序列长度 ({len(full_sequence)}) 超过模型最大长度 ({max_seq_length})，使用滑动窗口处理")
        
        # 滑动窗口参数
        window_size = max_seq_length
        overlap_size = max_seq_length // 4  # 25%重叠，确保连续性
        stride = window_size - overlap_size
        
        predicted_tokens = []
        gt_tokens = []
        processed_positions = set()  # 记录已处理的motion位置，避免重复
        
        with torch.no_grad():
            start_idx = 0
            window_idx = 0
            
            while start_idx < len(full_sequence):
                end_idx = min(start_idx + window_size, len(full_sequence))
                window_sequence = full_sequence[start_idx:end_idx]
                window_labels = token_labels[start_idx:end_idx]
                
                # 找到当前窗口内的motion位置（相对于全局序列的位置）
                window_motion_positions = [
                    pos for pos in motion_positions 
                    if start_idx <= pos < end_idx and pos not in processed_positions
                ]
                
                if len(window_motion_positions) == 0:
                    # 当前窗口没有motion token，跳过
                    start_idx += stride
                    window_idx += 1
                    continue
                
                # 处理当前窗口
                inputs = torch.tensor(window_sequence).unsqueeze(0).to(device)
                attn_mask = torch.ones_like(inputs)
                labels = torch.tensor(window_labels).unsqueeze(0).to(device)
                
                # 前向传播
                output = model(
                    input_data=inputs, 
                    attention_mask=attn_mask, 
                    labels=labels, 
                    label_tokens=None,
                    use_label_prediction_mode=False
                )
                
                if output.logits is None:
                    print(f"⚠️  窗口 {window_idx} 的logits为None，跳过")
                    start_idx += stride
                    window_idx += 1
                    continue
                
                logits = output.logits[0]  # [window_seq_len, vocab_size]
                
                # 提取当前窗口内motion token位置的预测
                for global_pos in window_motion_positions:
                    if global_pos in processed_positions:
                        continue
                    
                    # 转换为窗口内的相对位置
                    local_pos = global_pos - start_idx
                    
                    if local_pos < logits.shape[0]:
                        # 使用位置local_pos的logits来预测该位置的token
                        pred_logits = logits[local_pos]
                        predicted_token = torch.argmax(pred_logits, dim=-1).item()
                        gt_token = full_sequence[global_pos]
                        
                        # 过滤special token和padding token
                        if (predicted_token not in special_token_ids and 
                            gt_token not in special_token_ids and
                            predicted_token != motion_empty_token_id and
                            gt_token != motion_empty_token_id):
                            predicted_tokens.append(predicted_token)
                            gt_tokens.append(gt_token)
                            processed_positions.add(global_pos)
                
                # 移动到下一个窗口
                start_idx += stride
                window_idx += 1
        
        print(f"✅ 滑动窗口处理完成，共处理 {window_idx} 个窗口，提取了 {len(predicted_tokens)} 个motion token预测")
        
        # 确保预测的tokens和GT tokens按顺序对应
        # 由于我们按motion_positions的顺序处理，应该已经是按顺序的
        return predicted_tokens, gt_tokens


def load_jsonl_data(jsonl_path, max_samples=None):
    """从JSONL文件加载数据"""
    samples = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if max_samples and len(samples) >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                
                audio_tokens = None
                motion_tokens = None
                audio_path = None
                
                for msg in data.get('conversation', []):
                    if msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                        audio_tokens = msg['audio_tokens']
                        audio_path = msg.get('content', None)
                    elif msg.get('message_type') == 'audio_motion' and 'motion_tokens' in msg:
                        motion_tokens = msg['motion_tokens']
                
                if audio_tokens is None or motion_tokens is None:
                    continue
                
                # 转换为list
                if isinstance(audio_tokens, torch.Tensor):
                    audio_tokens = audio_tokens.tolist()
                if isinstance(motion_tokens, torch.Tensor):
                    motion_tokens = motion_tokens.tolist()
                
                samples.append({
                    'audio_tokens': audio_tokens,
                    'motion_tokens': motion_tokens,
                    'audio_path': audio_path,
                    'sample_idx': line_num
                })
                
            except Exception as e:
                print(f"⚠️  Error loading line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(samples)} samples from {jsonl_path}")
    return samples


def compute_token_accuracy(predicted, ground_truth):
    """计算token级别的准确率"""
    if len(predicted) == 0 or len(ground_truth) == 0:
        return 0.0
    
    min_len = min(len(predicted), len(ground_truth))
    predicted = predicted[:min_len]
    ground_truth = ground_truth[:min_len]
    
    correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return correct / min_len if min_len > 0 else 0.0


def visualize_attention_weights(attention_weights_list, output_path, token_sequence=None, title="First Layer Attention Weights"):
    """
    可视化第一层的attention weights
    
    参数:
        attention_weights_list: attention weights列表，每个元素是[seq_len, seq_len]的numpy数组
        output_path: 输出图片路径
        token_sequence: token序列（可选），用于标记奇数和偶数位置
        title: 图片标题
    """
    if not attention_weights_list:
        print("⚠️  No attention weights to visualize")
        return
    
    # 合并所有attention weights（取最后一个，因为它包含了完整的序列信息）
    # 或者我们可以可视化每个生成步骤的attention
    # 这里我们可视化最后一个attention weights（包含完整序列）
    final_attn = attention_weights_list[-1]  # [seq_len, seq_len]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：完整的attention matrix
    ax1 = axes[0]
    im1 = ax1.imshow(final_attn, cmap='viridis', aspect='auto', interpolation='nearest')
    ax1.set_title(f'{title} - Full Sequence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position (Token Index)', fontsize=12)
    ax1.set_ylabel('Query Position (Token Index)', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Attention Weight')
    
    # 标记奇数和偶数位置（如果提供了token序列）
    if token_sequence is not None and len(token_sequence) <= final_attn.shape[0]:
        # 在x轴上标记
        odd_positions = [i for i in range(len(token_sequence)) if i % 2 == 0]  # 偶数索引（0, 2, 4...）对应奇数位置（1st, 3rd, 5th...）
        even_positions = [i for i in range(len(token_sequence)) if i % 2 == 1]  # 奇数索引（1, 3, 5...）对应偶数位置（2nd, 4th, 6th...）
        
        # 添加垂直分割线
        for pos in odd_positions[1:]:  # 跳过第一个
            ax1.axvline(x=pos-0.5, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        for pos in even_positions:
            ax1.axvline(x=pos-0.5, color='blue', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # 添加水平分割线
        for pos in odd_positions[1:]:
            ax1.axhline(y=pos-0.5, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
        for pos in even_positions:
            ax1.axhline(y=pos-0.5, color='blue', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # 右图：分析奇数和偶数位置的attention模式
    ax2 = axes[1]
    
    # 计算每个query位置对奇数位置和偶数位置的attention总和
    seq_len = final_attn.shape[0]
    odd_attention = np.zeros(seq_len)
    even_attention = np.zeros(seq_len)
    
    for query_pos in range(seq_len):
        # 奇数位置（索引0, 2, 4...）
        odd_indices = [i for i in range(seq_len) if i % 2 == 0]
        # 偶数位置（索引1, 3, 5...）
        even_indices = [i for i in range(seq_len) if i % 2 == 1]
        
        if odd_indices:
            odd_attention[query_pos] = final_attn[query_pos, odd_indices].sum()
        if even_indices:
            even_attention[query_pos] = final_attn[query_pos, even_indices].sum()
    
    x_positions = np.arange(seq_len)
    width = 0.35
    ax2.bar(x_positions - width/2, odd_attention, width, label='Attention to Odd Positions (1st, 3rd, 5th...)', alpha=0.7, color='red')
    ax2.bar(x_positions + width/2, even_attention, width, label='Attention to Even Positions (2nd, 4th, 6th...)', alpha=0.7, color='blue')
    ax2.set_xlabel('Query Position (Token Index)', fontsize=12)
    ax2.set_ylabel('Total Attention Weight', fontsize=12)
    ax2.set_title('Attention Distribution: Odd vs Even Positions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 标记奇数和偶数位置
    for i in range(0, seq_len, 2):
        ax2.axvline(x=i, color='red', linestyle=':', alpha=0.2, linewidth=0.5)
    for i in range(1, seq_len, 2):
        ax2.axvline(x=i, color='blue', linestyle=':', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Attention weights visualization saved to: {output_path}")
    
    # 打印统计信息
    print(f"\n📊 Attention Statistics:")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Average attention to odd positions: {odd_attention.mean():.4f}")
    print(f"   - Average attention to even positions: {even_attention.mean():.4f}")
    print(f"   - Ratio (odd/even): {odd_attention.mean() / even_attention.mean() if even_attention.mean() > 0 else 'N/A'}")


def save_attention_weights_numpy(attention_weights_list, output_path):
    """保存attention weights为numpy文件"""
    if not attention_weights_list:
        print("⚠️  No attention weights to save")
        return
    
    # 保存所有attention weights
    attention_array = np.array(attention_weights_list)  # [num_steps, seq_len, seq_len]
    np.save(output_path, attention_array)
    print(f"✅ Attention weights saved to: {output_path}")


def merge_five_videos(v0, v1, v2, v3, v4, out_path, layout="hstack", height=720, crf=18,
                      preset="veryfast", copy_first_audio=True,
                      label0="GT", label1="Teacher-Forcing", label2="FR-Label=0", label3="FR-GT-Label", label4="FR-Pred-Label",
                      fontfile=None, fontsize=36, fontcolor="white",
                      box=True, boxcolor="black@0.5", boxborderw=10):
    """
    使用ffmpeg将五个视频横向合并，并在每个视频上添加文字标签
    
    参数:
        v0, v1, v2, v3, v4: 五个输入视频路径
        out_path: 输出视频路径
        layout: 布局方式，"hstack"表示横向排列，"vstack"表示纵向排列
        height: 输出视频高度
        crf: 视频质量参数，值越小质量越高（默认18）
        preset: 编码速度预设（ultrafast, veryfast, fast, medium, slow等）
        copy_first_audio: 是否使用第一个视频的音频
        label0, label1, label2, label3, label4: 五个视频的标签文字
        fontfile: 字体文件路径（可选）
        fontsize: 字体大小（默认36）
        fontcolor: 字体颜色（默认白色）
        box: 是否在文字周围添加半透明背景框（默认True）
        boxcolor: 背景框颜色（默认黑色半透明）
        boxborderw: 背景框边框宽度（默认10）
    """
    def build_drawtext_filter(label_text):
        """构建drawtext滤镜参数"""
        dt_params = [
            f"text='{label_text}'",
            f"fontsize={fontsize}",
            f"fontcolor={fontcolor}",
            "x=10",
            "y=10"
        ]
        
        if box:
            dt_params.append(f"box=1:boxcolor={boxcolor}:boxborderw={boxborderw}")
        
        if fontfile:
            dt_params.append(f"fontfile={fontfile}")
        
        return "drawtext=" + ":".join(dt_params)
    
    # 构建ffmpeg命令
    if layout == "hstack":
        # 横向合并
        filter_complex = (
            f"[0:v]scale=-1:{height}[v0scaled];"
            f"[1:v]scale=-1:{height}[v1scaled];"
            f"[2:v]scale=-1:{height}[v2scaled];"
            f"[3:v]scale=-1:{height}[v3scaled];"
            f"[4:v]scale=-1:{height}[v4scaled];"
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v2scaled]{build_drawtext_filter(label2)}[v2text];"
            f"[v3scaled]{build_drawtext_filter(label3)}[v3text];"
            f"[v4scaled]{build_drawtext_filter(label4)}[v4text];"
            f"[v0text][v1text][v2text][v3text][v4text]hstack=inputs=5[v]"
        )
    elif layout == "vstack":
        # 纵向合并
        filter_complex = (
            f"[0:v]scale=-1:{height}[v0scaled];"
            f"[1:v]scale=-1:{height}[v1scaled];"
            f"[2:v]scale=-1:{height}[v2scaled];"
            f"[3:v]scale=-1:{height}[v3scaled];"
            f"[4:v]scale=-1:{height}[v4scaled];"
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v2scaled]{build_drawtext_filter(label2)}[v2text];"
            f"[v3scaled]{build_drawtext_filter(label3)}[v3text];"
            f"[v4scaled]{build_drawtext_filter(label4)}[v4text];"
            f"[v0text][v1text][v2text][v3text][v4text]vstack=inputs=5[v]"
        )
    else:
        raise ValueError(f"不支持的布局方式: {layout}")
    
    # 构建完整的ffmpeg命令
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-i", v0,
        "-i", v1,
        "-i", v2,
        "-i", v3,
        "-i", v4,
        "-filter_complex", filter_complex,
        "-map", "[v]",
    ]
    
    if copy_first_audio:
        cmd.extend(["-map", "0:a?"])  # 使用第一个视频的音频（如果存在）
    
    cmd.extend([
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        out_path
    ])
    
    # 执行ffmpeg命令
    print(f"正在合并5个视频...")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"✅ 视频合并成功: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg执行失败:")
        print(f"错误信息: {e.stderr}")
        raise


def merge_four_videos(v0, v1, v2, v3, out_path, layout="hstack", height=720, crf=18,
                      preset="veryfast", copy_first_audio=True,
                      label0="GT", label1="Teacher-Forcing", label2="Free-Running", label3="FR-GT-Label",
                      fontfile=None, fontsize=36, fontcolor="white",
                      box=True, boxcolor="black@0.5", boxborderw=10):
    """
    使用ffmpeg将四个视频横向合并，并在每个视频上添加文字标签
    
    参数:
        v0, v1, v2, v3: 四个输入视频路径
        out_path: 输出视频路径
        layout: 布局方式，"hstack"表示横向排列，"vstack"表示纵向排列
        height: 输出视频高度
        crf: 视频质量参数，值越小质量越高（默认18）
        preset: 编码速度预设（ultrafast, veryfast, fast, medium, slow等）
        copy_first_audio: 是否使用第一个视频的音频
        label0, label1, label2, label3: 四个视频的标签文字
        fontfile: 字体文件路径，如果为None则使用系统默认字体
        fontsize: 字体大小
        fontcolor: 字体颜色
        box: 是否添加文字背景框
        boxcolor: 背景框颜色
        boxborderw: 背景框边框宽度
    """
    # 检查输入文件是否存在
    for video_path in [v0, v1, v2, v3]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 构建drawtext滤镜参数
    def build_drawtext_filter(text):
        """构建drawtext滤镜字符串"""
        dt_params = [
            f"text='{text}'",
            f"fontsize={fontsize}",
            f"fontcolor={fontcolor}",
            "x=(w-text_w)/2",  # 水平居中
            "y=30"  # 距离顶部30像素
        ]
        
        if fontfile and os.path.exists(fontfile):
            dt_params.append(f"fontfile='{fontfile}'")
        
        if box:
            dt_params.extend([
                "box=1",
                f"boxcolor={boxcolor}",
                f"boxborderw={boxborderw}"
            ])
        
        return "drawtext=" + ":".join(dt_params)
    
    # 构建ffmpeg命令
    if layout == "hstack":
        # 横向合并
        filter_complex = (
            f"[0:v]scale=-1:{height}[v0scaled];"
            f"[1:v]scale=-1:{height}[v1scaled];"
            f"[2:v]scale=-1:{height}[v2scaled];"
            f"[3:v]scale=-1:{height}[v3scaled];"
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v2scaled]{build_drawtext_filter(label2)}[v2text];"
            f"[v3scaled]{build_drawtext_filter(label3)}[v3text];"
            f"[v0text][v1text][v2text][v3text]hstack=inputs=4[v]"
        )
    else:
        # 纵向合并
        filter_complex = (
            f"[0:v]scale=-1:{height}[v0scaled];"
            f"[1:v]scale=-1:{height}[v1scaled];"
            f"[2:v]scale=-1:{height}[v2scaled];"
            f"[3:v]scale=-1:{height}[v3scaled];"
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v2scaled]{build_drawtext_filter(label2)}[v2text];"
            f"[v3scaled]{build_drawtext_filter(label3)}[v3text];"
            f"[v0text][v1text][v2text][v3text]vstack=inputs=4[v]"
        )
    
    # 构建完整的ffmpeg命令
    cmd = [
        "ffmpeg",
        "-i", v0,
        "-i", v1,
        "-i", v2,
        "-i", v3,
        "-filter_complex", filter_complex,
        "-map", "[v]",
    ]
    
    if copy_first_audio:
        cmd.extend(["-map", "0:a?"])  # 使用第一个视频的音频（如果存在）
    
    cmd.extend([
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",  # 覆盖输出文件
        out_path
    ])
    
    # 执行ffmpeg命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg合并视频失败: {result.stderr}")
    
    print(f"✅ 合并视频成功: {out_path}")


def merge_three_videos(v0, v1, v2, out_path, layout="hstack", height=720, crf=18, 
                       preset="veryfast", copy_first_audio=True, 
                       label0="GT", label1="Teacher-Forcing", label2="Free-Running",
                       fontfile=None, fontsize=36, fontcolor="white",
                       box=True, boxcolor="black@0.5", boxborderw=10):
    """
    使用ffmpeg将三个视频横向合并，并在每个视频上添加文字标签
    
    参数:
        v0, v1, v2: 三个输入视频路径
        out_path: 输出视频路径
        layout: 布局方式，"hstack"表示横向排列，"vstack"表示纵向排列
        height: 输出视频高度
        crf: 视频质量参数，值越小质量越高（默认18）
        preset: 编码速度预设（ultrafast, veryfast, fast, medium, slow等）
        copy_first_audio: 是否使用第一个视频的音频
        label0, label1, label2: 三个视频的标签文字
        fontfile: 字体文件路径，如果为None则使用系统默认字体
        fontsize: 字体大小
        fontcolor: 字体颜色
        box: 是否添加文字背景框
        boxcolor: 背景框颜色
        boxborderw: 背景框边框宽度
    """
    # 检查输入文件是否存在
    for video_path in [v0, v1, v2]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 构建drawtext滤镜参数
    def build_drawtext_filter(text):
        """构建drawtext滤镜字符串"""
        dt_params = [
            f"text='{text}'",
            f"fontsize={fontsize}",
            f"fontcolor={fontcolor}",
            "x=(w-text_w)/2",  # 水平居中
            "y=30"  # 距离顶部30像素
        ]
        
        if fontfile and os.path.exists(fontfile):
            dt_params.append(f"fontfile='{fontfile}'")
        
        if box:
            dt_params.extend([
                "box=1",
                f"boxcolor={boxcolor}",
                f"boxborderw={boxborderw}"
            ])
        
        return "drawtext=" + ":".join(dt_params)
    
    # 构建ffmpeg命令
    if layout == "hstack":
        # 横向合并
        filter_complex = (
            f"[0:v]scale=-1:{height}[v0scaled];"
            f"[1:v]scale=-1:{height}[v1scaled];"
            f"[2:v]scale=-1:{height}[v2scaled];"
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v2scaled]{build_drawtext_filter(label2)}[v2text];"
            f"[v0text][v1text][v2text]hstack=inputs=3[outv]"
        )
    elif layout == "vstack":
        # 纵向合并
        filter_complex = (
            f"[0:v]scale=-1:{height}[v0scaled];"
            f"[1:v]scale=-1:{height}[v1scaled];"
            f"[2:v]scale=-1:{height}[v2scaled];"
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v2scaled]{build_drawtext_filter(label2)}[v2text];"
            f"[v0text][v1text][v2text]vstack=inputs=3[outv]"
        )
    else:
        raise ValueError(f"不支持的布局方式: {layout}")
    
    # 构建完整的ffmpeg命令
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-i", v0,
        "-i", v1,
        "-i", v2,
        "-filter_complex", filter_complex,
        "-map", "[outv]"
    ]
    
    # 添加音频
    if copy_first_audio:
        cmd.extend(["-map", "0:a?"])  # 复制第一个视频的音频（如果存在）
    
    # 添加编码参数
    cmd.extend([
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "192k",
        out_path
    ])
    
    # 执行ffmpeg命令
    print(f"正在合并视频...")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"✅ 视频合并成功: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg执行失败:")
        print(f"错误信息: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Test GPT2 Adaptor on JSONL data with comparison')
    parser.add_argument('--jsonl_path', type=str, required=True,
                       help='Path to JSONL file containing training data')
    parser.add_argument('--motion_adaptor_path', type=str, required=True,
                       help='Path to motion adaptor checkpoint')
    parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='VQ-VAE config file name')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                       help='VQ-VAE checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./adaptor_comparison_results',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
    parser.add_argument('--temperature', type=float, default=1.8,
                       help='Temperature for free-running generation')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k for free-running generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.8,
                       help='Repetition penalty for free-running generation')
    parser.add_argument('--max_motion_tokens', type=int, default=512,
                       help='Maximum motion tokens to generate (should match model max_seq_length=512)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--enable_model_debug', action='store_true',
                       help='Enable model internal debug mode to diagnose label_logits issues')
    parser.add_argument('--save_attention_weights', action='store_true',
                       help='Save and visualize first layer attention weights during generation')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载GPT2 adaptor模型
    print("=" * 70)
    print("Loading GPT2 Adaptor Model")
    print("=" * 70)
    motion_adaptor = load_gpt2_model(args.motion_adaptor_path)
    
    # 加载VQ-VAE模型
    print("\n" + "=" * 70)
    print("Loading VQ-VAE Model")
    print("=" * 70)
    
    def open_yaml(path):
        with open(path, 'r', encoding="utf-8") as file:
            return yaml.safe_load(file)
    
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
    print(f"Loading VQ-VAE config from: {config_path}")
    motion_config = open_yaml(config_path)
    
    if args.vqvae_checkpoint:
        checkpoint_path = args.vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        finetuned_checkpoint = "output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt"
        pretrained_checkpoint = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
        
        if os.path.exists(finetuned_checkpoint):
            checkpoint_path = finetuned_checkpoint
        elif os.path.exists(pretrained_checkpoint):
            checkpoint_path = pretrained_checkpoint
        else:
            checkpoint_path = pretrained_checkpoint
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")
    
    test_mean, test_std = load_normalization_stats(motion_config)
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
    std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
    print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")
    
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device="cuda")
    print(f"✅ VQ-VAE model loaded successfully!")
    
    # 加载JSONL数据
    print("\n" + "=" * 70)
    print("Loading JSONL Data")
    print("=" * 70)
    all_samples = load_jsonl_data(args.jsonl_path)
    
    if len(all_samples) == 0:
        print("❌ No samples loaded from JSONL file")
        exit(1)
    
    # 随机选择样本
    if args.num_samples > len(all_samples):
        selected_samples = all_samples
    else:
        selected_samples = random.sample(all_samples, args.num_samples)
    
    print(f"Selected {len(selected_samples)} samples for testing")
    
    # 评估结果
    results = {
        'gt': [],
        'teacher_forcing': [],
        'free_running': []
    }
    
    # 处理每个样本
    for sample_idx, sample in enumerate(selected_samples):
        print("\n" + "=" * 70)
        print(f"Processing Sample {sample_idx + 1}/{len(selected_samples)}")
        print("=" * 70)
        
        audio_tokens = sample['audio_tokens']
        motion_tokens_gt = sample['motion_tokens']
        audio_path = sample['audio_path']
        print(audio_path)
        
        print(f"Audio tokens: {len(audio_tokens)}")
        print(f"Motion tokens (GT): {len(motion_tokens_gt)}")
        
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 1. GT模式：直接使用GT motion tokens
        print("\n--- GT Mode ---")
        try:
            motion_pkl_gt = decode_motion_tokens(motion_tokens_gt, motion_vae, mean_t, std_t)
            if motion_pkl_gt is not None:
                gt_pkl_path = os.path.join(sample_dir, "gt_motion.pkl")
                with open(gt_pkl_path, 'wb') as f:
                    pickle.dump(motion_pkl_gt, f)
                
                gt_csv_path = os.path.join(sample_dir, "gt_motion.csv")
                motion_csv = load_motion_pkl_as_csv_data(gt_pkl_path)
                np.savetxt(gt_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                
                results['gt'].append({
                    'sample_idx': sample_idx,
                    'motion_tokens': len(motion_tokens_gt),
                    'status': 'success'
                })
                print(f"✅ GT motion decoded successfully")
        except Exception as e:
            print(f"❌ GT mode failed: {e}")
            results['gt'].append({
                'sample_idx': sample_idx,
                'status': 'failed',
                'error': str(e)
            })
        
        # 2. Teacher-forcing模式
        print("\n--- Teacher-forcing Mode ---")
        try:
            predicted_tokens_tf, gt_tokens_tf = generate_motion_tokens_teacher_forcing(
                motion_adaptor, audio_tokens, motion_tokens_gt, device="cuda",
                enable_model_debug=args.enable_model_debug
            )
            
            if len(predicted_tokens_tf) > 0:
                # 计算准确率
                accuracy_tf = compute_token_accuracy(predicted_tokens_tf, gt_tokens_tf)
                print(f"Token accuracy (teacher-forcing): {accuracy_tf:.4f}")
                
                # 解码预测的motion tokens
                motion_pkl_tf = decode_motion_tokens(predicted_tokens_tf, motion_vae, mean_t, std_t)
                if motion_pkl_tf is not None:
                    tf_pkl_path = os.path.join(sample_dir, "teacher_forcing_motion.pkl")
                    with open(tf_pkl_path, 'wb') as f:
                        pickle.dump(motion_pkl_tf, f)
                    
                    tf_csv_path = os.path.join(sample_dir, "teacher_forcing_motion.csv")
                    motion_csv = load_motion_pkl_as_csv_data(tf_pkl_path)
                    np.savetxt(tf_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                    
                    results['teacher_forcing'].append({
                        'sample_idx': sample_idx,
                        'accuracy': accuracy_tf,
                        'predicted_tokens': len(predicted_tokens_tf),
                        'gt_tokens': len(gt_tokens_tf),
                        'status': 'success'
                    })
                    print(f"✅ Teacher-forcing motion decoded successfully")
        except Exception as e:
            print(f"❌ Teacher-forcing mode failed: {e}")
            import traceback
            traceback.print_exc()
            results['teacher_forcing'].append({
                'sample_idx': sample_idx,
                'status': 'failed',
                'error': str(e)
            })
        
        # 3. Free-running模式
        print("\n--- Free-running Mode ---")
        try:
            result_fr = generate_motion_tokens_free_running(
                motion_adaptor, audio_tokens, device="cuda",
                max_new_tokens=min(args.max_motion_tokens, len(motion_tokens_gt) * 2),
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                enable_model_debug=args.enable_model_debug,
                save_attention_weights=args.save_attention_weights
            )
            if args.save_attention_weights:
                if len(result_fr) == 2:
                    predicted_tokens_fr, attention_weights_fr = result_fr
                else:
                    predicted_tokens_fr = result_fr
                    attention_weights_fr = []
            else:
                predicted_tokens_fr = result_fr
                attention_weights_fr = []
            
            if len(predicted_tokens_fr) > 0:
                print(f"Generated {len(predicted_tokens_fr)} motion tokens")
                
                # 解码生成的motion tokens
                motion_pkl_fr = decode_motion_tokens(predicted_tokens_fr, motion_vae, mean_t, std_t)
                if motion_pkl_fr is not None:
                    fr_pkl_path = os.path.join(sample_dir, "free_running_motion.pkl")
                    with open(fr_pkl_path, 'wb') as f:
                        pickle.dump(motion_pkl_fr, f)
                    
                    fr_csv_path = os.path.join(sample_dir, "free_running_motion.csv")
                    motion_csv = load_motion_pkl_as_csv_data(fr_pkl_path)
                    np.savetxt(fr_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                    
                    results['free_running'].append({
                        'sample_idx': sample_idx,
                        'generated_tokens': len(predicted_tokens_fr),
                        'status': 'success'
                    })
                    print(f"✅ Free-running motion decoded successfully")
                    
                    # 保存和可视化attention weights
                    if args.save_attention_weights and attention_weights_fr:
                        print("\n--- Saving and Visualizing Attention Weights ---")
                        # 保存numpy文件
                        attn_npy_path = os.path.join(sample_dir, "attention_weights_first_layer.npy")
                        save_attention_weights_numpy(attention_weights_fr, attn_npy_path)
                        
                        # 可视化最后一个attention weights（包含完整序列）
                        # 构建完整的token序列用于标记（包含audio和motion tokens）
                        # 注意：在free-running生成中，序列是交替的audio和motion tokens
                        # 我们需要构建完整的序列来正确标记奇数和偶数位置
                        if len(attention_weights_fr) > 0:
                            attn_shape = attention_weights_fr[-1].shape[0]
                            # 由于我们不知道确切的序列结构，我们假设序列长度就是attention matrix的大小
                            # 在实际使用中，奇数和偶数位置应该对应两类不同的token（body和hand）
                            attn_viz_path = os.path.join(sample_dir, "attention_weights_first_layer.png")
                            visualize_attention_weights(
                                attention_weights_fr, 
                                attn_viz_path,
                                token_sequence=list(range(attn_shape)),  # 使用索引作为标记
                                title=f"First Layer Attention Weights - Sample {sample_idx}\n(Odd positions: 1st, 3rd, 5th... | Even positions: 2nd, 4th, 6th...)"
                            )
        except Exception as e:
            print(f"❌ Free-running mode failed: {e}")
            import traceback
            traceback.print_exc()
            results['free_running'].append({
                'sample_idx': sample_idx,
                'status': 'failed',
                'error': str(e)
            })
        
        # 生成可视化视频（如果有音频文件）
        if audio_path and os.path.exists(audio_path):
            print("\n--- Generating Visualization Videos ---")
            try:
                # 复制音频文件
                audio_copy_path = os.path.join(sample_dir, "audio.wav")
                shutil.copyfile(audio_path, audio_copy_path)
                
                # GT视频
                if os.path.exists(os.path.join(sample_dir, "gt_motion.csv")):
                    vis_audio_motion(
                        os.path.join(sample_dir, "gt_motion.csv"),
                        output_path=os.path.join(sample_dir, "gt_motion.mp4"),
                        audio_path=audio_copy_path,
                        robot_type="g1_brainco",
                        rate_limit=False,
                        motion_fps=25
                    )
                
                # Teacher-forcing视频
                if os.path.exists(os.path.join(sample_dir, "teacher_forcing_motion.csv")):
                    vis_audio_motion(
                        os.path.join(sample_dir, "teacher_forcing_motion.csv"),
                        output_path=os.path.join(sample_dir, "teacher_forcing_motion.mp4"),
                        audio_path=audio_copy_path,
                        robot_type="g1_brainco",
                        rate_limit=False,
                        motion_fps=25
                    )
                
                # Free-running视频
                if os.path.exists(os.path.join(sample_dir, "free_running_motion.csv")):
                    vis_audio_motion(
                        os.path.join(sample_dir, "free_running_motion.csv"),
                        output_path=os.path.join(sample_dir, "free_running_motion.mp4"),
                        audio_path=audio_copy_path,
                        robot_type="g1_brainco",
                        rate_limit=False,
                        motion_fps=25
                    )
                
                print(f"✅ Visualization videos generated")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
    
        # 合并视频进行对比
        try:
            gt_mp4 = os.path.join(sample_dir, "gt_motion.mp4")
            tf_mp4 = os.path.join(sample_dir, "teacher_forcing_motion.mp4")
            fr_mp4 = os.path.join(sample_dir, "free_running_motion.mp4")
            
            # 合并3个视频
            if os.path.exists(gt_mp4) and os.path.exists(tf_mp4) and os.path.exists(fr_mp4):
                combined_out = os.path.join(sample_dir, "combined_comparison.mp4")
                print("\n--- Merging comparison video (GT | TF | FR) ---")
                merge_three_videos(
                    v0=gt_mp4,
                    v1=tf_mp4,
                    v2=fr_mp4,
                    out_path=combined_out,
                    layout="hstack",
                    height=720,
                    crf=18,
                    preset="veryfast",
                    copy_first_audio=True,
                    label0="GT",
                    label1="Teacher-Forcing",
                    label2="Free-Running",
                    fontfile="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    fontsize=36,
                    fontcolor="white",
                    box=True,
                    boxcolor="black@0.5",
                    boxborderw=10,
                )
                print(f"✅ Combined comparison video saved: {combined_out}")
            else:
                print("⚠️  Skip merging: one or more input videos are missing.")
        except Exception as e:
            print(f"⚠️  Merge comparison video failed: {e}")
            import traceback
            traceback.print_exc()

    # 保存评估结果
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    
    gt_success = sum(1 for r in results['gt'] if r.get('status') == 'success')
    tf_success = sum(1 for r in results['teacher_forcing'] if r.get('status') == 'success')
    fr_success = sum(1 for r in results['free_running'] if r.get('status') == 'success')
    
    print(f"GT mode: {gt_success}/{len(selected_samples)} successful")
    print(f"Teacher-forcing mode: {tf_success}/{len(selected_samples)} successful")
    print(f"Free-running mode: {fr_success}/{len(selected_samples)} successful")
    
    if tf_success > 0:
        tf_accuracies = [r['accuracy'] for r in results['teacher_forcing'] if r.get('status') == 'success']
        avg_tf_accuracy = np.mean(tf_accuracies)
        print(f"Average teacher-forcing accuracy: {avg_tf_accuracy:.4f}")
    
    print(f"\n✅ Results saved to: {results_path}")
    print(f"✅ Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

