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


def check_model_has_label_support(model):
    """检查模型是否支持label功能"""
    # 检查是否有label_classifier层
    has_label = hasattr(model, 'label_classifier')
    if has_label:
        print("✅ 模型支持label功能（9分类）")
    else:
        print("ℹ️  模型不支持label功能（旧版本）")
    return has_label


def load_gpt2_model(model_path, device="cuda"):
    """加载训练完成的GPT2 adaptor模型"""
    print(f"Loading GPT2 adaptor model from: {model_path}")
    
    if model_path.endswith('.pt'):
        model = load_gpt2_from_checkpoint(model_path, device)
    else:
        model = load_gpt2_from_transformers(model_path, device)
    
    # 检查模型是否支持label功能
    check_model_has_label_support(model)
    
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
    special_token_ids = {512*2 + 2, 512*2 + 3, 512*2 + 4, 512*2 + 5}
    filtered_tokens = [t for t in motion_tokens if t not in special_token_ids]
    if len(filtered_tokens) != len(motion_tokens):
        print(f"⚠️  过滤了 {len(motion_tokens) - len(filtered_tokens)} 个special token")
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


def generate_motion_tokens_free_running(model, audio_tokens, device="cuda", max_new_tokens=256, 
                                       temperature=0.8, top_k=50, repetition_penalty=1.1, enable_model_debug=False):
    """Free-running模式：完全自回归生成motion tokens
    
    返回:
        generated_motion_tokens: 生成的motion tokens列表
        predicted_labels: 预测的label列表（如果模型支持label功能），否则为None
    """
    model.eval()
    
    # 检查模型是否支持label功能
    has_label_support = hasattr(model, 'label_classifier')
    
    gesture_start_token_id = getattr(model.config, 'gesture_start_token_id', 512*2 + 2)
    audio_gesture_start_token_id = getattr(model.config, 'audio_gesture_start_token_id', 512*2 + 3)
    gesture_end_token_id = getattr(model.config, 'gesture_end_token_id', 512*2 + 4)
    audio_gesture_end_token_id = getattr(model.config, 'audio_gesture_end_token_id', 512*2 + 5)
    
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    # 过滤输入中的special token
    audio_tokens = [t for t in audio_tokens if t not in special_token_ids]
    
    generated_motion_tokens = []
    predicted_labels = []  # 收集预测的labels
    current_seq = []
    token_labels = []
    generated_history = []
    
    interleave_audios, interleave_motions = 1, 1
    
    with torch.no_grad():
        for i, audio_token in enumerate(audio_tokens):
            current_seq.append(audio_token)
            token_labels.append(-100)
            
            if (i + 1) % interleave_audios == 0:
                for j in range(interleave_motions):
                    if len(generated_motion_tokens) >= max_new_tokens:
                        break
                    
                    inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
                    attn_mask = torch.ones_like(inputs)
                    labels = torch.tensor(token_labels).unsqueeze(0).to(device)
                    
                    # 临时启用模型内部的调试模式（仅用于诊断）
                    if enable_model_debug and hasattr(model, 'label_classifier'):
                        model._debug_label_loss = True
                    
                    # 推理时不需要GT label_tokens，传入None
                    output = model(input_data=inputs, attention_mask=attn_mask, labels=labels, label_tokens=None)
                    
                    if enable_model_debug and hasattr(model, 'label_classifier'):
                        model._debug_label_loss = False
                    next_token_logits = output.logits[0, -1, :]
                    
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
                        generated_motion_tokens.append(next_token)
                        generated_history.append(next_token)
                        
                        # 如果模型支持label功能，在生成motion token后，需要再次forward获取该位置的label预测
                        if has_label_support:
                            # 重新forward一次，这次包含新生成的motion token
                            temp_seq = current_seq + [next_token]
                            temp_labels = token_labels + [next_token]
                            temp_inputs = torch.tensor(temp_seq).unsqueeze(0).to(device)
                            temp_attn = torch.ones_like(temp_inputs)
                            temp_labels_tensor = torch.tensor(temp_labels).unsqueeze(0).to(device)
                            # 临时启用模型内部的调试模式（仅用于诊断）
                            if enable_model_debug and hasattr(model, 'label_classifier'):
                                model._debug_label_loss = True
                            
                            temp_output = model(input_data=temp_inputs, attention_mask=temp_attn, 
                                              labels=temp_labels_tensor, label_tokens=None)
                            
                            if enable_model_debug and hasattr(model, 'label_classifier'):
                                model._debug_label_loss = False
                            if hasattr(temp_output, 'label_logits') and temp_output.label_logits is not None:
                                # 取最后一个motion token位置的label预测
                                motion_mask = (temp_labels_tensor[0] != -100)
                                if motion_mask.any():
                                    last_motion_idx = torch.where(motion_mask)[0][-1].item()
                                    if last_motion_idx < temp_output.label_logits.shape[1]:
                                        label_logits_at_motion = temp_output.label_logits[0, last_motion_idx, :]
                                        predicted_label = torch.argmax(label_logits_at_motion, dim=-1).item()
                                        predicted_labels.append(predicted_label)
                    
                    for token, label in zip(tokens_to_add, tokens_labels_to_add):
                        current_seq.append(token)
                        token_labels.append(label)
                    
                    if len(generated_history) > 100:
                        generated_history = generated_history[-100:]
                
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    filtered_tokens = [t for t in generated_motion_tokens if t not in special_token_ids]
    # 确保predicted_labels长度与filtered_tokens一致
    if has_label_support:
        # 如果长度不一致，截断或填充
        if len(predicted_labels) > len(filtered_tokens):
            predicted_labels = predicted_labels[:len(filtered_tokens)]
        elif len(predicted_labels) < len(filtered_tokens):
            # 如果缺少，用-1填充（表示未知）
            predicted_labels.extend([-1] * (len(filtered_tokens) - len(predicted_labels)))
        return filtered_tokens, predicted_labels if predicted_labels else None
    else:
        return filtered_tokens, None


def generate_motion_tokens_free_running_with_gt_labels(model, audio_tokens, label_tokens_gt, device="cuda", max_new_tokens=256, 
                                                      temperature=0.8, top_k=50, repetition_penalty=1.1, enable_model_debug=False):
    """Free-running模式（使用GT label one-hot）：完全自回归生成motion tokens，但使用GT label token的one-hot向量替代模型预测的label_logits
    
    参数:
        label_tokens_gt: GT label tokens列表，与motion tokens一一对应
    
    返回:
        generated_motion_tokens: 生成的motion tokens列表
        predicted_labels: 使用的GT labels列表
    """
    model.eval()
    
    # 检查模型是否支持label功能
    has_label_support = hasattr(model, 'label_classifier')
    if not has_label_support:
        raise ValueError("模型不支持label功能，无法使用GT label one-hot模式")
    
    gesture_start_token_id = getattr(model.config, 'gesture_start_token_id', 512*2 + 2)
    audio_gesture_start_token_id = getattr(model.config, 'audio_gesture_start_token_id', 512*2 + 3)
    gesture_end_token_id = getattr(model.config, 'gesture_end_token_id', 512*2 + 4)
    audio_gesture_end_token_id = getattr(model.config, 'audio_gesture_end_token_id', 512*2 + 5)
    
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    # 过滤输入中的special token
    audio_tokens = [t for t in audio_tokens if t not in special_token_ids]
    
    generated_motion_tokens = []
    used_labels = []  # 记录使用的GT labels
    current_seq = []
    token_labels = []
    generated_history = []
    
    # 准备GT label tokens的索引（用于在生成motion token时查找对应的GT label）
    label_tokens_gt_index = 0
    
    interleave_audios, interleave_motions = 1, 1
    
    with torch.no_grad():
        for i, audio_token in enumerate(audio_tokens):
            current_seq.append(audio_token)
            token_labels.append(-100)
            
            if (i + 1) % interleave_audios == 0:
                for j in range(interleave_motions):
                    if len(generated_motion_tokens) >= max_new_tokens:
                        break
                    
                    inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
                    attn_mask = torch.ones_like(inputs)
                    labels = torch.tensor(token_labels).unsqueeze(0).to(device)
                    
                    # 如果有GT label token，将其转换为one-hot向量并传入模型
                    # 创建一个包装的nn.Module来替换label_classifier
                    original_label_classifier = None
                    gt_label = None
                    
                    # 获取当前motion token位置对应的GT label
                    if label_tokens_gt_index < len(label_tokens_gt):
                        gt_label = label_tokens_gt[label_tokens_gt_index]
                        if gt_label >= 0 and gt_label < 9:  # 有效的label（0-8）
                            # 创建一个包装的nn.Module来替换label_classifier
                            class OneHotClassifier(nn.Module):
                                def __init__(self, label_value, device):
                                    super().__init__()
                                    self.label_value = label_value
                                    self.device = device
                                
                                def forward(self, x):
                                    batch_size = x.shape[0]
                                    one_hot = torch.zeros(batch_size, 9, device=self.device, dtype=x.dtype)
                                    one_hot[:, self.label_value] = 1.0
                                    return one_hot
                            
                            # 保存原始的label_classifier
                            original_label_classifier = model.label_classifier
                            # 创建one-hot分类器
                            one_hot_classifier = OneHotClassifier(gt_label, device)
                            # 临时替换
                            model.label_classifier = one_hot_classifier
                    
                    # 临时启用模型内部的调试模式（仅用于诊断）
                    if enable_model_debug:
                        model._debug_label_loss = True
                    
                    # 推理时不需要GT label_tokens，传入None（因为我们已经通过替换label_classifier来使用one-hot）
                    output = model(input_data=inputs, attention_mask=attn_mask, labels=labels, label_tokens=None)
                    
                    # 恢复原始的label_classifier
                    if original_label_classifier is not None:
                        model.label_classifier = original_label_classifier
                    
                    if enable_model_debug:
                        model._debug_label_loss = False
                    next_token_logits = output.logits[0, -1, :]
                    
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
                        generated_motion_tokens.append(next_token)
                        generated_history.append(next_token)
                        
                        # 记录使用的GT label
                        if label_tokens_gt_index < len(label_tokens_gt):
                            used_labels.append(label_tokens_gt[label_tokens_gt_index])
                            label_tokens_gt_index += 1
                        else:
                            used_labels.append(-1)  # 没有更多GT label
                    
                    for token, label in zip(tokens_to_add, tokens_labels_to_add):
                        current_seq.append(token)
                        token_labels.append(label)
                    
                    if len(generated_history) > 100:
                        generated_history = generated_history[-100:]
                
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    filtered_tokens = [t for t in generated_motion_tokens if t not in special_token_ids]
    # 确保used_labels长度与filtered_tokens一致
    if len(used_labels) > len(filtered_tokens):
        used_labels = used_labels[:len(filtered_tokens)]
    elif len(used_labels) < len(filtered_tokens):
        used_labels.extend([-1] * (len(filtered_tokens) - len(used_labels)))
    
    return filtered_tokens, used_labels if used_labels else None


def generate_motion_tokens_teacher_forcing(model, audio_tokens, motion_tokens_gt, device="cuda", enable_model_debug=False):
    """Teacher-forcing模式：使用GT motion tokens构建完整序列，提取模型在每个motion位置的预测
    
    返回:
        predicted_tokens: 预测的motion tokens列表
        gt_tokens: GT motion tokens列表
        predicted_labels: 预测的label列表（如果模型支持label功能），否则为None
    """
    model.eval()
    
    # 检查模型是否支持label功能
    has_label_support = hasattr(model, 'label_classifier')
    
    gesture_start_token_id = getattr(model.config, 'gesture_start_token_id', 512*2 + 2)
    audio_gesture_start_token_id = getattr(model.config, 'audio_gesture_start_token_id', 512*2 + 3)
    gesture_end_token_id = getattr(model.config, 'gesture_end_token_id', 512*2 + 4)
    audio_gesture_end_token_id = getattr(model.config, 'audio_gesture_end_token_id', 512*2 + 5)
    
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    # 过滤输入中的special token
    audio_tokens_clean = [t for t in audio_tokens if t not in special_token_ids]
    motion_tokens_gt_clean = [t for t in motion_tokens_gt if t not in special_token_ids]
    
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
        return [], [], None
    
    # 使用完整序列进行前向传播，获取每个位置的预测
    with torch.no_grad():
        inputs = torch.tensor(full_sequence).unsqueeze(0).to(device)
        attn_mask = torch.ones_like(inputs)
        labels = torch.tensor(token_labels).unsqueeze(0).to(device)
        
        # 前向传播（推理时不需要GT label_tokens，传入None）
        # 临时启用模型内部的调试模式（仅用于诊断，不影响模型逻辑）
        if enable_model_debug and hasattr(model, 'label_classifier'):
            model._debug_label_loss = True
        
        # 保存修改前的hidden_states（用于验证label embedding是否被添加）
        # 注意：我们无法直接访问模型内部的hidden_states，但可以通过对比两次forward的结果来验证
        # 这里我们只检查label_logits的输出
        
        output = model(input_data=inputs, attention_mask=attn_mask, labels=labels, label_tokens=None)
        
        if enable_model_debug and hasattr(model, 'label_classifier'):
            model._debug_label_loss = False
        logits = output.logits[0]  # [seq_len, vocab_size]
        
        # 输出label准确率（如果有）
        if has_label_support and hasattr(output, 'label_accuracy') and output.label_accuracy is not None:
            print(f"[Label Accuracy] {output.label_accuracy*100:.2f}%")
        
        # 提取motion token位置的预测
        # 注意：在GPT模型中，logits[i]是在看到序列[0:i]后，预测位置i的token
        # 所以对于位置pos的motion token，我们应该使用logits[pos]，因为它已经看到了序列[0:pos]的所有信息
        predicted_tokens = []
        gt_tokens = []
        predicted_labels = []
        
        for pos in motion_positions:
            if pos < logits.shape[0]:
                # 使用位置pos的logits来预测该位置的token
                pred_logits = logits[pos]
                predicted_token = torch.argmax(pred_logits, dim=-1).item()
                gt_token = full_sequence[pos]
                
                # 过滤special token
                if predicted_token not in special_token_ids and gt_token not in special_token_ids:
                    predicted_tokens.append(predicted_token)
                    gt_tokens.append(gt_token)
                    
                    # 如果模型支持label功能，提取该位置的label预测
                    if has_label_support and hasattr(output, 'label_logits') and output.label_logits is not None:
                        if pos < output.label_logits.shape[1]:
                            label_logits_at_pos = output.label_logits[0, pos, :]
                            # 调试：检查label_logits是否为全零
                            if torch.allclose(label_logits_at_pos, torch.zeros_like(label_logits_at_pos), atol=1e-6):
                                print(f"  ⚠️  警告：位置{pos}的label_logits全为0！")
                                print(f"     label_logits_at_pos: {label_logits_at_pos}")
                            predicted_label = torch.argmax(label_logits_at_pos, dim=-1).item()
                            predicted_labels.append(predicted_label)
                        else:
                            print(f"  ⚠️  位置{pos}超出label_logits范围: shape={output.label_logits.shape}")
                    elif has_label_support:
                        print(f"  ⚠️  模型支持label但output.label_logits为None或不存在")
        
        # 确保predicted_labels长度与predicted_tokens一致
        if has_label_support:
            if len(predicted_labels) != len(predicted_tokens):
                # 如果长度不一致，调整
                if len(predicted_labels) > len(predicted_tokens):
                    predicted_labels = predicted_labels[:len(predicted_tokens)]
                elif len(predicted_labels) < len(predicted_tokens):
                    predicted_labels.extend([-1] * (len(predicted_tokens) - len(predicted_labels)))
            return predicted_tokens, gt_tokens, predicted_labels if predicted_labels else None
        else:
            return predicted_tokens, gt_tokens, None


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
                label_tokens = None
                
                for msg in data.get('conversation', []):
                    if msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                        audio_tokens = msg['audio_tokens']
                        audio_path = msg.get('content', None)
                    elif msg.get('message_type') == 'audio_motion' and 'motion_tokens' in msg:
                        motion_tokens = msg['motion_tokens']
                        # label_tokens在audio_motion消息中
                        if 'label_tokens' in msg:
                            label_tokens = msg['label_tokens']
                
                if audio_tokens is None or motion_tokens is None:
                    continue
                
                # 转换为list
                if isinstance(audio_tokens, torch.Tensor):
                    audio_tokens = audio_tokens.tolist()
                if isinstance(motion_tokens, torch.Tensor):
                    motion_tokens = motion_tokens.tolist()
                if label_tokens is not None and isinstance(label_tokens, torch.Tensor):
                    label_tokens = label_tokens.tolist()
                elif label_tokens is not None and isinstance(label_tokens, (np.ndarray, list)):
                    # 确保是list
                    label_tokens = list(label_tokens) if not isinstance(label_tokens, list) else label_tokens
                
                samples.append({
                    'audio_tokens': audio_tokens,
                    'motion_tokens': motion_tokens,
                    'label_tokens': label_tokens,  # 可能为None
                    'audio_path': audio_path,
                    'sample_idx': line_num
                })
                
            except Exception as e:
                print(f"⚠️  Error loading line {line_num}: {e}")
                continue
    
    # 统计label_tokens的加载情况
    samples_with_labels = sum(1 for s in samples if s.get('label_tokens') is not None)
    print(f"✅ Loaded {len(samples)} samples from {jsonl_path}")
    print(f"   - Samples with label_tokens: {samples_with_labels}/{len(samples)}")
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
    parser.add_argument('--max_motion_tokens', type=int, default=4096,
                       help='Maximum motion tokens to generate')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--enable_model_debug', action='store_true',
                       help='Enable model internal debug mode to diagnose label_logits issues')
    
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
        'free_running': [],
        'free_running_gt_labels': []  # 新增：使用GT label one-hot的free-running模式
    }
    
    # 处理每个样本
    for sample_idx, sample in enumerate(selected_samples):
        print("\n" + "=" * 70)
        print(f"Processing Sample {sample_idx + 1}/{len(selected_samples)}")
        print("=" * 70)
        
        audio_tokens = sample['audio_tokens']
        motion_tokens_gt = sample['motion_tokens']
        audio_path = sample['audio_path']
        
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
        
        # 检查模型是否支持label功能
        has_label_support = hasattr(motion_adaptor, 'label_classifier')
        
        # 2. Teacher-forcing模式
        print("\n--- Teacher-forcing Mode ---")
        try:
            result_tf = generate_motion_tokens_teacher_forcing(
                motion_adaptor, audio_tokens, motion_tokens_gt, device="cuda",
                enable_model_debug=args.enable_model_debug
            )
            if len(result_tf) == 3:
                predicted_tokens_tf, gt_tokens_tf, predicted_labels_tf = result_tf
            else:
                # 兼容旧版本（返回2个值）
                predicted_tokens_tf, gt_tokens_tf = result_tf
                predicted_labels_tf = None
            
            if len(predicted_tokens_tf) > 0:
                # 计算准确率
                accuracy_tf = compute_token_accuracy(predicted_tokens_tf, gt_tokens_tf)
                print(f"Token accuracy (teacher-forcing): {accuracy_tf:.4f}")
                
                # 如果模型支持label功能，输出label预测结果
                if has_label_support and predicted_labels_tf is not None:
                    print(f"Predicted labels (teacher-forcing): {predicted_labels_tf}")
                    # 统计label分布
                    if predicted_labels_tf:
                        unique_labels, counts = np.unique([l for l in predicted_labels_tf if l >= 0], return_counts=True)
                        print(f"Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
                
                # 解码预测的motion tokens
                motion_pkl_tf = decode_motion_tokens(predicted_tokens_tf, motion_vae, mean_t, std_t)
                if motion_pkl_tf is not None:
                    tf_pkl_path = os.path.join(sample_dir, "teacher_forcing_motion.pkl")
                    with open(tf_pkl_path, 'wb') as f:
                        pickle.dump(motion_pkl_tf, f)
                    
                    tf_csv_path = os.path.join(sample_dir, "teacher_forcing_motion.csv")
                    motion_csv = load_motion_pkl_as_csv_data(tf_pkl_path)
                    np.savetxt(tf_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                    
                    result_dict = {
                        'sample_idx': sample_idx,
                        'accuracy': accuracy_tf,
                        'predicted_tokens': len(predicted_tokens_tf),
                        'gt_tokens': len(gt_tokens_tf),
                        'status': 'success'
                    }
                    if has_label_support and predicted_labels_tf is not None:
                        result_dict['predicted_labels'] = predicted_labels_tf
                    results['teacher_forcing'].append(result_dict)
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
                enable_model_debug=args.enable_model_debug
            )
            if len(result_fr) == 2:
                predicted_tokens_fr, predicted_labels_fr = result_fr
            else:
                # 兼容旧版本（返回1个值）
                predicted_tokens_fr = result_fr
                predicted_labels_fr = None
            
            if len(predicted_tokens_fr) > 0:
                print(f"Generated {len(predicted_tokens_fr)} motion tokens")
                
                # 如果模型支持label功能，输出label预测结果
                if has_label_support and predicted_labels_fr is not None:
                    print(f"Predicted labels (free-running): {predicted_labels_fr}")
                    # 统计label分布
                    if predicted_labels_fr:
                        unique_labels, counts = np.unique([l for l in predicted_labels_fr if l >= 0], return_counts=True)
                        print(f"Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
                
                # 解码生成的motion tokens
                motion_pkl_fr = decode_motion_tokens(predicted_tokens_fr, motion_vae, mean_t, std_t)
                if motion_pkl_fr is not None:
                    fr_pkl_path = os.path.join(sample_dir, "free_running_motion.pkl")
                    with open(fr_pkl_path, 'wb') as f:
                        pickle.dump(motion_pkl_fr, f)
                    
                    fr_csv_path = os.path.join(sample_dir, "free_running_motion.csv")
                    motion_csv = load_motion_pkl_as_csv_data(fr_pkl_path)
                    np.savetxt(fr_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                    
                    result_dict = {
                        'sample_idx': sample_idx,
                        'generated_tokens': len(predicted_tokens_fr),
                        'status': 'success'
                    }
                    if has_label_support and predicted_labels_fr is not None:
                        result_dict['predicted_labels'] = predicted_labels_fr
                    results['free_running'].append(result_dict)
                    print(f"✅ Free-running motion decoded successfully")
        except Exception as e:
            print(f"❌ Free-running mode failed: {e}")
            import traceback
            traceback.print_exc()
            results['free_running'].append({
                'sample_idx': sample_idx,
                'status': 'failed',
                'error': str(e)
            })
        
        # 4. Free-running模式（使用GT label one-hot）
        # 调试：检查label_tokens是否存在
        if has_label_support:
            print(f"\n[DEBUG] 检查label_tokens:")
            print(f"  - has_label_support: {has_label_support}")
            print(f"  - sample.get('label_tokens'): {sample.get('label_tokens')}")
            print(f"  - sample.get('label_tokens') is not None: {sample.get('label_tokens') is not None}")
            if sample.get('label_tokens') is not None:
                print(f"  - label_tokens类型: {type(sample.get('label_tokens'))}")
                print(f"  - label_tokens长度: {len(sample.get('label_tokens')) if hasattr(sample.get('label_tokens'), '__len__') else 'N/A'}")
        
        if has_label_support and sample.get('label_tokens') is not None:
            print("\n--- Free-running Mode (with GT Label One-Hot) ---")
            try:
                label_tokens_gt = sample['label_tokens']
                print(f"[DEBUG] 使用label_tokens_gt，长度: {len(label_tokens_gt) if hasattr(label_tokens_gt, '__len__') else 'N/A'}")
                result_fr_gt = generate_motion_tokens_free_running_with_gt_labels(
                    motion_adaptor, audio_tokens, label_tokens_gt, device="cuda",
                    max_new_tokens=min(args.max_motion_tokens, len(motion_tokens_gt) * 2),
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    enable_model_debug=args.enable_model_debug
                )
                if len(result_fr_gt) == 2:
                    predicted_tokens_fr_gt, used_labels_fr_gt = result_fr_gt
                else:
                    predicted_tokens_fr_gt = result_fr_gt
                    used_labels_fr_gt = None
                
                if len(predicted_tokens_fr_gt) > 0:
                    print(f"Generated {len(predicted_tokens_fr_gt)} motion tokens (with GT label one-hot)")
                    
                    if used_labels_fr_gt is not None:
                        print(f"Used GT labels: {used_labels_fr_gt}")
                        unique_labels, counts = np.unique([l for l in used_labels_fr_gt if l >= 0], return_counts=True)
                        print(f"GT label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
                    
                    # 解码生成的motion tokens
                    print(f"[DEBUG] 开始解码motion tokens，数量: {len(predicted_tokens_fr_gt)}")
                    motion_pkl_fr_gt = decode_motion_tokens(predicted_tokens_fr_gt, motion_vae, mean_t, std_t)
                    print(f"[DEBUG] 解码结果: {motion_pkl_fr_gt is not None}")
                    if motion_pkl_fr_gt is not None:
                        fr_gt_pkl_path = os.path.join(sample_dir, "free_running_gt_labels_motion.pkl")
                        with open(fr_gt_pkl_path, 'wb') as f:
                            pickle.dump(motion_pkl_fr_gt, f)
                        
                        fr_gt_csv_path = os.path.join(sample_dir, "free_running_gt_labels_motion.csv")
                        motion_csv = load_motion_pkl_as_csv_data(fr_gt_pkl_path)
                        np.savetxt(fr_gt_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                        
                        result_dict = {
                            'sample_idx': sample_idx,
                            'generated_tokens': len(predicted_tokens_fr_gt),
                            'status': 'success'
                        }
                        if used_labels_fr_gt is not None:
                            result_dict['used_labels'] = used_labels_fr_gt
                        results['free_running_gt_labels'].append(result_dict)
                        print(f"✅ Free-running (GT label one-hot) motion decoded successfully")
            except Exception as e:
                print(f"❌ Free-running (GT label one-hot) mode failed: {e}")
                import traceback
                traceback.print_exc()
                results['free_running_gt_labels'].append({
                    'sample_idx': sample_idx,
                    'status': 'failed',
                    'error': str(e)
                })
        else:
            if not has_label_support:
                print("\n--- Skipping Free-running (GT Label One-Hot) Mode: Model does not support labels ---")
            elif sample.get('label_tokens') is None:
                print("\n--- Skipping Free-running (GT Label One-Hot) Mode: No GT label tokens in sample ---")
        
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
                
                # Free-running (GT label one-hot) 视频
                if os.path.exists(os.path.join(sample_dir, "free_running_gt_labels_motion.csv")):
                    vis_audio_motion(
                        os.path.join(sample_dir, "free_running_gt_labels_motion.csv"),
                        output_path=os.path.join(sample_dir, "free_running_gt_labels_motion.mp4"),
                        audio_path=audio_copy_path,
                        robot_type="g1_brainco",
                        rate_limit=False,
                        motion_fps=25
                    )
                
                print(f"✅ Visualization videos generated")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
    
        # 合并视频进行对比（优先4个视频，如果不存在则使用3个）
        try:
            gt_mp4 = os.path.join(sample_dir, "gt_motion.mp4")
            tf_mp4 = os.path.join(sample_dir, "teacher_forcing_motion.mp4")
            fr_mp4 = os.path.join(sample_dir, "free_running_motion.mp4")
            fr_gt_mp4 = os.path.join(sample_dir, "free_running_gt_labels_motion.mp4")
            
            # 尝试合并4个视频（如果都存在）
            if (os.path.exists(gt_mp4) and os.path.exists(tf_mp4) and 
                os.path.exists(fr_mp4) and os.path.exists(fr_gt_mp4)):
                combined_out = os.path.join(sample_dir, "combined_comparison.mp4")
                print("\n--- Merging comparison video (GT | TF | FR | FR-GT-Label) ---")
                merge_four_videos(
                    v0=gt_mp4,
                    v1=tf_mp4,
                    v2=fr_mp4,
                    v3=fr_gt_mp4,
                    out_path=combined_out,
                    layout="hstack",
                    height=720,
                    crf=18,
                    preset="veryfast",
                    copy_first_audio=True,
                    label0="GT",
                    label1="Teacher-Forcing",
                    label2="Free-Running",
                    label3="FR-GT-Label",
                    fontfile="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    fontsize=36,
                    fontcolor="white",
                    box=True,
                    boxcolor="black@0.5",
                    boxborderw=10,
                )
                print(f"✅ Combined comparison video saved: {combined_out}")
            # 如果只有3个视频，使用原来的3视频合并
            elif os.path.exists(gt_mp4) and os.path.exists(tf_mp4) and os.path.exists(fr_mp4):
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
    fr_gt_success = sum(1 for r in results['free_running_gt_labels'] if r.get('status') == 'success')
    
    print(f"GT mode: {gt_success}/{len(selected_samples)} successful")
    print(f"Teacher-forcing mode: {tf_success}/{len(selected_samples)} successful")
    print(f"Free-running mode: {fr_success}/{len(selected_samples)} successful")
    print(f"Free-running (GT label one-hot) mode: {fr_gt_success}/{len(results['free_running_gt_labels'])} successful")
    
    if tf_success > 0:
        tf_accuracies = [r['accuracy'] for r in results['teacher_forcing'] if r.get('status') == 'success']
        avg_tf_accuracy = np.mean(tf_accuracies)
        print(f"Average teacher-forcing accuracy: {avg_tf_accuracy:.4f}")
    
    print(f"\n✅ Results saved to: {results_path}")
    print(f"✅ Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

