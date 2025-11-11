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
import yaml
import random
import shutil
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
from HRI_mllm.test.merge_videos import merge_three_videos

from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

torch.cuda.set_device(0)


def load_gpt2_model(model_path, device="cuda"):
    """加载训练完成的GPT2 adaptor模型"""
    print(f"Loading GPT2 adaptor model from: {model_path}")
    
    if model_path.endswith('.pt'):
        return load_gpt2_from_checkpoint(model_path, device)
    else:
        return load_gpt2_from_transformers(model_path, device)


def load_gpt2_from_checkpoint(checkpoint_path, device="cuda"):
    """从.pt checkpoint文件加载模型"""
    print(f"🔄 Loading from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get('epoch', 0)
    print(f"📊 Checkpoint epoch: {epoch}")
    
    model_config = GPT2Config(
        vocab_size=1034,
        n_positions=4096,
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
                                       temperature=0.8, top_k=50, repetition_penalty=1.1):
    """Free-running模式：完全自回归生成motion tokens"""
    model.eval()
    
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
                    
                    output = model(input_data=inputs, attention_mask=attn_mask, labels=labels)
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
                    
                    for token, label in zip(tokens_to_add, tokens_labels_to_add):
                        current_seq.append(token)
                        token_labels.append(label)
                    
                    if len(generated_history) > 100:
                        generated_history = generated_history[-100:]
                
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    filtered_tokens = [t for t in generated_motion_tokens if t not in special_token_ids]
    return filtered_tokens


def generate_motion_tokens_teacher_forcing(model, audio_tokens, motion_tokens_gt, device="cuda"):
    """Teacher-forcing模式：使用GT motion tokens构建完整序列，提取模型在每个motion位置的预测"""
    model.eval()
    
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
        return [], []
    
    # 使用完整序列进行前向传播，获取每个位置的预测
    with torch.no_grad():
        inputs = torch.tensor(full_sequence).unsqueeze(0).to(device)
        attn_mask = torch.ones_like(inputs)
        labels = torch.tensor(token_labels).unsqueeze(0).to(device)
        
        # 前向传播
        output = model(input_data=inputs, attention_mask=attn_mask, labels=labels)
        logits = output.logits[0]  # [seq_len, vocab_size]
        
        # 提取motion token位置的预测
        # 注意：在GPT模型中，logits[i]是在看到序列[0:i]后，预测位置i的token
        # 所以对于位置pos的motion token，我们应该使用logits[pos]，因为它已经看到了序列[0:pos]的所有信息
        predicted_tokens = []
        gt_tokens = []
        
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
                motion_adaptor, audio_tokens, motion_tokens_gt, device="cuda"
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
            results['teacher_forcing'].append({
                'sample_idx': sample_idx,
                'status': 'failed',
                'error': str(e)
            })
        
        # 3. Free-running模式
        print("\n--- Free-running Mode ---")
        try:
            predicted_tokens_fr = generate_motion_tokens_free_running(
                motion_adaptor, audio_tokens, device="cuda",
                max_new_tokens=min(args.max_motion_tokens, len(motion_tokens_gt) * 2),
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty
            )
            
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
        except Exception as e:
            print(f"❌ Free-running mode failed: {e}")
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
    
        # 合并三个视频进行对比
        try:
            gt_mp4 = os.path.join(sample_dir, "gt_motion.mp4")
            tf_mp4 = os.path.join(sample_dir, "teacher_forcing_motion.mp4")
            fr_mp4 = os.path.join(sample_dir, "free_running_motion.mp4")
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

