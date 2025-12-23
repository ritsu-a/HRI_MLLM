#!/usr/bin/env python3
"""
测试音频-动作未来预测模型
任务：根据25帧audio token + 25帧motion token历史，预测14帧future motion token

使用滑动窗口形式进行预测，并生成视频对比GT和预测结果

使用示例：
    python HRI_mllm/test/test_future_prediction.py \
        --jsonl_path data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/synthetic_data_en_tokens_test.jsonl \
        --model_checkpoint output_disk0/motion_adaptor_v21_future_prediction/motion_future_prediction/checkpoints/epoch_100.pt \
        --vqvae_config g1_vqvae_arbitrary_length_balanced.yaml \
        --vqvae_checkpoint output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt \
        --output_dir ./future_prediction_results \
        --num_samples 5 \
        --history_audio_frames 25 \
        --history_motion_frames 25 \
        --future_motion_frames 14 \
        --window_step 1 \
        --temperature 1.0 \
        --top_k 50

参数说明：
    --jsonl_path: 测试数据JSONL文件路径
    --model_checkpoint: 未来预测模型checkpoint路径
    --vqvae_config: VQ-VAE配置文件名称
    --vqvae_checkpoint: VQ-VAE checkpoint路径（可选）
    --output_dir: 输出目录
    --num_samples: 测试样本数量
    --history_audio_frames: 历史audio帧数（默认25）
    --history_motion_frames: 历史motion帧数（默认25）
    --future_motion_frames: 未来motion帧数（默认14）
    --window_step: 滑动窗口步长（默认1，即每次移动1帧）
    --temperature: 采样温度（默认1.0）
    --top_k: top-k采样（默认50）
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

from transformers import GPT2Config, GPT2LMHeadModel
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

torch.cuda.set_device(0)


def load_future_prediction_model(checkpoint_path, device="cuda"):
    """加载训练完成的未来预测模型"""
    print(f"Loading future prediction model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get('epoch', 0)
    print(f"📊 Checkpoint epoch: {epoch}")
    
    # 模型配置（与训练时一致）
    motion_vocab_size = 512 * 2
    total_vocab_size = motion_vocab_size + 10
    
    model_config = GPT2Config(
        vocab_size=total_vocab_size,
        n_positions=64,  # 25 + 25 + 14
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    # 特殊token IDs
    model_config.gesture_start_token_id = motion_vocab_size + 2
    model_config.audio_gesture_start_token_id = motion_vocab_size + 3
    model_config.gesture_end_token_id = motion_vocab_size + 4
    model_config.audio_gesture_end_token_id = motion_vocab_size + 5
    
    model = MixedInputGPT2(model_config, audio_hidden_size=3584)
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print("✅ Loaded model_state from checkpoint")
    else:
        print("⚠️  model_state not found in checkpoint, using random weights")
    
    model.eval()
    model.to(device)
    print(f"✅ Future prediction model loaded successfully!")
    return model


def load_vqvae_model(vqvae_config_name='g1_vqvae_arbitrary_length_balanced.yaml', 
                     vqvae_checkpoint=None, device='cuda'):
    """加载VQ-VAE模型用于解码motion tokens"""
    def open_yaml(path):
        with open(path, 'r', encoding="utf-8") as file:
            return yaml.safe_load(file)
    
    config_path = os.path.join(ROOT, "model", "motion_encoder", vqvae_config_name)
    print(f"Loading VQ-VAE config from: {config_path}")
    motion_config = open_yaml(config_path)
    
    if vqvae_checkpoint:
        checkpoint_path = vqvae_checkpoint
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
        return None, None, None
    
    print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")
    
    test_mean, test_std = load_normalization_stats(motion_config)
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to(device)
    std_t = torch.tensor(test_std, dtype=torch.float32).to(device)
    print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")
    
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device)
    print(f"✅ VQ-VAE model loaded successfully!")
    return motion_vae, mean_t, std_t


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """解码motion tokens为motion features"""
    # 过滤所有special token IDs
    special_token_ids = {512*2 + 2, 512*2 + 3, 512*2 + 4, 512*2 + 5}
    motion_empty_token_id = 512*2 + 7
    # 只保留有效的motion tokens: [0, 1023]
    filtered_tokens = [t for t in motion_tokens 
                       if t not in special_token_ids 
                       and t != motion_empty_token_id
                       and 0 <= int(t) < 1024]
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


def predict_future_motion_sliding_window(
    model, 
    audio_tokens, 
    motion_tokens_history,
    history_audio_frames=25,
    history_motion_frames=25,
    future_motion_frames=14,
    window_step=1,
    device="cuda",
    temperature=1.0,
    top_k=50
):
    """
    使用滑动窗口预测future motion tokens
    
    Args:
        model: 未来预测模型
        audio_tokens: 完整的audio tokens序列
        motion_tokens_history: 初始的motion tokens历史（用于第一个窗口）
        history_audio_frames: 历史audio帧数（默认25）
        history_motion_frames: 历史motion帧数（默认25）
        future_motion_frames: 未来motion帧数（默认14）
        window_step: 滑动窗口步长（默认1，即每次移动1帧）
        device: 计算设备
        temperature: 采样温度
        top_k: top-k采样
    
    Returns:
        predicted_motion_tokens: 预测的所有future motion tokens列表
    """
    model.eval()
    
    # Token IDs
    audio_empty_token_id = 152063
    motion_empty_token_id = 512*2 + 7
    special_token_ids = {512*2 + 2, 512*2 + 3, 512*2 + 4, 512*2 + 5}
    
    # 过滤输入中的special tokens
    audio_tokens_clean = [t for t in audio_tokens if t not in special_token_ids]
    motion_tokens_clean = [t for t in motion_tokens_history if t not in special_token_ids and 0 <= int(t) < 1024]
    
    # 计算可以预测多少个窗口
    audio_len = len(audio_tokens_clean)
    motion_len = len(motion_tokens_clean)
    
    # 初始化motion历史缓冲区
    motion_history_buffer = motion_tokens_clean.copy()
    
    # 存储所有预测的future motion tokens
    all_predicted_future_tokens = []
    
    # 滑动窗口预测
    max_audio_idx = audio_len - history_audio_frames
    window_idx = 0
    
    with torch.no_grad():
        while True:
            # 计算当前窗口的audio起始位置
            audio_start_idx = window_idx * window_step
            
            if audio_start_idx >= max_audio_idx:
                break
            
            audio_end_idx = min(audio_start_idx + history_audio_frames, audio_len)
            
            # 提取audio历史
            if audio_end_idx - audio_start_idx < history_audio_frames:
                # 如果audio不足，在前面padding
                history_audio = [audio_empty_token_id] * history_audio_frames
                available_audio = audio_tokens_clean[audio_start_idx:audio_end_idx]
                history_audio[-len(available_audio):] = available_audio
            else:
                history_audio = audio_tokens_clean[audio_start_idx:audio_end_idx]
            
            # 提取motion历史（从缓冲区中取最后history_motion_frames个）
            if len(motion_history_buffer) >= history_motion_frames:
                history_motion = motion_history_buffer[-history_motion_frames:]
            else:
                # 如果motion历史不足，在前面padding
                history_motion = [motion_empty_token_id] * history_motion_frames
                available_motion = motion_history_buffer
                history_motion[-len(available_motion):] = available_motion
            
            # 【已修复】训练时现在使用：[25 audio] + [25 motion_history] + [14 padding] 作为输入
            # labels在future_motion位置有真实token用于计算loss
            # 测试时与训练时完全一致：一次性预测所有future motion tokens，使用padding作为输入
            
            # 构建输入序列：[25 audio] + [25 motion] + [14 padding]
            input_sequence = history_audio + history_motion + [motion_empty_token_id] * future_motion_frames
            
            # 创建labels：训练时future_motion位置的labels是真实token ID（用于计算loss）
            # 测试时我们也用motion_empty_token_id作为占位符，确保这些位置被识别为motion token类型
            # （模型通过 labels != -100 来判断motion token位置）
            labels = [-100] * (history_audio_frames + history_motion_frames) + [motion_empty_token_id] * future_motion_frames
            
            # 转换为tensor
            inputs = torch.tensor(input_sequence).unsqueeze(0).to(device).long()
            labels_tensor = torch.tensor(labels).unsqueeze(0).to(device).long()
            attention_mask = torch.ones_like(inputs)
            
            # 前向传播
            outputs = model(inputs, labels=labels_tensor, attention_mask=attention_mask)
            
            if outputs.logits is None:
                print(f"⚠️  窗口 {window_idx} 的logits为None，跳过")
                break
            
            # 提取future motion位置的logits
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            future_start_idx = history_audio_frames + history_motion_frames
            future_logits = logits[future_start_idx:future_start_idx + future_motion_frames]  # [14, vocab_size]
            
            # 一次性预测所有future motion tokens
            predicted_future_tokens = []
            for i in range(future_motion_frames):
                token_logits = future_logits[i]
                
                # 排除padding token和special tokens
                if motion_empty_token_id < token_logits.size(-1):
                    token_logits[motion_empty_token_id] = float('-inf')
                for special_id in special_token_ids:
                    if special_id < token_logits.size(-1):
                        token_logits[special_id] = float('-inf')
                
                # 采样
                if temperature > 0:
                    token_logits = token_logits / temperature
                    if top_k > 0:
                        k = min(top_k, token_logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(token_logits, k)
                        masked = torch.full_like(token_logits, float('-inf'))
                        masked[top_k_indices] = top_k_logits
                        token_logits = masked
                    probs = torch.softmax(token_logits, dim=-1)
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() <= 0:
                        predicted_token = torch.argmax(token_logits, dim=-1).item()
                    else:
                        predicted_token = torch.multinomial(probs, 1).item()
                else:
                    if top_k > 0:
                        k = min(top_k, token_logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(token_logits, k)
                        masked = torch.full_like(token_logits, float('-inf'))
                        masked[top_k_indices] = top_k_logits
                        token_logits = masked
                    predicted_token = torch.argmax(token_logits, dim=-1).item()
                
                # 确保token在有效范围内
                if predicted_token < 0 or predicted_token >= 1024:
                    predicted_token = predicted_token % 1024
                
                predicted_future_tokens.append(predicted_token)
            
            # 将预测的tokens添加到结果中（根据window_step决定添加多少）
            if window_step <= future_motion_frames:
                # 如果步长小于等于预测长度，添加前window_step个tokens
                tokens_to_add = predicted_future_tokens[:window_step]
                all_predicted_future_tokens.extend(tokens_to_add)
                # 更新motion历史缓冲区：添加所有预测的tokens（用于下一个窗口的历史）
                motion_history_buffer.extend(predicted_future_tokens)
            else:
                # 如果步长大于预测长度，添加所有预测的tokens，并用padding填充
                all_predicted_future_tokens.extend(predicted_future_tokens)
                motion_history_buffer.extend(predicted_future_tokens)
                # 如果还需要更多tokens，用padding填充（但这种情况应该很少见）
                remaining = window_step - future_motion_frames
                if remaining > 0:
                    all_predicted_future_tokens.extend([motion_empty_token_id] * remaining)
                    motion_history_buffer.extend([motion_empty_token_id] * remaining)
            
            # 保持缓冲区大小合理（只保留最近的历史，至少保留history_motion_frames个）
            min_buffer_size = history_motion_frames
            max_buffer_size = history_motion_frames * 3
            if len(motion_history_buffer) > max_buffer_size:
                motion_history_buffer = motion_history_buffer[-max_buffer_size:]
            elif len(motion_history_buffer) < min_buffer_size:
                # 如果缓冲区太小，用padding填充
                padding_needed = min_buffer_size - len(motion_history_buffer)
                motion_history_buffer = [motion_empty_token_id] * padding_needed + motion_history_buffer
            
            window_idx += 1
            
            if window_idx % 10 == 0:
                print(f"  处理窗口 {window_idx}，已预测 {len(all_predicted_future_tokens)} 个motion tokens")
    
    print(f"✅ 滑动窗口预测完成，共处理 {window_idx} 个窗口，预测了 {len(all_predicted_future_tokens)} 个future motion tokens")
    return all_predicted_future_tokens


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


def merge_two_videos(v0, v1, out_path, layout="hstack", height=720, crf=18,
                     preset="veryfast", copy_first_audio=True,
                     label0="GT", label1="Predicted",
                     fontfile=None, fontsize=36, fontcolor="white",
                     box=True, boxcolor="black@0.5", boxborderw=10):
    """使用ffmpeg将两个视频横向合并，并在每个视频上添加文字标签"""
    # 检查输入文件是否存在
    for video_path in [v0, v1]:
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
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v0text][v1text]hstack=inputs=2[outv]"
        )
    else:
        # 纵向合并
        filter_complex = (
            f"[0:v]scale=-1:{height}[v0scaled];"
            f"[1:v]scale=-1:{height}[v1scaled];"
            f"[v0scaled]{build_drawtext_filter(label0)}[v0text];"
            f"[v1scaled]{build_drawtext_filter(label1)}[v1text];"
            f"[v0text][v1text]vstack=inputs=2[outv]"
        )
    
    # 构建完整的ffmpeg命令
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-i", v0,
        "-i", v1,
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
    parser = argparse.ArgumentParser(description='Test Future Prediction Model with Sliding Window')
    parser.add_argument('--jsonl_path', type=str, required=True,
                       help='Path to JSONL file containing test data')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to future prediction model checkpoint')
    parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='VQ-VAE config file name')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                       help='VQ-VAE checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./future_prediction_results',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
    parser.add_argument('--history_audio_frames', type=int, default=25,
                       help='Number of history audio frames')
    parser.add_argument('--history_motion_frames', type=int, default=25,
                       help='Number of history motion frames')
    parser.add_argument('--future_motion_frames', type=int, default=14,
                       help='Number of future motion frames to predict')
    parser.add_argument('--window_step', type=int, default=1,
                       help='Sliding window step size')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k for sampling')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载未来预测模型
    print("=" * 70)
    print("Loading Future Prediction Model")
    print("=" * 70)
    future_prediction_model = load_future_prediction_model(args.model_checkpoint)
    
    # 加载VQ-VAE模型
    print("\n" + "=" * 70)
    print("Loading VQ-VAE Model")
    print("=" * 70)
    motion_vae, mean_t, std_t = load_vqvae_model(
        args.vqvae_config, 
        args.vqvae_checkpoint
    )
    
    if motion_vae is None:
        print("❌ Failed to load VQ-VAE model")
        exit(1)
    
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
        
        # 1. 解码GT motion tokens
        print("\n--- Decoding GT Motion ---")
        try:
            motion_pkl_gt = decode_motion_tokens(motion_tokens_gt, motion_vae, mean_t, std_t)
            if motion_pkl_gt is not None:
                gt_pkl_path = os.path.join(sample_dir, "gt_motion.pkl")
                with open(gt_pkl_path, 'wb') as f:
                    pickle.dump(motion_pkl_gt, f)
                
                gt_csv_path = os.path.join(sample_dir, "gt_motion.csv")
                motion_csv = load_motion_pkl_as_csv_data(gt_pkl_path)
                np.savetxt(gt_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                print(f"✅ GT motion decoded successfully")
        except Exception as e:
            print(f"❌ GT motion decoding failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 2. 使用滑动窗口预测future motion tokens
        print("\n--- Predicting Future Motion with Sliding Window ---")
        try:
            # 使用前history_motion_frames个motion tokens作为初始历史
            initial_motion_history = motion_tokens_gt[:args.history_motion_frames]
            
            predicted_future_tokens = predict_future_motion_sliding_window(
                future_prediction_model,
                audio_tokens,
                initial_motion_history,
                history_audio_frames=args.history_audio_frames,
                history_motion_frames=args.history_motion_frames,
                future_motion_frames=args.future_motion_frames,
                window_step=args.window_step,
                device="cuda",
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            if len(predicted_future_tokens) > 0:
                print(f"✅ Predicted {len(predicted_future_tokens)} future motion tokens")
                
                # 构建完整的预测motion序列：历史 + 预测
                predicted_motion_tokens = initial_motion_history + predicted_future_tokens
                
                # 为了对比，确保预测序列长度不超过GT序列长度
                # 或者可以选择截断到相同长度
                max_length = len(motion_tokens_gt)
                if len(predicted_motion_tokens) > max_length:
                    predicted_motion_tokens = predicted_motion_tokens[:max_length]
                    print(f"⚠️  预测序列长度 ({len(predicted_motion_tokens)}) 超过GT长度 ({max_length})，已截断")
                
                # 解码预测的motion tokens
                motion_pkl_pred = decode_motion_tokens(predicted_motion_tokens, motion_vae, mean_t, std_t)
                if motion_pkl_pred is not None:
                    pred_pkl_path = os.path.join(sample_dir, "predicted_motion.pkl")
                    with open(pred_pkl_path, 'wb') as f:
                        pickle.dump(motion_pkl_pred, f)
                    
                    pred_csv_path = os.path.join(sample_dir, "predicted_motion.csv")
                    motion_csv = load_motion_pkl_as_csv_data(pred_pkl_path)
                    np.savetxt(pred_csv_path, motion_csv, delimiter=',', fmt='%.8f')
                    print(f"✅ Predicted motion decoded successfully")
        except Exception as e:
            print(f"❌ Future prediction failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 3. 生成可视化视频
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
                
                # 预测视频
                if os.path.exists(os.path.join(sample_dir, "predicted_motion.csv")):
                    vis_audio_motion(
                        os.path.join(sample_dir, "predicted_motion.csv"),
                        output_path=os.path.join(sample_dir, "predicted_motion.mp4"),
                        audio_path=audio_copy_path,
                        robot_type="g1_brainco",
                        rate_limit=False,
                        motion_fps=25
                    )
                
                print(f"✅ Visualization videos generated")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. 合并视频进行对比
        try:
            gt_mp4 = os.path.join(sample_dir, "gt_motion.mp4")
            pred_mp4 = os.path.join(sample_dir, "predicted_motion.mp4")
            
            if os.path.exists(gt_mp4) and os.path.exists(pred_mp4):
                combined_out = os.path.join(sample_dir, "comparison.mp4")
                print("\n--- Merging comparison video (GT | Predicted) ---")
                merge_two_videos(
                    v0=gt_mp4,
                    v1=pred_mp4,
                    out_path=combined_out,
                    layout="hstack",
                    height=720,
                    crf=18,
                    preset="veryfast",
                    copy_first_audio=True,
                    label0="GT",
                    label1="Predicted",
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
    
    print("\n" + "=" * 70)
    print("Testing Completed")
    print("=" * 70)
    print(f"✅ Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

