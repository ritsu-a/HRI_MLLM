#!/usr/bin/env python3
"""
简化版测试脚本：只测试Teacher-forcing模式并显示第一层attention weights
用于分析奇数和偶数位置的token是否都被学习到
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import yaml
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT
from transformers import GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

torch.cuda.set_device(0)


def load_gpt2_model(model_path, device="cuda"):
    """加载训练完成的GPT2 adaptor模型"""
    print(f"Loading GPT2 adaptor model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
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
                
                for msg in data.get('conversation', []):
                    if msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                        audio_tokens = msg['audio_tokens']
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
                    'sample_idx': line_num
                })
                
            except Exception as e:
                print(f"⚠️  Error loading line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(samples)} samples from {jsonl_path}")
    return samples


def generate_motion_tokens_teacher_forcing(model, audio_tokens, motion_tokens_gt, device="cuda"):
    """Teacher-forcing模式：使用GT motion tokens构建完整序列，提取模型在每个motion位置的预测和attention weights
    
    返回:
        predicted_tokens: 预测的motion tokens列表
        gt_tokens: GT motion tokens列表
        attention_weights: 第一层的attention weights [seq_len, seq_len]
    """
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
    
    # 使用完整序列进行前向传播，获取每个位置的预测和attention weights
    with torch.no_grad():
        inputs = torch.tensor(full_sequence).unsqueeze(0).to(device)
        attn_mask = torch.ones_like(inputs)
        labels = torch.tensor(token_labels).unsqueeze(0).to(device)
        
        # 前向传播，获取attention weights
        output = model(
            input_data=inputs, 
            attention_mask=attn_mask, 
            labels=labels, 
            label_tokens=None,
            use_label_prediction_mode=False,
            output_attentions=True  # 获取attention weights
        )
        
        # 检查logits是否存在
        if output.logits is None:
            raise ValueError("Model output logits is None. This should not happen in teacher-forcing mode.")
        logits = output.logits[0]  # [seq_len, vocab_size]
        
        # 提取第一层的attention weights
        attention_weights = None
        if hasattr(output, 'attentions') and output.attentions is not None:
            if len(output.attentions) > 0:
                first_layer_attn = output.attentions[0]  # [batch_size, num_heads, seq_len, seq_len]
                # 平均所有head的attention weights
                attention_weights = first_layer_attn[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
                print(f"✅ Extracted first layer attention weights: shape {attention_weights.shape}")
        
        # 提取motion token位置的预测
        predicted_tokens = []
        gt_tokens = []
        
        for pos in motion_positions:
            if pos < logits.shape[0]:
                pred_logits = logits[pos]
                predicted_token = torch.argmax(pred_logits, dim=-1).item()
                gt_token = full_sequence[pos]
                
                # 过滤special token
                if predicted_token not in special_token_ids and gt_token not in special_token_ids:
                    predicted_tokens.append(predicted_token)
                    gt_tokens.append(gt_token)
        
        return predicted_tokens, gt_tokens, attention_weights, motion_positions


def visualize_attention_weights(attention_weights, output_path, motion_positions, title="First Layer Attention Weights"):
    """
    可视化第一层的attention weights（只显示motion token位置）
    
    参数:
        attention_weights: attention weights矩阵 [seq_len, seq_len]
        output_path: 输出图片路径
        motion_positions: motion token在序列中的位置列表（必需）
        title: 图片标题
    """
    if attention_weights is None:
        print("⚠️  No attention weights to visualize")
        return
    
    if motion_positions is None or len(motion_positions) == 0:
        print("⚠️  No motion positions provided")
        return
    
    seq_len = attention_weights.shape[0]
    
    # 只提取motion token位置的attention weights
    # motion_positions是motion token在完整序列中的位置
    # 我们需要提取：motion token作为query时，对所有key的attention
    motion_positions_filtered = [pos for pos in motion_positions if pos < seq_len]
    
    if len(motion_positions_filtered) == 0:
        print("⚠️  No valid motion positions")
        return
    
    # 提取motion token位置的attention weights
    # [num_motion_tokens, seq_len] - 每个motion token对所有位置的attention
    motion_attn = attention_weights[motion_positions_filtered, :]
    
    # 在motion token序列中，奇数位置（1st, 3rd, 5th...）对应motion_positions中的索引0, 2, 4...
    # 偶数位置（2nd, 4th, 6th...）对应motion_positions中的索引1, 3, 5...
    num_motion_tokens = len(motion_positions_filtered)
    odd_motion_indices = [i for i in range(num_motion_tokens) if i % 2 == 0]  # 奇数位置（1st, 3rd, 5th...）
    even_motion_indices = [i for i in range(num_motion_tokens) if i % 2 == 1]   # 偶数位置（2nd, 4th, 6th...）
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：motion token的attention matrix（使用更好的colormap提高区分度）
    ax1 = axes[0]
    # 使用'plasma'或'hot' colormap提高区分度，并设置vmin/vmax来增强对比度
    vmin = motion_attn.min()
    vmax = motion_attn.max()
    # 使用更激进的对比度增强
    vmax = vmax * 0.8  # 稍微压缩最大值，让高值更突出
    
    im1 = ax1.imshow(motion_attn, cmap='plasma', aspect='auto', interpolation='nearest', 
                     vmin=vmin, vmax=vmax)
    ax1.set_title(f'{title} - Motion Tokens Only', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Key Position (Full Sequence Index)', fontsize=12)
    ax1.set_ylabel('Motion Token Index (1st, 2nd, 3rd...)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Attention Weight')
    cbar1.ax.tick_params(labelsize=10)
    
    # 标记奇数和偶数motion token位置
    for idx in odd_motion_indices:
        ax1.axhline(y=idx, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Odd positions' if idx == odd_motion_indices[0] else '')
    for idx in even_motion_indices:
        ax1.axhline(y=idx, color='blue', linestyle='--', alpha=0.6, linewidth=1.5, label='Even positions' if idx == even_motion_indices[0] else '')
    
    # 添加图例（只显示一次）
    if len(odd_motion_indices) > 0 and len(even_motion_indices) > 0:
        ax1.legend(loc='upper right', fontsize=10)
    
    # 右图：分析奇数和偶数motion token位置的attention模式
    ax2 = axes[1]
    
    # 计算每个motion token对奇数位置和偶数位置的attention总和
    # 注意：这里指的是motion token序列中的奇偶位置
    # 奇数motion token（1st, 3rd, 5th...）在完整序列中的位置
    odd_key_positions_in_seq = [motion_positions_filtered[i] for i in odd_motion_indices if i < len(motion_positions_filtered)]
    # 偶数motion token（2nd, 4th, 6th...）在完整序列中的位置
    even_key_positions_in_seq = [motion_positions_filtered[i] for i in even_motion_indices if i < len(motion_positions_filtered)]
    
    odd_attention = np.zeros(num_motion_tokens)
    even_attention = np.zeros(num_motion_tokens)
    
    for motion_idx in range(num_motion_tokens):
        # 计算对奇数motion token位置的attention（在完整序列中的位置）
        if odd_key_positions_in_seq:
            valid_odd_positions = [pos for pos in odd_key_positions_in_seq if pos < motion_attn.shape[1]]
            if valid_odd_positions:
                odd_attention[motion_idx] = motion_attn[motion_idx, valid_odd_positions].sum()
        # 计算对偶数motion token位置的attention（在完整序列中的位置）
        if even_key_positions_in_seq:
            valid_even_positions = [pos for pos in even_key_positions_in_seq if pos < motion_attn.shape[1]]
            if valid_even_positions:
                even_attention[motion_idx] = motion_attn[motion_idx, valid_even_positions].sum()
    
    x_positions = np.arange(num_motion_tokens)
    width = 0.35
    ax2.bar(x_positions - width/2, odd_attention, width, label='Attention to Odd Motion Tokens (1st, 3rd, 5th...)', 
            alpha=0.8, color='red', edgecolor='darkred', linewidth=1.5)
    ax2.bar(x_positions + width/2, even_attention, width, label='Attention to Even Motion Tokens (2nd, 4th, 6th...)', 
            alpha=0.8, color='blue', edgecolor='darkblue', linewidth=1.5)
    ax2.set_xlabel('Motion Token Index', fontsize=12)
    ax2.set_ylabel('Total Attention Weight', fontsize=12)
    ax2.set_title('Attention Distribution: Odd vs Even Motion Tokens', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 标记奇数和偶数motion token位置
    for idx in odd_motion_indices:
        ax2.axvline(x=idx, color='red', linestyle=':', alpha=0.3, linewidth=1)
    for idx in even_motion_indices:
        ax2.axvline(x=idx, color='blue', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')  # 提高DPI
    plt.close()
    
    print(f"✅ Attention weights visualization saved to: {output_path}")
    
    # 打印统计信息
    print(f"\n📊 Attention Statistics (Motion Tokens Only):")
    print(f"   - Number of motion tokens: {num_motion_tokens}")
    print(f"   - Odd motion tokens (1st, 3rd, 5th...): {len(odd_motion_indices)}")
    print(f"   - Even motion tokens (2nd, 4th, 6th...): {len(even_motion_indices)}")
    print(f"   - Average attention to odd motion tokens: {odd_attention.mean():.4f}")
    print(f"   - Average attention to even motion tokens: {even_attention.mean():.4f}")
    if even_attention.mean() > 0:
        ratio = odd_attention.mean() / even_attention.mean()
        print(f"   - Ratio (odd/even): {ratio:.4f}")
        if ratio > 1.2:
            print(f"   - ⚠️  Model pays more attention to odd positions")
        elif ratio < 0.8:
            print(f"   - ⚠️  Model pays more attention to even positions")
        else:
            print(f"   - ✅ Model pays balanced attention to both odd and even positions")
    else:
        print(f"   - Ratio (odd/even): N/A")


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
    parser = argparse.ArgumentParser(description='Test GPT2 Adaptor Teacher-Forcing with Attention Weights')
    parser.add_argument('--jsonl_path', type=str, required=True,
                       help='Path to JSONL file containing training data')
    parser.add_argument('--motion_adaptor_path', type=str, required=True,
                       help='Path to motion adaptor checkpoint')
    parser.add_argument('--output_dir', type=str, default='./attention_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
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
    results = []
    
    # 处理每个样本
    for sample_idx, sample in enumerate(selected_samples):
        print("\n" + "=" * 70)
        print(f"Processing Sample {sample_idx + 1}/{len(selected_samples)}")
        print("=" * 70)
        
        audio_tokens = sample['audio_tokens']
        motion_tokens_gt = sample['motion_tokens']
        
        print(f"Audio tokens: {len(audio_tokens)}")
        print(f"Motion tokens (GT): {len(motion_tokens_gt)}")
        
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Teacher-forcing模式
        print("\n--- Teacher-forcing Mode ---")
        try:
            predicted_tokens, gt_tokens, attention_weights, motion_positions = generate_motion_tokens_teacher_forcing(
                motion_adaptor, audio_tokens, motion_tokens_gt, device="cuda"
            )
            
            if len(predicted_tokens) > 0:
                # 计算准确率
                accuracy = compute_token_accuracy(predicted_tokens, gt_tokens)
                print(f"Token accuracy (teacher-forcing): {accuracy:.4f}")
                
                # 保存attention weights
                if attention_weights is not None and motion_positions is not None:
                    # 只保存motion token位置的attention weights
                    motion_positions_filtered = [pos for pos in motion_positions if pos < attention_weights.shape[0]]
                    if len(motion_positions_filtered) > 0:
                        motion_attn = attention_weights[motion_positions_filtered, :]
                        attn_npy_path = os.path.join(sample_dir, "attention_weights_first_layer_motion_only.npy")
                        np.save(attn_npy_path, motion_attn)
                        print(f"✅ Motion token attention weights saved to: {attn_npy_path}")
                    
                    # 可视化attention weights（只显示motion token位置）
                    attn_viz_path = os.path.join(sample_dir, "attention_weights_first_layer.png")
                    visualize_attention_weights(
                        attention_weights,
                        attn_viz_path,
                        motion_positions=motion_positions,
                        title=f"First Layer Attention Weights - Sample {sample_idx} (Teacher-Forcing)"
                    )
                
                results.append({
                    'sample_idx': sample_idx,
                    'accuracy': accuracy,
                    'predicted_tokens': len(predicted_tokens),
                    'gt_tokens': len(gt_tokens),
                    'status': 'success'
                })
                print(f"✅ Teacher-forcing completed successfully")
        except Exception as e:
            print(f"❌ Teacher-forcing mode failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'sample_idx': sample_idx,
                'status': 'failed',
                'error': str(e)
            })
    
    # 保存评估结果
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r.get('status') == 'success')
    print(f"Success: {success_count}/{len(selected_samples)}")
    
    if success_count > 0:
        accuracies = [r['accuracy'] for r in results if r.get('status') == 'success']
        avg_accuracy = np.mean(accuracies)
        print(f"Average accuracy: {avg_accuracy:.4f}")
    
    print(f"\n✅ Results saved to: {results_path}")
    print(f"✅ Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

