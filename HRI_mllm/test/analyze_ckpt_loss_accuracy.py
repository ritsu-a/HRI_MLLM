#!/usr/bin/env python3
"""
分析checkpoint在训练和测试集上的teacher-forcing loss和accuracy分布

使用示例:
    python HRI_mllm/test/analyze_ckpt_loss_accuracy.py \
        --checkpoint_dir /root/workspace/HRI_MLLM/output_disk0/motion_adaptor_v18/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints \
        --train_jsonl /root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens_with_labels_train_other.jsonl \
        --test_jsonl /root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens_with_labels_test.jsonl \
        --output_dir ./ckpt_loss_accuracy_analysis \
        --start_epoch 50 \
        --end_epoch 800 \
        --epoch_interval 50 \
        --num_samples 100
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT
from transformers import GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

torch.cuda.set_device(0)


def load_gpt2_from_checkpoint(checkpoint_path, device="cuda"):
    """从.pt checkpoint文件加载模型"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get('epoch', 0)
    
    model_config = GPT2Config(
        vocab_size=1034,
        n_positions=256,
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
    else:
        print("⚠️  model_state not found in checkpoint, using random weights")
    
    model.eval()
    model.to(device)
    return model, epoch


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
                })
                
            except Exception as e:
                print(f"⚠️  Error loading line {line_num}: {e}")
                continue
    
    return samples


def compute_teacher_forcing_loss_accuracy(model, audio_tokens, motion_tokens_gt, device="cuda"):
    """Teacher-forcing模式：计算loss和accuracy
    
    返回:
        loss: 平均loss值
        accuracy: token级别的准确率
        num_tokens: 用于计算的token数量
    """
    model.eval()
    
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
        return None, None, 0
    
    # 使用完整序列进行前向传播，获取loss和预测
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
        
        # 计算loss (模型会自动计算，只计算labels != -100的位置)
        loss = output.loss.item() if output.loss is not None else None
        
        # 计算accuracy
        # 注意：GPT模型的loss使用shift方式：logits[i]预测labels[i+1]
        # 所以对于位置pos的motion token，应该使用logits[pos-1]来预测（如果pos > 0）
        logits = output.logits[0]  # [seq_len, vocab_size]
        predicted_tokens = []
        gt_tokens = []
        
        for pos in motion_positions:
            # 使用shift方式：logits[i]预测位置i+1的token
            # 所以对于位置pos的token，使用logits[pos-1]（需要pos > 0）
            if pos > 0 and pos - 1 < logits.shape[0]:
                pred_logits = logits[pos - 1]
                predicted_token = torch.argmax(pred_logits, dim=-1).item()
                gt_token = full_sequence[pos]
                
                # 过滤special token和padding token
                if (predicted_token not in special_token_ids and 
                    gt_token not in special_token_ids and
                    predicted_token != motion_empty_token_id and
                    gt_token != motion_empty_token_id):
                    predicted_tokens.append(predicted_token)
                    gt_tokens.append(gt_token)
        
        # 计算准确率
        if len(predicted_tokens) > 0 and len(gt_tokens) > 0:
            min_len = min(len(predicted_tokens), len(gt_tokens))
            correct = sum(1 for p, g in zip(predicted_tokens[:min_len], gt_tokens[:min_len]) if p == g)
            accuracy = correct / min_len if min_len > 0 else 0.0
            num_tokens = min_len
        else:
            accuracy = 0.0
            num_tokens = 0
    
    return loss, accuracy, num_tokens


def evaluate_checkpoint(checkpoint_path, jsonl_path, num_samples, device="cuda"):
    """评估单个checkpoint在数据集上的表现"""
    # 加载模型
    model, epoch = load_gpt2_from_checkpoint(checkpoint_path, device)
    
    # 加载数据
    samples = load_jsonl_data(jsonl_path, max_samples=num_samples)
    
    if len(samples) == 0:
        return None, None, 0
    
    # 评估所有样本
    losses = []
    accuracies = []
    total_tokens = 0
    
    for sample in tqdm(samples, desc=f"Epoch {epoch}", leave=False):
        audio_tokens = sample['audio_tokens']
        motion_tokens_gt = sample['motion_tokens']
        
        try:
            loss, accuracy, num_tokens = compute_teacher_forcing_loss_accuracy(
                model, audio_tokens, motion_tokens_gt, device
            )
            
            if loss is not None and accuracy is not None:
                losses.append(loss)
                accuracies.append(accuracy)
                total_tokens += num_tokens
        except Exception as e:
            print(f"⚠️  Error processing sample: {e}")
            continue
    
    # 计算平均值
    avg_loss = np.mean(losses) if len(losses) > 0 else None
    avg_accuracy = np.mean(accuracies) if len(accuracies) > 0 else None
    
    return avg_loss, avg_accuracy, len(samples)


def get_checkpoint_paths(checkpoint_dir, start_epoch, end_epoch, epoch_interval):
    """获取指定范围内的checkpoint路径"""
    checkpoint_paths = []
    
    for epoch in range(start_epoch, end_epoch + 1, epoch_interval):
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint_paths.append((epoch, checkpoint_path))
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_paths


def plot_results(results, output_dir):
    """绘制loss和accuracy的分布图"""
    epochs = [r['epoch'] for r in results]
    train_losses = [r['train_loss'] for r in results if r['train_loss'] is not None]
    test_losses = [r['test_loss'] for r in results if r['test_loss'] is not None]
    train_accuracies = [r['train_accuracy'] for r in results if r['train_accuracy'] is not None]
    test_accuracies = [r['test_accuracy'] for r in results if r['test_accuracy'] is not None]
    
    # 过滤None值
    train_epochs = [r['epoch'] for r in results if r['train_loss'] is not None]
    test_epochs = [r['epoch'] for r in results if r['test_loss'] is not None]
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Loss图
    ax1 = axes[0]
    if train_epochs:
        ax1.plot(train_epochs, train_losses, 'o-', label='Train Loss', color='blue', linewidth=2, markersize=6)
    if test_epochs:
        ax1.plot(test_epochs, test_losses, 's-', label='Test Loss', color='red', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Teacher-Forcing Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(epochs) - 50, max(epochs) + 50])
    
    # Accuracy图
    ax2 = axes[1]
    train_acc_epochs = [r['epoch'] for r in results if r['train_accuracy'] is not None]
    test_acc_epochs = [r['epoch'] for r in results if r['test_accuracy'] is not None]
    train_accs = [r['train_accuracy'] for r in results if r['train_accuracy'] is not None]
    test_accs = [r['test_accuracy'] for r in results if r['test_accuracy'] is not None]
    
    if train_acc_epochs:
        ax2.plot(train_acc_epochs, train_accs, 'o-', label='Train Accuracy', color='blue', linewidth=2, markersize=6)
    if test_acc_epochs:
        ax2.plot(test_acc_epochs, test_accs, 's-', label='Test Accuracy', color='red', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Teacher-Forcing Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(epochs) - 50, max(epochs) + 50])
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_accuracy_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to: {os.path.join(output_dir, 'loss_accuracy_distribution.png')}")


def main():
    parser = argparse.ArgumentParser(description='Analyze checkpoint loss and accuracy distribution')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--train_jsonl', type=str, required=True,
                       help='Path to training JSONL file')
    parser.add_argument('--test_jsonl', type=str, required=True,
                       help='Path to test JSONL file')
    parser.add_argument('--output_dir', type=str, default='./ckpt_loss_accuracy_analysis',
                       help='Output directory for results')
    parser.add_argument('--start_epoch', type=int, default=50,
                       help='Start epoch')
    parser.add_argument('--end_epoch', type=int, default=800,
                       help='End epoch')
    parser.add_argument('--epoch_interval', type=int, default=50,
                       help='Epoch interval')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate per dataset')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取checkpoint路径列表
    checkpoint_paths = get_checkpoint_paths(
        args.checkpoint_dir, 
        args.start_epoch, 
        args.end_epoch, 
        args.epoch_interval
    )
    
    if len(checkpoint_paths) == 0:
        print("❌ No checkpoints found!")
        return
    
    print(f"Found {len(checkpoint_paths)} checkpoints to evaluate")
    
    # 评估结果
    results = []
    
    # 遍历所有checkpoint
    for epoch, checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating checkpoints"):
        print(f"\n{'='*70}")
        print(f"Evaluating Epoch {epoch}")
        print(f"{'='*70}")
        
        # 评估训练集
        print(f"\n--- Training Set ---")
        train_loss, train_accuracy, train_samples = evaluate_checkpoint(
            checkpoint_path, args.train_jsonl, args.num_samples
        )
        print(f"Train Loss: {train_loss:.6f}" if train_loss is not None else "Train Loss: N/A")
        print(f"Train Accuracy: {train_accuracy:.4f}" if train_accuracy is not None else "Train Accuracy: N/A")
        print(f"Train Samples: {train_samples}")
        
        # 评估测试集
        print(f"\n--- Test Set ---")
        test_loss, test_accuracy, test_samples = evaluate_checkpoint(
            checkpoint_path, args.test_jsonl, args.num_samples
        )
        print(f"Test Loss: {test_loss:.6f}" if test_loss is not None else "Test Loss: N/A")
        print(f"Test Accuracy: {test_accuracy:.4f}" if test_accuracy is not None else "Test Accuracy: N/A")
        print(f"Test Samples: {test_samples}")
        
        results.append({
            'epoch': epoch,
            'checkpoint_path': checkpoint_path,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_samples': train_samples,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_samples': test_samples,
        })
    
    # 保存结果
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # 绘制图表
    print("\n--- Generating plots ---")
    plot_results(results, args.output_dir)
    
    # 打印摘要
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    print("-"*70)
    for r in results:
        train_loss_str = f"{r['train_loss']:.6f}" if r['train_loss'] is not None else "N/A"
        train_acc_str = f"{r['train_accuracy']:.4f}" if r['train_accuracy'] is not None else "N/A"
        test_loss_str = f"{r['test_loss']:.6f}" if r['test_loss'] is not None else "N/A"
        test_acc_str = f"{r['test_accuracy']:.4f}" if r['test_accuracy'] is not None else "N/A"
        print(f"{r['epoch']:<8} {train_loss_str:<12} {train_acc_str:<12} {test_loss_str:<12} {test_acc_str:<12}")


if __name__ == "__main__":
    main()

