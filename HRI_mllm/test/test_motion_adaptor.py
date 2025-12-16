#!/usr/bin/env python3
"""
测试Motion Adaptor模型在测试集上的性能

支持：
- 从checkpoint加载模型
- 在多个测试集上评估模型性能
- Teacher-forcing模式
- 计算准确率、损失等指标

使用示例：
    python HRI_mllm/test/test_motion_adaptor.py \
        --checkpoint_path output_disk0/motion_adaptor_v19_synthetic_data_en/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_300.pt \
        --test_jsonl /path/to/test1.jsonl /path/to/test2.jsonl \
        --output_dir ./test_results \
        --num_samples 100 \
        --batch_size 8
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from HRI_mllm.datasets.jsonl_audio_motion_dataset import JSONLAudioMotionDataset

torch.cuda.set_device(0)


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载训练好的Motion Adaptor模型"""
    print(f"🔄 加载模型checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get('epoch', 0)
    print(f"📊 Checkpoint epoch: {epoch}")
    
    # 模型配置（与训练时保持一致）
    model_config = GPT2Config(
        vocab_size=512*2 + 10,  # 1034
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    # 添加特殊token ID
    model_config.audio_gesture_start_token_id = 512*2 + 3
    model_config.audio_gesture_end_token_id = 512*2 + 5
    model_config.gesture_start_token_id = 512*2 + 2
    model_config.gesture_end_token_id = 512*2 + 4
    
    model = MixedInputGPT2(model_config, audio_hidden_size=3584)
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print("✅ 成功加载模型权重")
    else:
        print("⚠️  checkpoint中未找到model_state，使用随机权重")
    
    model.eval()
    model.to(device)
    print(f"✅ 模型加载完成！")
    return model


def create_test_config():
    """创建测试配置（与训练配置保持一致）"""
    class TestConfig:
        def __init__(self):
            self.audio_vocab_size = 16384
            self.motion_vocab_size = 512*2
            self.total_vocab_size = 512*2 + 10
            self.max_seq_length = 512
            self.min_seq_length = 32
            self.batch_size = 256
            self.learning_rate = 1e-4
            self.sliding_window_step = 32
            self.pad_token_id = 512*2 + 1
            self.gesture_start_token_id = 512*2 + 2
            self.audio_gesture_start_token_id = 512*2 + 3
            self.gesture_end_token_id = 512*2 + 4
            self.audio_gesture_end_token_id = 512*2 + 5
            self.audio_empty_token_id = 152063
            self.motion_empty_token_id = 512*2 + 7
            self.interleave_ratio = [1, 1]
    
    return TestConfig()


def evaluate_teacher_forcing(model, dataloader, device="cuda"):
    """Teacher-forcing模式评估：使用GT motion tokens，计算预测准确率"""
    model.eval()
    
    total_tokens = 0
    correct_tokens = 0
    total_loss = 0.0
    num_batches = 0
    
    special_token_ids = {
        512*2 + 2,  # gesture_start_token_id
        512*2 + 3,  # audio_gesture_start_token_id
        512*2 + 4,  # gesture_end_token_id
        512*2 + 5,  # audio_gesture_end_token_id
    }
    motion_empty_token_id = 512*2 + 7
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Teacher-forcing评估"):
            inputs = batch['tokens'].to(device).long()
            masks = batch['mask'].to(device).long()
            lengths = batch['lengths']
            
            attn_mask = (inputs != 512*2 + 1).float().to(device)  # pad_token_id
            
            labels = inputs.clone().long()
            labels[masks == 0] = -100  # 非motion token位置设为-100
            
            outputs = model(inputs, labels=labels, attention_mask=attn_mask)
            loss = outputs.loss
            
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            
            # 计算准确率（只对motion token位置）
            # 注意：loss计算是shifted的，即logits[i]预测labels[i+1]
            # 所以accuracy计算也需要shift：使用logits[:, :-1, :]预测labels[:, 1:]
            if outputs.logits is not None:
                # Shift logits和labels以匹配loss的计算方式
                shift_logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
                shift_labels = labels[:, 1:]  # [batch, seq_len-1]
                shift_inputs = inputs[:, 1:]  # [batch, seq_len-1]
                shift_masks = masks[:, 1:]  # [batch, seq_len-1]
                
                pred_tokens = torch.argmax(shift_logits, dim=-1)  # [batch, seq_len-1]
                
                # 只统计motion token位置（shift_masks == 1的位置，且shift_labels != -100）
                # shift_labels != -100 表示这是需要预测的motion token位置
                motion_mask = (shift_masks == 1) & (shift_labels != -100)
                
                if motion_mask.any():
                    gt_tokens = shift_inputs[motion_mask]
                    pred_tokens_motion = pred_tokens[motion_mask]
                    
                    # 过滤special token和padding token
                    valid_mask = ~torch.isin(gt_tokens, torch.tensor(list(special_token_ids) + [motion_empty_token_id], device=device))
                    if valid_mask.any():
                        valid_gt = gt_tokens[valid_mask]
                        valid_pred = pred_tokens_motion[valid_mask]
                        
                        correct = (valid_gt == valid_pred).sum().item()
                        total = valid_mask.sum().item()
                        
                        correct_tokens += correct
                        total_tokens += total
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct_tokens': correct_tokens,
        'total_tokens': total_tokens
    }


def evaluate_free_running(model, dataloader, device="cuda", max_new_tokens=512, 
                          temperature=0.8, top_k=50, repetition_penalty=1.1):
    """Free-running模式评估：完全自回归生成，计算与GT的相似度"""
    model.eval()
    
    all_gt_tokens = []
    all_pred_tokens = []
    
    special_token_ids = {
        512*2 + 2, 512*2 + 3, 512*2 + 4, 512*2 + 5
    }
    motion_empty_token_id = 512*2 + 7
    audio_empty_token_id = 152063
    
    def generate_motion_tokens(audio_tokens, max_new_tokens=512):
        """从audio tokens生成motion tokens"""
        generated_motion_tokens = []
        current_seq = []
        token_labels = []
        generated_history = []
        
        interleave_audios, interleave_motions = 1, 1
        motion_padding_count = 0
        max_motion_padding = 10
        
        # 过滤special token
        audio_tokens_clean = [t for t in audio_tokens if t not in special_token_ids]
        audio_tokens_clean = audio_tokens_clean + [audio_empty_token_id] * 10
        
        with torch.no_grad():
            for i, audio_token in enumerate(audio_tokens_clean):
                current_seq.append(audio_token)
                token_labels.append(-100)
                
                if (i + 1) % interleave_audios == 0:
                    if motion_padding_count < max_motion_padding:
                        current_seq.append(motion_empty_token_id)
                        token_labels.append(-100)
                        motion_padding_count += 1
                    else:
                        for j in range(interleave_motions):
                            if len(generated_motion_tokens) >= max_new_tokens:
                                break
                            
                            if len(current_seq) >= 512:  # max_seq_length
                                break
                            
                            inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
                            if inputs.shape[1] > 512:
                                inputs = inputs[:, -512:]
                                attn_mask = torch.ones_like(inputs)
                                labels = torch.tensor(token_labels[-512:]).unsqueeze(0).to(device)
                            else:
                                attn_mask = torch.ones_like(inputs)
                                labels = torch.tensor(token_labels).unsqueeze(0).to(device)
                            
                            output = model(
                                input_data=inputs,
                                attention_mask=attn_mask,
                                labels=labels,
                                label_tokens=None,
                                use_label_prediction_mode=False
                            )
                            
                            if output.logits is None:
                                break
                            
                            next_token_logits = output.logits[0, -1, :]
                            
                            # 排除motion_empty_token_id
                            if motion_empty_token_id < next_token_logits.size(-1):
                                next_token_logits[motion_empty_token_id] = float('-inf')
                            
                            # 应用repetition penalty
                            if repetition_penalty != 1.0 and generated_history:
                                for token_id in set(generated_history):
                                    if token_id not in special_token_ids and token_id < next_token_logits.size(-1):
                                        next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                            
                            # 采样
                            if temperature > 0:
                                next_token_logits = next_token_logits / temperature
                                if top_k > 0:
                                    k = min(int(top_k), next_token_logits.size(-1))
                                    top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                                    masked = torch.full_like(next_token_logits, float('-inf'))
                                    masked[top_k_indices] = top_k_logits
                                    next_token_logits = masked
                                probs = torch.softmax(next_token_logits, dim=-1)
                                next_token = torch.multinomial(probs, 1).item()
                            else:
                                next_token = torch.argmax(next_token_logits, dim=-1).item()
                            
                            if next_token != motion_empty_token_id and next_token not in special_token_ids:
                                generated_motion_tokens.append(next_token)
                                generated_history.append(next_token)
                            
                            current_seq.append(next_token)
                            token_labels.append(next_token)
                            
                            if len(generated_history) > 100:
                                generated_history = generated_history[-100:]
                            
                            if len(generated_motion_tokens) >= max_new_tokens:
                                break
                    
                    if len(generated_motion_tokens) >= max_new_tokens:
                        break
        
        return generated_motion_tokens
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Free-running评估")):
            inputs = batch['tokens'].to(device).long()
            masks = batch['mask'].to(device).long()
            
            # 从batch中提取audio tokens和GT motion tokens
            for i in range(inputs.shape[0]):
                seq = inputs[i].cpu().tolist()
                mask = masks[i].cpu().tolist()
                length = batch['lengths'][i].item()
                
                # 提取audio tokens（mask == 0的位置）
                audio_tokens = [seq[j] for j in range(length) if mask[j] == 0]
                # 提取GT motion tokens（mask == 1的位置）
                gt_motion_tokens = [seq[j] for j in range(length) if mask[j] == 1]
                
                # 过滤special token
                gt_motion_tokens_clean = [t for t in gt_motion_tokens 
                                         if t not in special_token_ids and t != motion_empty_token_id
                                         and 0 <= t < 1024]
                
                if len(audio_tokens) > 0:
                    # 生成motion tokens
                    pred_motion_tokens = generate_motion_tokens(audio_tokens, max_new_tokens=max_new_tokens)
                    
                    all_gt_tokens.append(gt_motion_tokens_clean)
                    all_pred_tokens.append(pred_motion_tokens)
    
    # 计算BLEU-like相似度（简单的token匹配率）
    total_gt_tokens = sum(len(tokens) for tokens in all_gt_tokens)
    total_pred_tokens = sum(len(tokens) for tokens in all_pred_tokens)
    
    # 计算平均长度差异
    length_diffs = [abs(len(gt) - len(pred)) for gt, pred in zip(all_gt_tokens, all_pred_tokens)]
    avg_length_diff = np.mean(length_diffs) if length_diffs else 0.0
    
    return {
        'total_samples': len(all_gt_tokens),
        'total_gt_tokens': total_gt_tokens,
        'total_pred_tokens': total_pred_tokens,
        'avg_gt_length': total_gt_tokens / len(all_gt_tokens) if all_gt_tokens else 0.0,
        'avg_pred_length': total_pred_tokens / len(all_pred_tokens) if all_pred_tokens else 0.0,
        'avg_length_diff': avg_length_diff
    }


def main():
    parser = argparse.ArgumentParser(description='测试Motion Adaptor模型')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--test_jsonl', type=str, nargs='+', required=True,
                       help='测试集JSONL文件路径（可以指定多个）')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='测试样本数量（None表示使用全部）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint_path, device)
    
    print(f"\n{'='*80}")
    print(f"开始测试模型")
    print(f"{'='*80}\n")
    
    # 存储所有测试集的结果
    all_results = {}
    config = create_test_config()
    
    # 对每个测试集进行评估
    for test_jsonl_path in args.test_jsonl:
        print(f"\n{'='*80}")
        print(f"测试集: {test_jsonl_path}")
        print(f"{'='*80}\n")
        
        # 创建测试数据集
        dataset = JSONLAudioMotionDataset(test_jsonl_path, config)
        total_samples = len(dataset)
        
        if args.num_samples is not None and args.num_samples < total_samples:
            from torch.utils.data import Subset
            indices = list(range(min(args.num_samples, total_samples)))
            dataset = Subset(dataset, indices)
            print(f"📊 使用 {len(dataset)} 个测试样本（总共 {total_samples} 个）")
        else:
            print(f"📊 使用全部 {total_samples} 个测试样本")
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                'tokens': torch.stack([item['tokens'] for item in batch]).long(),
                'mask': torch.stack([item['mask'] for item in batch]).long(),
                'lengths': torch.tensor([item['seq_length'] for item in batch])
            },
            num_workers=4
        )
        
        # Teacher-forcing评估
        print("📊 Teacher-forcing模式评估...")
        tf_results = evaluate_teacher_forcing(model, dataloader, device)
        print(f"  Loss: {tf_results['loss']:.4f}")
        print(f"  Accuracy: {tf_results['accuracy']:.4f} ({tf_results['correct_tokens']}/{tf_results['total_tokens']})")
        
        # 保存该测试集的结果
        dataset_name = os.path.basename(test_jsonl_path).replace('.jsonl', '')
        all_results[dataset_name] = {
            'test_jsonl': test_jsonl_path,
            'teacher_forcing': tf_results
        }
    
    # 保存所有结果
    results = {
        'checkpoint_path': args.checkpoint_path,
        'test_datasets': all_results
    }
    
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print(f"测试汇总")
    print(f"{'='*80}\n")
    for dataset_name, dataset_results in all_results.items():
        tf = dataset_results['teacher_forcing']
        print(f"{dataset_name}:")
        print(f"  Loss: {tf['loss']:.4f}")
        print(f"  Accuracy: {tf['accuracy']:.4f} ({tf['correct_tokens']}/{tf['total_tokens']})")
        print()
    
    print(f"✅ 测试完成！结果已保存到: {results_path}")


if __name__ == '__main__':
    main()

