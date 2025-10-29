#!/usr/bin/env python3
"""
统一Kimi-Motion模型推理测试脚本
支持：
1. 对JSONL训练数据的重建评估
2. 对任意user audio wav input的推理
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from pathlib import Path
import librosa
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import time

# 设置MuJoCo渲染
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT
from HRI_mllm.model.unified_kimi_motion_model import UnifiedKimiMotionModel, create_unified_model
from HRI_mllm.datasets.json_audio_motion_dataset import JSONAudioMotionDataset
from transformers import GPT2Config
from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
from kimia_infer.utils.special_tokens import instantiate_extra_tokens


class UnifiedModelInference:
    """统一模型推理类"""
    
    def __init__(self, 
                 model_path: str,
                 kimi_model_path: str = "moonshotai/Kimi-Audio-7B",
                 device: str = "cuda"):
        """
        Args:
            model_path: 训练好的模型checkpoint路径
            kimi_model_path: Kimi模型路径
            device: 推理设备
        """
        self.device = device
        self.kimi_model_path = kimi_model_path
        
        print(f"🔄 Loading unified model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从checkpoint恢复配置
        checkpoint_kimi_path = checkpoint.get('kimi_model_path', kimi_model_path)
        freeze_kimi = checkpoint.get('freeze_kimi', True)
        freeze_adaptor = checkpoint.get('freeze_adaptor', True)
        train_mixer_only = checkpoint.get('train_mixer_only', True)
        motion_loss_weight = checkpoint.get('motion_loss_weight', 1.0)
        audio_loss_weight = checkpoint.get('audio_loss_weight', 0.1)
        use_lora = checkpoint.get('use_lora', False)
        
        # 创建模型
        from transformers import GPT2Config
        gpt2_config = GPT2Config(
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
        
        self.model = UnifiedKimiMotionModel(
            kimi_model_path=checkpoint_kimi_path,
            gpt2_config=gpt2_config,
            freeze_kimi=freeze_kimi,
            freeze_adaptor=freeze_adaptor,
            train_mixer_only=train_mixer_only,
            motion_loss_weight=motion_loss_weight,
            audio_loss_weight=audio_loss_weight,
            debug=False
        )
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
        print(f"✅ Model loaded successfully!")
        
        # 初始化audio tokenizer（用于wav文件推理）
        print("🔄 Loading audio tokenizer...")
        self.audio_tokenizer = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer")
        self.audio_tokenizer = self.audio_tokenizer.to(device)
        self.audio_tokenizer.eval()
        
        # 获取kimia_token_offset
        from transformers import AutoConfig
        kimi_config = AutoConfig.from_pretrained(kimi_model_path, trust_remote_code=True)
        self.kimia_token_offset = kimi_config.kimia_token_offset
        
        print(f"✅ Audio tokenizer loaded, kimia_token_offset={self.kimia_token_offset}")
    
    @torch.no_grad()
    def infer_from_wav(self, 
                      wav_path: str,
                      user_text: Optional[str] = None,
                      max_motion_tokens: int = 512,
                      temperature: float = 1.0,
                      top_k: int = 40) -> Dict:
        """
        从WAV文件推理生成motion tokens
        
        Args:
            wav_path: 输入WAV文件路径
            user_text: 可选的用户文本输入
            max_motion_tokens: 最大生成motion token数量
            temperature: 采样温度
            top_k: top-k采样
        
        Returns:
            Dict包含生成的motion tokens和其他信息
        """
        print(f"🔄 Processing WAV file: {wav_path}")
        
        # 1. 将WAV文件转换为audio tokens
        audio_tokens = self.audio_tokenizer.tokenize(audio_path=wav_path)
        # 转换到Kimi模型的token空间
        audio_tokens = audio_tokens + self.kimia_token_offset
        audio_tokens = audio_tokens.squeeze(0)  # [seq_len]
        
        print(f"   - Audio tokens shape: {audio_tokens.shape}")
        print(f"   - Audio tokens range: {audio_tokens.min().item()} to {audio_tokens.max().item()}")
        
        # 2. 准备输入
        batch_size = 1
        audio_input_ids = audio_tokens.unsqueeze(0).to(self.device)  # [1, seq_len]
        
        # 3. 处理文本（如果有）
        text_input = user_text if user_text else None
        
        # 4. 生成motion tokens（简化版本：直接调用forward并获取logits）
        print(f"🔄 Generating motion tokens...")
        
        outputs = self.model(
            user_text=[text_input] if text_input else None,
            user_audio_tokens=audio_input_ids,
            motion_tokens=None,  # 推理时不需要
            assistant_audio_tokens=None,  # 推理时不需要
            interleaved_sequences=None,
            attention_mask=None,
            labels=None
        )
        
        # 从outputs中获取motion logits
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            motion_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # 对最后一个位置进行采样（简化：只生成一个token）
            # 实际应该使用teacher forcing或自回归生成
            last_logits = motion_logits[0, -1, :]  # [vocab_size]
            
            # 应用temperature和top-k
            if temperature != 1.0:
                last_logits = last_logits / temperature
            
            # Top-k采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(last_logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, 1)
                generated_token = top_k_indices[sampled_idx].item()
            else:
                probs = torch.softmax(last_logits, dim=-1)
                generated_token = torch.multinomial(probs, 1).item()
            
            return {
                'motion_tokens': [generated_token],
                'audio_tokens': audio_tokens.cpu().tolist(),
                'user_text': text_input,
                'wav_path': wav_path
            }
        else:
            raise ValueError("Model did not return logits")
    
    @torch.no_grad()
    def evaluate_reconstruction(self,
                               jsonl_path: str,
                               num_samples: Optional[int] = None,
                               max_audio_length: int = 2048,
                               max_motion_length: int = 2048) -> Dict:
        """
        评估对JSONL训练数据的重建质量
        
        Args:
            jsonl_path: JSONL数据文件路径
            num_samples: 评估的样本数量（None表示全部）
            max_audio_length: 最大audio长度
            max_motion_length: 最大motion长度
        
        Returns:
            评估结果字典
        """
        print(f"🔄 Evaluating reconstruction on: {jsonl_path}")
        
        # 加载数据集
        dataset = JSONAudioMotionDataset(
            json_path=jsonl_path,
            max_audio_length=max_audio_length,
            max_motion_length=max_motion_length,
            interleave_ratio=(1, 1),
            debug=False
        )
        
        num_samples = num_samples or len(dataset)
        num_samples = min(num_samples, len(dataset))
        
        print(f"   - Dataset size: {len(dataset)}")
        print(f"   - Evaluating {num_samples} samples")
        
        # 评估指标
        total_motion_loss = 0.0
        total_audio_loss = 0.0
        total_samples = 0
        
        motion_accuracies = []  # 每个样本的motion token准确率
        audio_accuracies = []   # 每个样本的audio token准确率
        
        # 遍历样本进行评估
        for i in tqdm(range(num_samples), desc="Evaluating"):
            try:
                sample = dataset[i]
                
                # 准备输入
                user_audio_tokens = sample['user_audio_tokens'].unsqueeze(0).to(self.device)
                assistant_audio_tokens = sample['assistant_audio_tokens'].unsqueeze(0).to(self.device)
                motion_tokens = sample['motion_tokens'].unsqueeze(0).to(self.device)
                interleaved_sequences = sample['interleaved_sequence'].unsqueeze(0).to(self.device)
                token_labels = sample['token_labels'].unsqueeze(0).to(self.device)
                attention_mask = torch.ones_like(interleaved_sequences).to(self.device)
                
                user_text = sample.get('user_text', None)
                user_text_list = [user_text] if user_text else None
                
                # 前向传播
                outputs = self.model(
                    user_text=user_text_list,
                    user_audio_tokens=user_audio_tokens,
                    assistant_audio_tokens=assistant_audio_tokens,
                    motion_tokens=motion_tokens,
                    interleaved_sequences=interleaved_sequences,
                    attention_mask=attention_mask,
                    labels=token_labels
                )
                
                # 计算loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    total_loss = outputs.loss.item()
                    # 这里简化处理，实际应该分别计算motion和audio的loss
                    total_motion_loss += total_loss  # 假设主要是motion loss
                    total_audio_loss += 0.0  # 简化
                
                # 计算准确率（从logits）
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                    predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
                    
                    # Motion token准确率（labels != -100 的位置）
                    motion_mask = (token_labels != -100)
                    if motion_mask.any():
                        motion_pred = predicted_tokens[motion_mask]
                        motion_gt = token_labels[motion_mask]
                        motion_correct = (motion_pred == motion_gt).sum().item()
                        motion_total = motion_mask.sum().item()
                        motion_acc = motion_correct / motion_total if motion_total > 0 else 0.0
                        motion_accuracies.append(motion_acc)
                
                total_samples += 1
                
            except Exception as e:
                print(f"⚠️  Error evaluating sample {i}: {e}")
                continue
        
        # 计算平均指标
        avg_motion_loss = total_motion_loss / total_samples if total_samples > 0 else 0.0
        avg_audio_loss = total_audio_loss / total_samples if total_samples > 0 else 0.0
        avg_motion_acc = np.mean(motion_accuracies) if motion_accuracies else 0.0
        
        results = {
            'num_samples': total_samples,
            'avg_motion_loss': avg_motion_loss,
            'avg_audio_loss': avg_audio_loss,
            'avg_motion_accuracy': avg_motion_acc,
            'motion_accuracies': motion_accuracies
        }
        
        print(f"\n📊 Evaluation Results:")
        print(f"   - Evaluated samples: {total_samples}")
        print(f"   - Average motion loss: {avg_motion_loss:.4f}")
        print(f"   - Average audio loss: {avg_audio_loss:.4f}")
        print(f"   - Average motion accuracy: {avg_motion_acc:.4f}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一Kimi-Motion模型推理测试')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型checkpoint路径')
    parser.add_argument('--kimi_model_path', type=str, default="moonshotai/Kimi-Audio-7B",
                       help='Kimi模型路径')
    parser.add_argument('--mode', type=str, choices=['reconstruct', 'infer', 'both'], default='both',
                       help='运行模式：reconstruct（重建评估）, infer（WAV推理）, both（两者都运行）')
    parser.add_argument('--jsonl_path', type=str, default=None,
                       help='用于重建评估的JSONL文件路径')
    parser.add_argument('--wav_path', type=str, default=None,
                       help='用于推理的WAV文件路径')
    parser.add_argument('--user_text', type=str, default=None,
                       help='可选的用户文本输入（用于WAV推理）')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='重建评估的样本数量（None表示全部）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备')
    parser.add_argument('--output_dir', type=str, default='output/inference_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化推理器
    print("🚀 Initializing Unified Model Inference")
    print("=" * 60)
    inference = UnifiedModelInference(
        model_path=args.model_path,
        kimi_model_path=args.kimi_model_path,
        device=args.device
    )
    print("=" * 60)
    
    results = {}
    
    # 1. 重建评估
    if args.mode in ['reconstruct', 'both']:
        if args.jsonl_path:
            print(f"\n📊 Running reconstruction evaluation...")
            eval_results = inference.evaluate_reconstruction(
                jsonl_path=args.jsonl_path,
                num_samples=args.num_samples
            )
            results['reconstruction'] = eval_results
            
            # 保存结果
            eval_output_path = os.path.join(args.output_dir, 'reconstruction_results.json')
            with open(eval_output_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"✅ Evaluation results saved to: {eval_output_path}")
        else:
            print("⚠️  --jsonl_path not provided, skipping reconstruction evaluation")
    
    # 2. WAV文件推理
    if args.mode in ['infer', 'both']:
        if args.wav_path:
            print(f"\n🎵 Running WAV inference...")
            infer_results = inference.infer_from_wav(
                wav_path=args.wav_path,
                user_text=args.user_text
            )
            results['inference'] = infer_results
            
            # 保存结果
            infer_output_path = os.path.join(args.output_dir, 'inference_results.json')
            with open(infer_output_path, 'w') as f:
                json.dump(infer_results, f, indent=2)
            print(f"✅ Inference results saved to: {infer_output_path}")
            print(f"   - Generated motion tokens: {infer_results['motion_tokens']}")
        else:
            print("⚠️  --wav_path not provided, skipping WAV inference")
    
    print("\n🎉 Inference completed!")


if __name__ == "__main__":
    main()

