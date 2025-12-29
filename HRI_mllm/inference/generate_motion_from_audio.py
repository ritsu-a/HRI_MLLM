#!/usr/bin/env python3
"""
从audio tokens生成motion tokens的推理脚本

支持：
- 从checkpoint加载模型
- 从JSONL文件读取audio tokens或从音频文件生成audio tokens
- 自回归生成motion tokens
- 保存生成的motion tokens

使用示例：
    python HRI_mllm/inference/generate_motion_from_audio.py \
        --checkpoint_path output_disk0/motion_adaptor_v19_synthetic_data_en/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_300.pt \
        --audio_tokens_file input_audio_tokens.json \
        --output_file output_motion_tokens.json \
        --max_new_tokens 512 \
        --temperature 0.8 \
        --top_k 50
"""

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

from transformers import GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

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


def generate_motion_tokens(
    model,
    audio_tokens: List[int],
    device: str = "cuda",
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    max_seq_length: int = 512
) -> List[int]:
    """
    从audio tokens生成motion tokens
    
    Args:
        model: 训练好的Motion Adaptor模型
        audio_tokens: 输入的audio token序列
        device: 设备
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_k: Top-k采样
        repetition_penalty: 重复惩罚
        max_seq_length: 模型最大序列长度
    
    Returns:
        generated_motion_tokens: 生成的motion token序列
    """
    model.eval()
    
    # 特殊token IDs
    gesture_start_token_id = 512*2 + 2
    audio_gesture_start_token_id = 512*2 + 3
    gesture_end_token_id = 512*2 + 4
    audio_gesture_end_token_id = 512*2 + 5
    motion_empty_token_id = 512*2 + 7
    audio_empty_token_id = 152063
    
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    # 过滤输入中的special token
    audio_tokens_clean = [t for t in audio_tokens if t not in special_token_ids]
    
    # 按照训练时的格式：在audio tokens后面添加10个audio_empty_token
    audio_tokens_clean = audio_tokens_clean + [audio_empty_token_id] * 10
    
    # 限制max_new_tokens
    estimated_audio_length = len(audio_tokens_clean) + 10
    max_allowed_motion_tokens = max(1, max_seq_length - estimated_audio_length - 50)
    if max_new_tokens > max_allowed_motion_tokens:
        print(f"⚠️  警告: max_new_tokens ({max_new_tokens}) 超过模型容量。"
              f"限制为 {max_allowed_motion_tokens} (模型max_seq_length={max_seq_length})")
        max_new_tokens = max_allowed_motion_tokens
    
    generated_motion_tokens = []
    current_seq = []
    token_labels = []
    generated_history = []
    
    interleave_audios, interleave_motions = 1, 1
    motion_padding_count = 0
    max_motion_padding = 10
    
    with torch.no_grad():
        for i, audio_token in enumerate(audio_tokens_clean):
            current_seq.append(audio_token)
            token_labels.append(-100)  # audio token不计算损失
            
            if (i + 1) % interleave_audios == 0:
                # 前10个audio token位置，添加motion_empty_token
                if motion_padding_count < max_motion_padding:
                    current_seq.append(motion_empty_token_id)
                    token_labels.append(-100)
                    motion_padding_count += 1
                else:
                    # 生成真实的motion token
                    for j in range(interleave_motions):
                        if len(generated_motion_tokens) >= max_new_tokens:
                            break
                        
                        # 检查序列长度
                        if len(current_seq) >= max_seq_length:
                            print(f"⚠️  警告: 序列长度 ({len(current_seq)}) 达到模型最大长度 ({max_seq_length})。停止生成。")
                            break
                        
                        # 准备输入
                        inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
                        if inputs.shape[1] > max_seq_length:
                            inputs = inputs[:, -max_seq_length:]
                            attn_mask = torch.ones_like(inputs)
                            labels = torch.tensor(token_labels[-max_seq_length:]).unsqueeze(0).to(device)
                        else:
                            attn_mask = torch.ones_like(inputs)
                            labels = torch.tensor(token_labels).unsqueeze(0).to(device)
                        
                        # 模型前向传播
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
                        
                        # 应用重复惩罚
                        if repetition_penalty != 1.0 and generated_history:
                            for token_id in set(generated_history):
                                if token_id not in special_token_ids and token_id < next_token_logits.size(-1):
                                    next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                        
                        # 采样策略
                        if temperature > 0:
                            next_token_logits = next_token_logits / temperature
                            if top_k > 0:
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
                            # 贪心选择
                            if top_k > 0:
                                k = min(int(top_k), next_token_logits.size(-1))
                                top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
                                masked = torch.full_like(next_token_logits, float('-inf'))
                                masked[top_k_indices] = top_k_logits
                                next_token_logits = masked
                            next_token = torch.argmax(next_token_logits, dim=-1).item()
                        
                        # 处理特殊token
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
                            
                            # 只保存有效的motion token
                            if next_token != motion_empty_token_id:
                                generated_motion_tokens.append(next_token)
                                generated_history.append(next_token)
                        
                        # 添加到序列中
                        for token, label in zip(tokens_to_add, tokens_labels_to_add):
                            current_seq.append(token)
                            token_labels.append(label)
                        
                        # 限制历史记录长度
                        if len(generated_history) > 100:
                            generated_history = generated_history[-100:]
                        
                        if len(generated_motion_tokens) >= max_new_tokens:
                            break
                
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    # 过滤special token和motion_empty_token
    filtered_tokens = [t for t in generated_motion_tokens 
                      if t not in special_token_ids and t != motion_empty_token_id]
    
    return filtered_tokens


def load_audio_tokens_from_json(json_path: str) -> List[int]:
    """从JSON文件加载audio tokens"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持多种格式
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if 'audio_tokens' in data:
            return data['audio_tokens']
        elif 'tokens' in data:
            return data['tokens']
        else:
            raise ValueError(f"JSON文件中未找到audio_tokens或tokens字段")
    else:
        raise ValueError(f"不支持的JSON格式: {type(data)}")


def main():
    parser = argparse.ArgumentParser(description='从audio tokens生成motion tokens')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--audio_tokens_file', type=str, required=True,
                       help='输入audio tokens文件路径（JSON格式）')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出motion tokens文件路径（JSON格式）')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k采样')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='重复惩罚')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 加载模型
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint_path, device)
    
    # 加载audio tokens
    print(f"\n📂 加载audio tokens: {args.audio_tokens_file}")
    audio_tokens = load_audio_tokens_from_json(args.audio_tokens_file)
    print(f"   Audio tokens数量: {len(audio_tokens)}")
    
    # 生成motion tokens
    print(f"\n🔄 生成motion tokens...")
    print(f"   参数: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, "
          f"top_k={args.top_k}, repetition_penalty={args.repetition_penalty}")
    
    motion_tokens = generate_motion_tokens(
        model,
        audio_tokens,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty
    )
    
    print(f"✅ 生成完成！Motion tokens数量: {len(motion_tokens)}")
    
    # 保存结果
    output_data = {
        'audio_tokens': audio_tokens,
        'motion_tokens': motion_tokens,
        'generation_params': {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'repetition_penalty': args.repetition_penalty
        }
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 结果已保存到: {args.output_file}")


if __name__ == '__main__':
    main()




