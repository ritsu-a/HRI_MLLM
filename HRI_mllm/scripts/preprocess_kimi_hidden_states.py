#!/usr/bin/env python3
"""
预处理Kimi模型的hidden states
读取训练数据，使用Kimi模型处理并保存hidden states，加速训练
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

# 设置环境
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm.model.kimi_motion.model import MoonshotKimiaForCausalLM


def process_jsonl_and_save_hidden_states(
    input_jsonl_path: str,
    output_dir: str,
    kimi_model_path: str,
    batch_size: int = 4,
    max_audio_length: int = 1024,
    device: str = "cuda",
    num_workers: int = 0
):
    """
    处理JSONL文件，使用Kimi模型提取hidden states并保存
    
    Args:
        input_jsonl_path: 输入JSONL文件路径
        output_dir: 输出目录（保存预处理后的数据）
        kimi_model_path: Kimi模型路径
        batch_size: 批次大小
        max_audio_length: 最大音频token长度
        device: 设备
        num_workers: 数据加载工作进程数
    """
    
    print(f"🚀 Starting preprocessing...")
    print(f"   - Input: {input_jsonl_path}")
    print(f"   - Output: {output_dir}")
    print(f"   - Kimi model: {kimi_model_path}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Max audio length: {max_audio_length}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "hidden_states"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
    
    # 加载Kimi模型
    print(f"🔄 Loading Kimi model...")
    kimi_model = MoonshotKimiaForCausalLM.from_pretrained(
        kimi_model_path,
        trust_remote_code=True
    )
    kimi_model.to(device)
    kimi_model.eval()  # 设置为评估模式
    
    # 加载tokenizer
    print(f"🔄 Loading tokenizer...")
    text_tokenizer = AutoTokenizer.from_pretrained(
        kimi_model_path,
        trust_remote_code=True
    )
    
    # 获取kimia_text_blank token ID
    try:
        from kimia_infer.utils.special_tokens import instantiate_extra_tokens
        extra_tokens = instantiate_extra_tokens(text_tokenizer)
        kimia_text_blank = extra_tokens.kimia_text_blank
    except:
        # 如果没有找到，使用默认值（需要根据实际情况调整）
        kimia_text_blank = text_tokenizer.convert_tokens_to_ids("<|kimia_text_blank|>")
        if kimia_text_blank == text_tokenizer.unk_token_id:
            kimia_text_blank = 0  # 使用0作为fallback
    
    print(f"   - kimia_text_blank token ID: {kimia_text_blank}")
    
    # 读取JSONL文件
    print(f"🔄 Reading JSONL file...")
    samples = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error at line {line_num + 1}: {e}")
                    continue
    
    print(f"📊 Found {len(samples)} samples")
    
    # 处理每个样本
    processed_samples = []
    total_processed = 0
    total_skipped = 0
    
    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                # 提取数据
                user_text = None
                user_audio_tokens = None
                assistant_audio_tokens = None
                motion_tokens = None
                
                if 'conversation' in sample:
                    # 对话格式
                    for msg in sample['conversation']:
                        if msg.get('role') == 'user' and msg.get('message_type') == 'text':
                            user_text = msg.get('content')
                        elif msg.get('role') == 'user' and msg.get('message_type') == 'audio' and 'audio_tokens' in msg:
                            user_audio_tokens = msg['audio_tokens']
                        elif msg.get('role') == 'assistant' and msg.get('message_type') == 'audio_motion':
                            if 'audio_tokens' in msg:
                                assistant_audio_tokens = msg['audio_tokens']
                            if 'motion_tokens' in msg:
                                motion_tokens = msg['motion_tokens']
                else:
                    # 直接格式
                    user_audio_tokens = sample.get('user_audio_tokens') or sample.get('audio_tokens')
                    assistant_audio_tokens = sample.get('assistant_audio_tokens')
                    motion_tokens = sample.get('motion_tokens')
                
                # 验证必要字段
                if user_audio_tokens is None or assistant_audio_tokens is None or motion_tokens is None:
                    total_skipped += 1
                    continue
                
                # 转换为tensor并截断
                if not isinstance(user_audio_tokens, torch.Tensor):
                    user_audio_tokens = torch.tensor(user_audio_tokens, dtype=torch.long)
                if not isinstance(assistant_audio_tokens, torch.Tensor):
                    assistant_audio_tokens = torch.tensor(assistant_audio_tokens, dtype=torch.long)
                if not isinstance(motion_tokens, torch.Tensor):
                    motion_tokens = torch.tensor(motion_tokens, dtype=torch.long)
                
                # 截断过长序列
                if len(user_audio_tokens) > max_audio_length:
                    user_audio_tokens = user_audio_tokens[:max_audio_length]
                if len(assistant_audio_tokens) > max_audio_length:
                    assistant_audio_tokens = assistant_audio_tokens[:max_audio_length]
                
                # 处理文本（如果存在）
                text_input_ids = None
                if user_text and user_text.strip():
                    try:
                        encoded = text_tokenizer.encode(user_text, bos=False, eos=False)
                        if len(encoded) > 512:
                            encoded = encoded[:512]
                        text_input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
                    except:
                        text_input_ids = None
                
                # 准备Kimi模型输入
                audio_input_ids = user_audio_tokens.unsqueeze(0).to(device)  # [1, seq_len]
                audio_seq_len = audio_input_ids.shape[1]
                
                # 创建text_input_ids（对齐长度）
                if text_input_ids is None:
                    text_input_ids = torch.full(
                        (1, audio_seq_len),
                        kimia_text_blank,
                        device=device,
                        dtype=torch.long
                    )
                else:
                    # 对齐长度
                    text_seq_len = text_input_ids.shape[1]
                    if text_seq_len != audio_seq_len:
                        padded_text = torch.full(
                            (1, audio_seq_len),
                            kimia_text_blank,
                            device=device,
                            dtype=torch.long
                        )
                        actual_len = min(text_seq_len, audio_seq_len)
                        padded_text[:, :actual_len] = text_input_ids[:, :actual_len]
                        text_input_ids = padded_text
                
                # 创建attention mask
                attention_mask = torch.ones(1, audio_seq_len, device=device, dtype=torch.long)
                
                # 调用Kimi模型获取hidden states
                kimi_outputs = kimi_model(
                    input_ids=audio_input_ids,
                    text_input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 提取hidden states
                if hasattr(kimi_outputs, 'hidden_states') and kimi_outputs.hidden_states:
                    # 获取最后一层的hidden states
                    last_hidden_states = kimi_outputs.hidden_states[-1]
                    
                    if isinstance(last_hidden_states, tuple):
                        text_hidden_states, audio_hidden_states = last_hidden_states[0], last_hidden_states[1]
                    else:
                        # 如果只有一个，两个都使用它
                        text_hidden_states = last_hidden_states
                        audio_hidden_states = last_hidden_states
                else:
                    # 从logits反推（降级方案）
                    text_hidden_states = audio_hidden_states = None
                
                if text_hidden_states is None or audio_hidden_states is None:
                    total_skipped += 1
                    continue
                
                # 转换为CPU并保存
                text_hidden_states = text_hidden_states.cpu().squeeze(0)  # [seq_len, hidden_size]
                audio_hidden_states = audio_hidden_states.cpu().squeeze(0)  # [seq_len, hidden_size]
                
                # 保存hidden states
                hidden_states_path = os.path.join(
                    output_dir,
                    "hidden_states",
                    f"sample_{sample_idx:06d}.pt"
                )
                torch.save({
                    'text_hidden_states': text_hidden_states,
                    'audio_hidden_states': audio_hidden_states,
                }, hidden_states_path)
                
                # 保存元数据
                metadata = {
                    'sample_idx': sample_idx,
                    'user_text': user_text,
                    'user_audio_tokens': user_audio_tokens.cpu().tolist(),
                    'assistant_audio_tokens': assistant_audio_tokens.cpu().tolist(),
                    'motion_tokens': motion_tokens.cpu().tolist(),
                    'user_audio_length': len(user_audio_tokens),
                    'assistant_audio_length': len(assistant_audio_tokens),
                    'motion_length': len(motion_tokens),
                    'hidden_states_path': hidden_states_path,
                    'text_hidden_size': text_hidden_states.shape[-1],
                    'audio_hidden_size': audio_hidden_states.shape[-1],
                }
                
                metadata_path = os.path.join(
                    output_dir,
                    "metadata",
                    f"sample_{sample_idx:06d}.json"
                )
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                processed_samples.append(metadata)
                total_processed += 1
                
            except Exception as e:
                print(f"⚠️  Error processing sample {sample_idx}: {e}")
                total_skipped += 1
                continue
    
    # 保存索引文件
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(samples),
            'processed_samples': total_processed,
            'skipped_samples': total_skipped,
            'samples': processed_samples
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Preprocessing completed!")
    print(f"   - Total samples: {len(samples)}")
    print(f"   - Processed: {total_processed}")
    print(f"   - Skipped: {total_skipped}")
    print(f"   - Output directory: {output_dir}")
    print(f"   - Index file: {index_path}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Kimi hidden states')
    parser.add_argument('--input_jsonl', type=str, required=True,
                       help='Input JSONL file path')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for preprocessed data')
    parser.add_argument('--kimi_model_path', type=str, default="moonshotai/Kimi-Audio-7B",
                       help='Kimi model path')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (currently supports 1)')
    parser.add_argument('--max_audio_length', type=int, default=1024,
                       help='Max audio token length')
    parser.add_argument('--device', type=str, default="cuda",
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    process_jsonl_and_save_hidden_states(
        input_jsonl_path=args.input_jsonl,
        output_dir=args.output_dir,
        kimi_model_path=args.kimi_model_path,
        batch_size=args.batch_size,
        max_audio_length=args.max_audio_length,
        device=args.device
    )


if __name__ == "__main__":
    main()

