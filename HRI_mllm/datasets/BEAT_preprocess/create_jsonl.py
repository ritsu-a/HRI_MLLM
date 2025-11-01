import argparse
import numpy as np
import time
from tqdm import tqdm
import os
import pickle
import soundfile as sf
import torch
import yaml
import json
from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT


def find_audio_file(base_name, data_dir, dataset_name):
    """查找音频文件"""
    # 尝试多个可能的音频目录
    audio_dirs = []
    
    if "beat" in dataset_name.lower():
        # BEAT相关数据集
        audio_dirs.append(os.path.join(DATA_ROOT, "BEAT_v2"))
        audio_dirs.append(os.path.join(DATA_ROOT, "beat_english_v0.2.1"))
    else:
        # internet_data或其他
        audio_dirs.append(os.path.join(DATA_ROOT, "internet_data_1021"))
        audio_dirs.append(os.path.join(DATA_ROOT, "internet_data_v1_kimi"))
    
    for audio_dir in audio_dirs:
        if not os.path.exists(audio_dir):
            continue
        
        # 尝试直接匹配
        for ext in ['.wav', '.mp3']:
            potential_audio = os.path.join(audio_dir, base_name + ext)
            if os.path.exists(potential_audio):
                return potential_audio
        
        # BEAT特殊格式：{number}/filename.wav
        if "_" in base_name:
            parts = base_name.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                potential_audio = os.path.join(audio_dir, parts[0], base_name + '.wav')
                if os.path.exists(potential_audio):
                    return potential_audio
        
        # 递归搜索子目录
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.startswith(base_name) and file.endswith(('.wav', '.mp3')):
                    return os.path.join(root, file)
    
    return None


def process_dataset(data_dir, dataset_name, output_file, user_prompt):
    """处理单个数据集"""
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*80}\n")
    
    tokens_dir = os.path.join(data_dir, "tokens")
    if not os.path.exists(tokens_dir):
        print(f"❌ Tokens directory not found: {tokens_dir}")
        return 0
    
    # 获取所有body token文件
    body_files = sorted([f for f in os.listdir(tokens_dir) if f.endswith('_body_tokens.pt')])
    
    if len(body_files) == 0:
        print(f"⚠️  No body token files found")
        return 0
    
    print(f"Found {len(body_files)} samples\n")
    
    metadata = []
    success_count = 0
    skip_count = 0
    
    for body_file in tqdm(body_files, desc=f"Processing {dataset_name}"):
        try:
            base_name = body_file.replace('_body_tokens.pt', '')
            
            body_path = os.path.join(tokens_dir, body_file)
            hand_path = os.path.join(tokens_dir, base_name + '_hand_tokens.pt')
            audio_path = os.path.join(tokens_dir, base_name + '_audio_tokens.pt')
            
            # 检查文件是否存在
            if not os.path.exists(hand_path):
                skip_count += 1
                continue
            if not os.path.exists(audio_path):
                skip_count += 1
                continue
            
            # 加载tokens
            body_tokens = torch.load(body_path)
            hand_tokens = torch.load(hand_path)
            audio_tokens = torch.load(audio_path)
            
            # 转换为numpy并展平
            if isinstance(body_tokens, torch.Tensor):
                body_tokens = body_tokens.cpu().numpy().reshape(-1)
            else:
                body_tokens = np.array(body_tokens).reshape(-1)
            
            if isinstance(hand_tokens, torch.Tensor):
                hand_tokens = hand_tokens.cpu().numpy().reshape(-1)
            else:
                hand_tokens = np.array(hand_tokens).reshape(-1)
            
            if isinstance(audio_tokens, torch.Tensor):
                audio_tokens = audio_tokens.cpu().numpy().reshape(-1)
            else:
                audio_tokens = np.array(audio_tokens).reshape(-1)
            
            # 合并body和hand tokens到统一的codebook
            # body tokens: [0, code_num-1] = [0, 511]
            # hand tokens: [code_num, 2*code_num-1] = [512, 1023] (已在generate_motion_tokens.py中添加偏移)
            # 最终motion tokens: body[0], hand[0], body[1], hand[1], ... (逐对交替)
            # 统一codebook范围: [0, 1023]
            motion_tokens = []
            max_len = max(len(body_tokens), len(hand_tokens))
            for i in range(max_len):
                if i < len(body_tokens):
                    motion_tokens.append(int(body_tokens[i]))  # body token: [0, 511]
                if i < len(hand_tokens):
                    # hand tokens已经包含code_num偏移 (hand_code + code_num)
                    motion_tokens.append(int(hand_tokens[i]))  # hand token: [512, 1023]
            
            # 查找音频文件
            wav_path = find_audio_file(base_name, data_dir, dataset_name)
            if wav_path is None:
                # 如果找不到音频文件，尝试从目录中提取
                for audio_dir in [os.path.join(DATA_ROOT, "single_motion_sentence_version2")]:
                    if os.path.exists(audio_dir):
                        wav_path = os.path.join(audio_dir, base_name + '.wav')
                        if os.path.exists(wav_path):
                            break
            
            if wav_path is None:
                # 使用placeholder
                wav_path = f"<audio_path_for_{base_name}>"
            
            # 转换为list
            audio_tokens = audio_tokens.tolist()
            
            metadata.append({
                "task_type": "s2s",
                "conversation": [
                    {
                        "role": "user",
                        'message_type': 'text',
                        "content": user_prompt
                    },
                    {
                        "role": "user",
                        'message_type': 'audio', 
                        'content': wav_path,
                        'audio_tokens': audio_tokens
                    },
                    {
                        "role": "assistant",
                        "message_type": "audio_motion",
                        'audio_tokens': audio_tokens,
                        'content': wav_path,
                        'motion_tokens': motion_tokens
                    },
                ]
            })
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {body_file}: {e}")
            skip_count += 1
            continue
    
    # 保存到输出文件
    if metadata:
        with open(output_file, 'a', encoding='utf-8') as f:
            for data in metadata:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Completed: {success_count}/{len(body_files)} successful, {skip_count} skipped")
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create jsonl files from token data')
    parser.add_argument("--datasets", type=str, nargs='+', 
                       default=["single_motion_sentence_version2_kimi"],
                       help="List of datasets to process")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for jsonl files (default: DATA_ROOT)")
    parser.add_argument("--user_prompt", type=str, 
                       default="Please repeat the following spoken content with a corresponding full-body motion sequence.")
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = DATA_ROOT
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    print(f"User prompt: {args.user_prompt}\n")
    
    total_success = 0
    for dataset_name in args.datasets:
        data_dir = os.path.join(DATA_ROOT, dataset_name)
        if not os.path.exists(data_dir):
            print(f"⚠️  Directory not found: {data_dir}")
            continue
        
        # 为每个数据集创建独立的输出文件
        output_file = os.path.join(args.output_dir, f"{dataset_name}_tokens.jsonl")
        
        # 创建输出文件（清空或创建）
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('')
        
        print(f"\nProcessing {dataset_name} -> {output_file}")
        
        count = process_dataset(data_dir, dataset_name, output_file, args.user_prompt)
        total_success += count
        
        print(f"✅ {dataset_name}: {count} samples saved to {output_file}")
    
    print(f"\n{'='*80}")
    print(f"🎉 All processing completed!")
    print(f"Total samples processed: {total_success}")
    print(f"Output files saved to: {args.output_dir}")
    print(f"{'='*80}")