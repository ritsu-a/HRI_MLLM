import argparse
import json
import os
import shutil
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from kimia_infer.api.prompt_manager import KimiAPromptManager
from kimia_infer.api.kimia import KimiAudio


def load_jsonl(file_path):
    """加载jsonl文件"""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def decode_audio_tokens(audio_tokens, kimi_audio_model, kimia_token_offset, output_path):
    """
    解码音频tokens并保存为音频文件
    
    Args:
        audio_tokens: list of int, audio token ids
        kimi_audio_model: KimiAudio实例
        kimia_token_offset: token offset
        output_path: 输出音频文件路径
    """
    try:
        # 将tokens转换为tensor
        # audio_tokens应该是原始的token ids（包含offset）
        tokens = torch.tensor(audio_tokens, dtype=torch.long)
        
        # 减去offset，因为detokenize_audio需要的是去掉offset的tokens
        adjusted_tokens = tokens - kimia_token_offset
        
        # 添加batch维度
        tokens_tensor = adjusted_tokens.unsqueeze(0).to(torch.cuda.current_device())
        
        # 使用 KimiAudio 的 detokenize_audio 方法解码
        with torch.no_grad():
            audio_waveform = kimi_audio_model.detokenize_audio(tokens_tensor)
        
        # audio_waveform 应该是 tensor (1, samples) 或 (samples,)
        if isinstance(audio_waveform, torch.Tensor):
            # 确保是 2D tensor (channels, samples)
            if audio_waveform.dim() == 1:
                audio_waveform = audio_waveform.unsqueeze(0)
            elif audio_waveform.dim() == 3:
                audio_waveform = audio_waveform.squeeze(0)
            
            # 保存为 wav 文件，采样率 24000
            torchaudio.save(output_path, audio_waveform.cpu(), sample_rate=24000)
            return True
        else:
            print(f"  ⚠️  Unexpected audio_waveform type: {type(audio_waveform)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error decoding audio: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_entry(entry, kimi_audio_model, kimia_token_offset, output_base_dir, copy_original=True):
    """
    处理单个jsonl条目，解码音频并拷贝原始音频
    
    Args:
        entry: jsonl条目
        kimi_audio_model: KimiAudio实例
        kimia_token_offset: token offset
        output_base_dir: 输出基础目录
        copy_original: 是否拷贝原始音频
    """
    conversation = entry['conversation']
    
    # 获取用户音频和助手音频
    user_audio_msg = conversation[1]
    assistant_audio_msg = conversation[2]
    
    user_audio_path = user_audio_msg['content']
    user_audio_tokens = user_audio_msg['audio_tokens']
    
    assistant_audio_path = assistant_audio_msg['content']
    assistant_audio_tokens = assistant_audio_msg['audio_tokens']
    
    # 提取文件名
    user_audio_basename = os.path.basename(user_audio_path)  # 717_A.wav
    assistant_audio_basename = os.path.basename(assistant_audio_path)  # 717_B.wav
    
    # 提取 ID
    file_id = user_audio_basename.replace('_A.wav', '')  # 717
    
    # 创建输出子目录
    output_dir = os.path.join(output_base_dir, file_id)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'id': file_id,
        'user_audio_decoded': False,
        'assistant_audio_decoded': False,
        'user_audio_copied': False,
        'assistant_audio_copied': False
    }
    
    # 拷贝原始音频
    if copy_original:
        # 拷贝用户音频 (A)
        if os.path.exists(user_audio_path):
            original_user_path = os.path.join(output_dir, f"{file_id}_A_original.wav")
            shutil.copy2(user_audio_path, original_user_path)
            results['user_audio_copied'] = True
        
        # 拷贝助手音频 (B)
        if os.path.exists(assistant_audio_path):
            original_assistant_path = os.path.join(output_dir, f"{file_id}_B_original.wav")
            shutil.copy2(assistant_audio_path, original_assistant_path)
            results['assistant_audio_copied'] = True
    
    # 解码用户音频
    decoded_user_path = os.path.join(output_dir, f"{file_id}_A_decoded.wav")
    results['user_audio_decoded'] = decode_audio_tokens(
        user_audio_tokens, kimi_audio_model, kimia_token_offset, decoded_user_path
    )
    
    # 解码助手音频
    decoded_assistant_path = os.path.join(output_dir, f"{file_id}_B_decoded.wav")
    results['assistant_audio_decoded'] = decode_audio_tokens(
        assistant_audio_tokens, kimi_audio_model, kimia_token_offset, decoded_assistant_path
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode audio tokens from HAND_RING jsonl')
    parser.add_argument("--jsonl_file", type=str, 
                       default="/root/workspace/HRI_MLLM/data/HAND_RING_1111_tokens.jsonl",
                       help="Path to jsonl file")
    parser.add_argument("--model_name_or_path", type=str, 
                       default="moonshotai/Kimi-Audio-7B",
                       help="Kimi model path")
    parser.add_argument("--output_dir", type=str,
                       default="/root/workspace/HRI_MLLM/data/HAND_RING_decoded_audio",
                       help="Output directory for decoded audio")
    parser.add_argument("--max_entries", type=int, default=3,
                       help="Maximum number of entries to process (default: 3 for testing)")
    parser.add_argument("--copy_original", action="store_true", default=True,
                       help="Copy original audio files for comparison")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 KimiAudio 模型
    print(f"Loading Kimi model from: {args.model_name_or_path}")
    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)
    
    # 加载 KimiAudio 用于解码音频
    kimi_audio_model = KimiAudio(model_path=cache_path, load_detokenizer=True)
    
    # 获取 token offset
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    kimia_token_offset = model_config.kimia_token_offset
    
    print(f"✅ Kimi model loaded successfully!")
    print(f"   Token offset: {kimia_token_offset}")
    
    # 加载jsonl文件
    print(f"\nLoading jsonl file: {args.jsonl_file}")
    entries = load_jsonl(args.jsonl_file)
    print(f"Loaded {len(entries)} entries")
    
    # 处理条目
    max_entries = min(args.max_entries, len(entries)) if args.max_entries else len(entries)
    print(f"\nProcessing {max_entries} entries...")
    print(f"Output directory: {args.output_dir}\n")
    
    success_count = 0
    
    for i, entry in enumerate(tqdm(entries[:max_entries], desc="Decoding audio")):
        try:
            results = process_entry(entry, kimi_audio_model, kimia_token_offset, args.output_dir, args.copy_original)
            
            if results['user_audio_decoded'] and results['assistant_audio_decoded']:
                success_count += 1
            
            # 打印结果
            print(f"\n[{results['id']}]")
            if args.copy_original:
                print(f"  Original copied: User={'✅' if results['user_audio_copied'] else '❌'}, "
                      f"Assistant={'✅' if results['assistant_audio_copied'] else '❌'}")
            print(f"  Decoded: User={'✅' if results['user_audio_decoded'] else '❌'}, "
                  f"Assistant={'✅' if results['assistant_audio_decoded'] else '❌'}")
            
        except Exception as e:
            print(f"\n❌ Error processing entry {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Decoding Summary")
    print(f"{'='*80}")
    print(f"Total entries processed: {max_entries}")
    print(f"Successfully decoded: {success_count}")
    print(f"Failed: {max_entries - success_count}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")

