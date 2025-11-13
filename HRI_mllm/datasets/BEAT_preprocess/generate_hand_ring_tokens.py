import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import yaml
import json
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats

# Audio tokenizer imports
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)


def process_motion_to_tokens(motion_data, motion_vae, mean_t, std_t, device, code_num=512):
    """
    将动作数据编码为tokens，body和hand分开编码（整段处理）
    
    Args:
        motion_data: numpy array, shape (T, 491)
        motion_vae: VQVaeBodyHand模型
        mean_t: 归一化均值
        std_t: 归一化标准差
        device: 计算设备
        code_num: codebook大小
        
    Returns:
        body_codes: torch.Tensor, body部分编码
        hand_codes: torch.Tensor, hand部分编码
    """
    # 转换为tensor并归一化
    data_tensor = torch.from_numpy(motion_data).unsqueeze(0).to(device).float()
    normalized_input = (data_tensor - mean_t) / std_t
    
    # 整段处理：不使用window_size，直接处理完整序列
    # 只确保序列长度满足模型的最小要求（通常为8帧）
    input_length = normalized_input.shape[1]
    min_required_length = 8  # 模型架构要求的最小长度
    
    if input_length < min_required_length:
        # 如果序列太短，padding到最小长度（仅确保模型可以处理）
        padding_size = min_required_length - input_length
        padding = torch.zeros(1, padding_size, normalized_input.shape[2], device=device)
        normalized_input = torch.cat([normalized_input, padding], dim=1)
    
    # 使用标准编码方式（支持任意长度）
    with torch.no_grad():
        # 使用encode方法获取body和hand的编码
        (body_code, hand_code), _ = motion_vae.encode(normalized_input)
    
    # 获取第一个batch的结果并展平
    body_codes = body_code.squeeze(0)  # shape: (T_body,)
    hand_codes = hand_code.squeeze(0) + code_num  # offset hand codes, shape: (T_hand,)
    
    return body_codes, hand_codes


def process_audio_to_tokens(audio_path, prompt_manager):
    """将音频文件编码为tokens"""
    try:
        kimi_tokens = torch.from_numpy(np.array(prompt_manager._tokenize_audio(audio_path)))
        return kimi_tokens
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None


def create_jsonl_entry(user_audio_tokens, assistant_body_tokens, assistant_hand_tokens, 
                      assistant_audio_tokens, user_audio_path, assistant_audio_path, 
                      base_name, user_prompt, code_num=512):
    """创建jsonl条目
    
    Args:
        user_audio_tokens: 用户音频的token
        assistant_body_tokens: assistant的body motion tokens
        assistant_hand_tokens: assistant的hand motion tokens
        assistant_audio_tokens: assistant的音频token
        user_audio_path: 用户音频文件路径
        assistant_audio_path: assistant音频文件路径
        base_name: 文件基础名称
        user_prompt: 用户提示语
        code_num: codebook大小
    """
    # 转换为numpy并展平
    if isinstance(assistant_body_tokens, torch.Tensor):
        assistant_body_tokens = assistant_body_tokens.cpu().numpy().reshape(-1)
    else:
        assistant_body_tokens = np.array(assistant_body_tokens).reshape(-1)
    
    if isinstance(assistant_hand_tokens, torch.Tensor):
        assistant_hand_tokens = assistant_hand_tokens.cpu().numpy().reshape(-1)
    else:
        assistant_hand_tokens = np.array(assistant_hand_tokens).reshape(-1)
    
    if isinstance(user_audio_tokens, torch.Tensor):
        user_audio_tokens = user_audio_tokens.cpu().numpy().reshape(-1)
    else:
        user_audio_tokens = np.array(user_audio_tokens).reshape(-1)
    
    if isinstance(assistant_audio_tokens, torch.Tensor):
        assistant_audio_tokens = assistant_audio_tokens.cpu().numpy().reshape(-1)
    else:
        assistant_audio_tokens = np.array(assistant_audio_tokens).reshape(-1)
    
    # 合并body和hand tokens到统一的codebook
    # body tokens: [0, code_num-1] = [0, 511]
    # hand tokens: [code_num, 2*code_num-1] = [512, 1023] (已在process_motion_to_tokens中添加偏移)
    # 最终motion tokens: body[0], hand[0], body[1], hand[1], ... (逐对交替)
    motion_tokens = []
    max_len = max(len(assistant_body_tokens), len(assistant_hand_tokens))
    for i in range(max_len):
        if i < len(assistant_body_tokens):
            motion_tokens.append(int(assistant_body_tokens[i]))  # body token: [0, 511]
        if i < len(assistant_hand_tokens):
            motion_tokens.append(int(assistant_hand_tokens[i]))  # hand token: [512, 1023]
    
    # 转换为list
    user_audio_tokens = user_audio_tokens.tolist()
    assistant_audio_tokens = assistant_audio_tokens.tolist()
    
    return {
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
                'content': user_audio_path,
                'audio_tokens': user_audio_tokens
            },
            {
                "role": "assistant",
                "message_type": "audio_motion",
                'audio_tokens': assistant_audio_tokens,
                'content': assistant_audio_path,
                'motion_tokens': motion_tokens
            },
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate motion and audio tokens from HAND_RING data')
    parser.add_argument("--vqvae_config", type=str, default="g1_vqvae_arbitrary_length_balanced.yaml", 
                       help="VQ-VAE config file")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                       help="Path to VQ-VAE checkpoint (default: from config)")
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--data_dir", type=str, 
                       default="/root/workspace/HRI_MLLM/data/HAND_RING_1111_joint_vecs_0_1000",
                       help="Data directory containing npy/, wav/, json/ subdirectories")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for tokens (default: data_dir/tokens)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_mode", action="store_true", help="Process only first 3 files")
    
    # JSONL相关参数
    parser.add_argument("--create_jsonl", action="store_true", default=True,
                       help="Create jsonl file after generating tokens")
    parser.add_argument("--jsonl_output_dir", type=str, default=None,
                       help="Output directory for jsonl files (default: DATA_ROOT)")
    parser.add_argument("--jsonl_output_name", type=str, default="HAND_RING_1111_tokens.jsonl",
                       help="Output jsonl file name")
    parser.add_argument("--user_prompt", type=str, 
                       default="Please response the following spoken content with speech and a corresponding full-body motion sequence.",
                       help="User prompt for jsonl entries")
    
    args = parser.parse_args()
    
    # 加载VQ-VAE模型
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
    print(f"Loading VQ-VAE config from: {config_path}")
    motion_config = open_yaml(config_path)
    
    # 确定checkpoint路径
    if args.vqvae_checkpoint:
        checkpoint_path = args.vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        checkpoint_path = "output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt"
        print(f"⚠️  No checkpoint specified, using default: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")
    
    # 加载归一化统计量
    test_mean, test_std = load_normalization_stats(motion_config)
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to(args.device)
    std_t = torch.tensor(test_std, dtype=torch.float32).to(args.device)
    
    # 加载模型
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device=args.device)
    print(f"✅ VQ-VAE model loaded successfully!")
    
    # 加载音频tokenizer
    print(f"Loading audio tokenizer from: {args.model_name_or_path}")
    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)
    
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    prompt_manager = KimiAPromptManager(
        model_path=cache_path, 
        kimia_token_offset=model_config.kimia_token_offset, 
        kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
    )
    print(f"✅ Audio tokenizer loaded successfully!")
    
    # 设置目录路径
    data_dir = args.data_dir
    npy_dir = os.path.join(data_dir, "npy")
    wav_dir = os.path.join(data_dir, "wav")
    
    if not os.path.exists(npy_dir):
        print(f"❌ NPY directory not found: {npy_dir}")
        exit(1)
    
    if not os.path.exists(wav_dir):
        print(f"❌ WAV directory not found: {wav_dir}")
        exit(1)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(data_dir, "tokens")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Processing: {data_dir}")
    print(f"NPY dir: {npy_dir}")
    print(f"WAV dir: {wav_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*80}")
    
    # 获取所有 *_B.npy 文件
    motion_files = [f for f in os.listdir(npy_dir) if f.endswith('_B.npy')]
    print(f"Found {len(motion_files)} motion files (*_B.npy)")
    
    if args.test_mode:
        motion_files = motion_files[:3]
        print(f"⚠️  TEST MODE: Processing only first 3 files")
    
    # 处理每个文件
    success_count = 0
    jsonl_entries = []  # 存储jsonl条目
    code_num = motion_config.get("code_num", 512)
    
    for motion_file in tqdm(motion_files, desc="Processing files"):
        try:
            # 解析文件名，例如 "1000_B.npy" -> base = "1000"
            base_name = motion_file.replace('_B.npy', '')
            
            # 构建文件路径
            motion_path = os.path.join(npy_dir, motion_file)  # 1000_B.npy
            user_audio_path = os.path.join(wav_dir, base_name + '_A.wav')  # 1000_A.wav
            assistant_audio_path = os.path.join(wav_dir, base_name + '_B.wav')  # 1000_B.wav
            
            # 检查文件是否存在
            if not os.path.exists(user_audio_path):
                print(f"⚠️  User audio not found: {user_audio_path}")
                continue
            
            if not os.path.exists(assistant_audio_path):
                print(f"⚠️  Assistant audio not found: {assistant_audio_path}")
                continue
            
            # 处理用户音频 (A音频)
            user_audio_tokens = process_audio_to_tokens(user_audio_path, prompt_manager)
            if user_audio_tokens is None:
                print(f"❌ Failed to process user audio: {user_audio_path}")
                continue
            
            # 处理助手音频 (B音频)
            assistant_audio_tokens = process_audio_to_tokens(assistant_audio_path, prompt_manager)
            if assistant_audio_tokens is None:
                print(f"❌ Failed to process assistant audio: {assistant_audio_path}")
                continue
            
            # 处理动作数据 (B动作)
            motion_data = np.load(motion_path)
            body_tokens, hand_tokens = process_motion_to_tokens(
                motion_data, motion_vae, mean_t, std_t, 
                args.device, code_num=code_num
            )
            
            # 保存tokens
            body_motion_output_path = os.path.join(output_dir, base_name + '_B_body_tokens.pt')
            hand_motion_output_path = os.path.join(output_dir, base_name + '_B_hand_tokens.pt')
            user_audio_output_path = os.path.join(output_dir, base_name + '_A_audio_tokens.pt')
            assistant_audio_output_path = os.path.join(output_dir, base_name + '_B_audio_tokens.pt')
            
            torch.save(body_tokens, body_motion_output_path)
            torch.save(hand_tokens, hand_motion_output_path)
            torch.save(user_audio_tokens, user_audio_output_path)
            torch.save(assistant_audio_tokens, assistant_audio_output_path)
            
            # 创建jsonl条目
            if args.create_jsonl:
                jsonl_entry = create_jsonl_entry(
                    user_audio_tokens, body_tokens, hand_tokens,
                    assistant_audio_tokens, user_audio_path, assistant_audio_path,
                    base_name, args.user_prompt, code_num=code_num
                )
                jsonl_entries.append(jsonl_entry)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {motion_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✅ Completed processing: {success_count}/{len(motion_files)} successful")
    
    # 保存jsonl文件
    if args.create_jsonl and jsonl_entries:
        jsonl_output_dir = args.jsonl_output_dir if args.jsonl_output_dir else DATA_ROOT
        os.makedirs(jsonl_output_dir, exist_ok=True)
        
        # 确定输出文件名
        jsonl_output_filename = args.jsonl_output_name
        if not jsonl_output_filename.endswith('.jsonl'):
            jsonl_output_filename += '.jsonl'
        
        jsonl_output_file = os.path.join(jsonl_output_dir, jsonl_output_filename)
        
        with open(jsonl_output_file, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✅ JSONL file saved: {jsonl_output_file} ({len(jsonl_entries)} entries)")
    
    print(f"\n{'='*80}")
    print(f"🎉 All processing completed!")
    print(f"✅ Successfully processed {success_count} files")
    if args.create_jsonl:
        print(f"✅ JSONL file has been created")
    print(f"{'='*80}")

