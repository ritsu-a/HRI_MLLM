import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import yaml
from HRI_mllm import ROOT
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats

# Audio tokenizer imports
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)


def process_motion_to_tokens(motion_data, motion_vae, mean_t, std_t, device, target_length, window_size=256, stride=None, code_num=512):
    """
    将动作数据编码为tokens，body和hand分开编码
    
    Args:
        motion_data: numpy array, shape (T, 491)
        motion_vae: VQVaeBodyHand模型
        mean_t: 归一化均值
        std_t: 归一化标准差
        device: 计算设备
        target_length: 目标token长度（与audio对齐，未使用）
        window_size: 窗口大小（用于padding短序列）
        stride: 未使用，保留以兼容性
        code_num: codebook大小
        
    Returns:
        body_codes: torch.Tensor, body部分编码
        hand_codes: torch.Tensor, hand部分编码
    """
    # 转换为tensor并归一化
    data_tensor = torch.from_numpy(motion_data).unsqueeze(0).to(device).float()
    normalized_input = (data_tensor - mean_t) / std_t
    
    # 如果序列太短，padding到window_size
    input_length = normalized_input.shape[1]
    if input_length < window_size:
        padding_size = window_size - input_length
        padding = torch.zeros(1, padding_size, normalized_input.shape[2], device=device)
        normalized_input = torch.cat([normalized_input, padding], dim=1)
    
    # 使用标准编码方式
    with torch.no_grad():
        # 使用encode方法获取body和hand的编码
        (body_code, hand_code), _ = motion_vae.encode(normalized_input)
        
        # body_code: (1, T_body), hand_code: (1, T_hand)
        # 这里T_body和T_hand因为下采样而小于等于原始长度（down_t=2, stride_t=2，约除以4）
        # 每个token对应原始序列中的多个帧
        
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


def find_audio_file(base_name, audio_dirs):
    """在多个音频目录中查找音频文件"""
    for audio_dir in audio_dirs:
        if not audio_dir:
            continue
        # 首先尝试直接匹配
        for ext in ['.wav', '.mp3']:
            potential_audio = os.path.join(audio_dir, base_name + ext)
            if os.path.exists(potential_audio):
                return potential_audio
        
        # 递归搜索子目录
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.startswith(base_name) and file.endswith(('.wav', '.mp3')):
                    return os.path.join(root, file)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate motion and audio tokens from data')
    parser.add_argument("--vqvae_config", type=str, default="g1_vqvae_arbitrary_length_balanced.yaml", 
                       help="VQ-VAE config file")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                       help="Path to VQ-VAE checkpoint (default: from config)")
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--data_dirs", type=str, nargs='+', 
                       default=["single_motion_sentence_version2_kimi"],
                       help="List of data directories to process.")
    parser.add_argument("--motion_subdir", type=str, default="new_joint_vecs")
    parser.add_argument("--audio_subdir", type=str, default=None)
    parser.add_argument("--output_subdir", type=str, default="tokens")
    parser.add_argument("--window_size", type=int, default=256, help="Window size for padding short sequences (BEAT: 256, SeG: 64)")
    parser.add_argument("--stride", type=int, default=None, help="Not used, kept for compatibility")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_mode", action="store_true")
    
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
        checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
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
    
    # 处理每个数据集目录
    from HRI_mllm import DATA_ROOT
    
    for data_dir_name in args.data_dirs:
        data_dir = os.path.join(DATA_ROOT, data_dir_name)
        if not os.path.exists(data_dir):
            print(f"⚠️  Data directory not found: {data_dir}")
            continue
        
        motion_dir = os.path.join(data_dir, args.motion_subdir)
        if not os.path.exists(motion_dir):
            print(f"⚠️  Motion directory not found: {motion_dir}")
            continue
        
        # 构建音频目录列表
        audio_dirs = []
        
        # 1. 尝试数据集内的音频目录
        for possible_dir in ["audio", "wav", "wavs"]:
            potential_dir = os.path.join(data_dir, possible_dir)
            if os.path.exists(potential_dir):
                audio_dirs.append(potential_dir)
                break
        
        # 2. 特殊处理：映射关系
        if "internet_data_v1_kimi" in data_dir_name:
            # internet_data_v1_kimi -> internet_data_1021
            internet_audio_dir = os.path.join(DATA_ROOT, "internet_data_1021")
            if os.path.exists(internet_audio_dir):
                audio_dirs.append(internet_audio_dir)
        
        if "1025_60_60fps_kimi" in data_dir_name:
            # 1025_60_60fps_kimi -> 1025_60_60fps
            fps_audio_dir = os.path.join(DATA_ROOT, "1025_60_60fps")
            if os.path.exists(fps_audio_dir):
                audio_dirs.append(fps_audio_dir)
        
        # 3. 支持SG_2_or_3_long_sentence_1030_en_kimi自动音频处理
        if "SG_2_or_3_long_sentence_1030_en_kimi" in data_dir_name:
            sg_audio_dir = os.path.join(DATA_ROOT, "SG_2_or_3_long_sentence_1030_en")
            if os.path.exists(sg_audio_dir):
                audio_dirs.append(sg_audio_dir)

        if "single_motion_sentence_version2_kimi" in data_dir_name:
            single_motion_sentence_audio_dir = os.path.join(DATA_ROOT, "single_motion_sentence_version2")
            if os.path.exists(single_motion_sentence_audio_dir):
                audio_dirs.append(single_motion_sentence_audio_dir)
           
        
        # 3. 尝试BEAT原始位置
        if "BEAT" in data_dir_name:

            beat_audio_dir = os.path.join(DATA_ROOT, "BEAT_v2")
            if os.path.exists(beat_audio_dir):
                audio_dirs.append(beat_audio_dir)
        
        if not audio_dirs:
            print(f"⚠️  No audio directory found")
        
        output_dir = os.path.join(data_dir, args.output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Processing: {data_dir_name}")
        print(f"Motion dir: {motion_dir}")
        print(f"Audio dirs: {audio_dirs}")
        print(f"Output dir: {output_dir}")
        print(f"{'='*80}")
        
        # 获取所有.npy文件
        motion_files = [f for f in os.listdir(motion_dir) if f.endswith('.npy')]
        print(f"Found {len(motion_files)} motion files")
        
        if args.test_mode:
            motion_files = motion_files[:3]
            print(f"⚠️  TEST MODE: Processing only first 3 files")
        
        # 处理每个文件
        success_count = 0
        for motion_file in tqdm(motion_files, desc=f"Processing {data_dir_name}"):
            try:
                motion_path = os.path.join(motion_dir, motion_file)
                base_name = motion_file.replace('.npy', '')
                
                # 查找音频文件
                audio_file = find_audio_file(base_name, audio_dirs)
                
                # 如果没找到，尝试文件名匹配
                if not audio_file and '_' in base_name:
                    parts = base_name.split('_')
                    if len(parts) >= 2:
                        # BEAT格式：尝试 {number}/filename.wav
                        for audio_dir in audio_dirs:
                            potential = os.path.join(audio_dir, parts[0], base_name + '.wav')
                            if os.path.exists(potential):
                                audio_file = potential
                                break
                
                if audio_file and os.path.exists(audio_file):
                    audio_tokens = process_audio_to_tokens(audio_file, prompt_manager)
                    if audio_tokens is None:
                        continue
                    
                    target_length = len(audio_tokens)
                    
                    # window_size兼容，仅seg为64
                    if "seg" in data_dir_name.lower():
                        window_size = 64
                    else:
                        window_size = 256
                    
                    # 编码动作数据
                    motion_data = np.load(motion_path)
                    body_tokens, hand_tokens = process_motion_to_tokens(
                        motion_data, motion_vae, mean_t, std_t, 
                        args.device, target_length,
                        window_size=window_size,
                        stride=args.stride,
                        code_num=motion_config.get("code_num", 512)
                    )
                    
                    
                    
                    # 保存
                    body_motion_output_path = os.path.join(output_dir, base_name + '_body_tokens.pt')
                    hand_motion_output_path = os.path.join(output_dir, base_name + '_hand_tokens.pt')
                    audio_output_path = os.path.join(output_dir, base_name + '_audio_tokens.pt')
                    
                    torch.save(body_tokens, body_motion_output_path)
                    torch.save(hand_tokens, hand_motion_output_path)
                    torch.save(audio_tokens, audio_output_path)
                    
                    success_count += 1
                else:
                    print(f"Audio file not found for: {motion_file}")
                    
            except Exception as e:
                print(f"\n❌ Error processing {motion_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ Completed processing {data_dir_name}: {success_count}/{len(motion_files)} successful")
    
    print(f"\n{'='*80}")
    print(f"🎉 All processing completed!")
    print(f"{'='*80}") 