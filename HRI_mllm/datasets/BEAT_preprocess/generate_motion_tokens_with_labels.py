import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import yaml
import json
import pandas as pd
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats

# Audio tokenizer imports
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager

# Label映射关系
LABEL_MAP = {
    "BACKGROUND": 0, 
    "FIST_BEAT": 1,
    "FOREFINGER_RAISE_ONE": 2,
    "FOREFINGER_RAISE_ONE-2": 2, 
    "HAND_CALL": 3,
    "HAND_RING": 4,
    "HAND_V_SIGN": 5,
    "PALM_HALT": 6,
    "THUMB_FOREFINGER_AND_LITTLE_FINGER_RAISE": 7,
    "THUMB_UP": 8,
}

# Motion token的采样率：1秒对应12.5个token
TOKENS_PER_SECOND = 12.5


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)


def process_motion_to_tokens(motion_data, motion_vae, mean_t, std_t, device, target_length, window_size=None, stride=None, code_num=512):
    """
    将动作数据编码为tokens，body和hand分开编码（整段处理，不使用窗口）
    
    Args:
        motion_data: numpy array, shape (T, 491)
        motion_vae: VQVaeBodyHand模型
        mean_t: 归一化均值
        std_t: 归一化标准差
        device: 计算设备
        target_length: 目标token长度（与audio对齐，未使用）
        window_size: 窗口大小（已废弃，不再使用）
        stride: 未使用，保留以兼容性
        code_num: codebook大小
        
    Returns:
        body_codes: torch.Tensor, body部分编码
        hand_codes: torch.Tensor, hand部分编码
    """
    # 转换为tensor并归一化
    data_tensor = torch.from_numpy(motion_data).unsqueeze(0).to(device).float()
    normalized_input = (data_tensor - mean_t) / std_t
    
    # 🔧 整段处理：不使用window_size，直接处理完整序列
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
        # VQ-VAE的encoder可以处理任意长度的序列
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


def find_audio_file_for_jsonl(base_name, data_dir, dataset_name):
    """查找音频文件（用于jsonl创建）"""
    # 尝试多个可能的音频目录
    audio_dirs = []
    
    # 首先尝试数据集内的音频目录
    for possible_dir in ["audio", "wav", "wavs"]:
        potential_dir = os.path.join(data_dir, possible_dir)
        if os.path.exists(potential_dir):
            audio_dirs.append(potential_dir)
            break
    
    if "beat" in dataset_name.lower():
        # BEAT相关数据集
        audio_dirs.append(os.path.join(DATA_ROOT, "BEAT_v2"))
        audio_dirs.append(os.path.join(DATA_ROOT, "beat_english_v0.2.1"))
    elif "single_motion_for_tokenizer" in dataset_name:
        # single_motion_for_tokenizer数据集，音频在数据目录的wav文件夹下
        audio_dirs.append(os.path.join(data_dir, "wav"))
    else:
        # internet_data或其他
        audio_dirs.append(os.path.join(DATA_ROOT, "internet_data_1021"))
        audio_dirs.append(os.path.join(DATA_ROOT, "internet_data_v1_kimi"))
        audio_dirs.append(os.path.join(DATA_ROOT, "single_motion_sentence_version2"))
    
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


def find_json_file(base_name, data_dir, dataset_name):
    """查找JSON文件"""
    json_path = None
    
    # 首先尝试在数据目录的json子目录中查找
    json_dir = os.path.join(data_dir, "json")
    if os.path.exists(json_dir):
        potential_json = os.path.join(json_dir, base_name + '.json')
        if os.path.exists(potential_json):
            return potential_json
    
    # 尝试在数据目录中直接查找
    potential_json_paths = [
        os.path.join(data_dir, base_name + '.json'),
        os.path.join(DATA_ROOT, 'single_motion_for_tokenizer_1108', base_name + '.json'),
    ]
    
    # 如果base_name包含目录信息，尝试在对应目录下查找
    if '_' in base_name:
        # 尝试从base_name中提取目录名（例如：FIST_BEAT_blend_npz_0.4_60_60_linear_with_annotation_1）
        parts = base_name.split('_')
        # 查找可能的目录前缀
        for i in range(len(parts)):
            potential_dir_name = '_'.join(parts[:i+1])
            potential_json_dir = os.path.join(data_dir, potential_dir_name, "blend_npz_0.4_60_60_linear_with_annotation")
            if os.path.exists(potential_json_dir):
                potential_json = os.path.join(potential_json_dir, parts[-1] + '.json')
                if os.path.exists(potential_json):
                    return potential_json
    
    for path in potential_json_paths:
        if os.path.exists(path):
            json_path = path
            break
    
    return json_path


def find_csv_file(base_name, data_dir, dataset_name, csv_dir=None):
    """查找CSV文件（如果存在）"""
    if csv_dir:
        csv_path = os.path.join(csv_dir, base_name + '.csv')
        if os.path.exists(csv_path):
            return csv_path
    
    # 尝试在数据目录中查找
    potential_csv_paths = [
        os.path.join(data_dir, "csv", base_name + '.csv'),
        os.path.join(data_dir, base_name + '.csv'),
    ]
    
    for path in potential_csv_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_motion_labels_from_json(json_path, motion_tokens_count, total_duration):
    """
    从JSON文件加载motion标签信息
    
    Args:
        json_path: JSON文件路径
        motion_tokens_count: motion_tokens的总数量
        total_duration: 总时长（秒），如果为None，从JSON中读取
        
    Returns:
        labels: List[Dict]，每个Dict包含motion名称和时间范围信息
        total_duration: 总时长（秒）
    """
    if json_path is None or not os.path.exists(json_path):
        return [], None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从JSON中获取total_duration
        if total_duration is None:
            total_duration = data.get('total_duration', None)
            if total_duration is None:
                # 尝试从actual_npz_duration获取
                total_duration = data.get('actual_npz_duration', None)
        
        if total_duration is None:
            print(f"Warning: Cannot determine total_duration from JSON file {json_path}")
            return [], None
        
        # 提取blended_timeline中的标签信息
        labels = []
        blended_timeline = data.get('blended_timeline', [])
        
        for motion_item in blended_timeline:
            motion_name = motion_item.get('motion', '')
            actual_start_time = motion_item.get('actual_start_time', None)
            actual_end_time = motion_item.get('actual_end_time', None)
            
            if actual_start_time is not None and actual_end_time is not None:
                labels.append({
                    'motion': motion_name,
                    'start_time': actual_start_time,
                    'end_time': actual_end_time,
                })
        
        return labels, total_duration
        
    except Exception as e:
        print(f"Warning: Failed to load JSON file {json_path}: {e}")
        return [], None


def load_motion_labels_from_csv(csv_path, motion_tokens_count, total_duration):
    """
    从CSV文件加载motion标签信息
    
    Args:
        csv_path: CSV文件路径
        motion_tokens_count: motion_tokens的总数量
        total_duration: 总时长（秒），如果为None，需要从CSV或其他地方获取
        
    Returns:
        labels: List[Dict]，每个Dict包含motion名称和时间范围信息
        total_duration: 总时长（秒）
    """
    if csv_path is None or not os.path.exists(csv_path):
        return [], total_duration
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 检查是否有motion_start和motion_end列
        if 'motion_start' not in df.columns or 'motion_end' not in df.columns:
            print(f"Warning: CSV file {csv_path} does not have motion_start and motion_end columns")
            return [], total_duration
        
        # 检查是否有motion或gesture列
        motion_col = None
        for col in ['motion', 'gesture', 'label']:
            if col in df.columns:
                motion_col = col
                break
        
        if motion_col is None:
            print(f"Warning: CSV file {csv_path} does not have motion/gesture/label column")
            return [], total_duration
        
        labels = []
        for _, row in df.iterrows():
            motion_name = str(row[motion_col])
            motion_start = float(row['motion_start'])
            motion_end = float(row['motion_end'])
            
            if motion_name and not pd.isna(motion_start) and not pd.isna(motion_end):
                labels.append({
                    'motion': motion_name,
                    'start_time': motion_start,
                    'end_time': motion_end,
                })
        
        # 如果没有提供total_duration，尝试从CSV中推断（取最大的motion_end）
        if total_duration is None and len(labels) > 0:
            max_end_time = max(label['end_time'] for label in labels)
            total_duration = max_end_time
        
        return labels, total_duration
        
    except Exception as e:
        print(f"Warning: Failed to load CSV file {csv_path}: {e}")
        return [], total_duration


def map_motion_name_to_label(motion_name):
    """
    将motion名称映射到label值
    
    Args:
        motion_name: motion名称，例如 "FIST_BEAT-4" 或 "FIST_BEAT"
        
    Returns:
        label: 对应的label值（0-8）
    """
    if not motion_name:
        return 0
    
    # 移除可能的后缀（例如 "FIST_BEAT-4" -> "FIST_BEAT"）
    motion_base = motion_name.split('-')[0].strip()
    
    # 直接在LABEL_MAP中查找
    if motion_base in LABEL_MAP:
        return LABEL_MAP[motion_base]
    
    # 如果没找到，尝试匹配部分名称
    for key, value in LABEL_MAP.items():
        if key in motion_base or motion_base in key:
            return value
    
    # 默认返回0（BACKGROUND）
    return 0


def generate_label_tokens(motion_tokens, labels, total_duration):
    """
    根据motion tokens和标签信息生成label tokens
    
    Args:
        motion_tokens: List[int]，motion tokens列表
        labels: List[Dict]，每个Dict包含motion名称和时间范围信息
        total_duration: 总时长（秒）
        
    Returns:
        label_tokens: List[int]，label tokens列表，长度与motion_tokens相同
    """
    motion_tokens_count = len(motion_tokens)
    
    # 初始化label_tokens，默认值为0
    label_tokens = [0] * motion_tokens_count
    
    if not labels or total_duration is None:
        return label_tokens
    
    # 计算每秒对应的token数量
    # 理论上应该是 TOKENS_PER_SECOND (12.5)，但实际可能略有差异
    # 使用实际值：motion_tokens_count / total_duration
    tokens_per_second = motion_tokens_count / total_duration
    
    # 处理每个标签
    for label_info in labels:
        motion_name = label_info.get('motion', '')
        start_time = label_info.get('start_time', 0)
        end_time = label_info.get('end_time', 0)
        
        if start_time >= end_time:
            continue
        
        # 将时间转换为token索引
        start_token_idx = int(start_time * tokens_per_second)
        end_token_idx = int(end_time * tokens_per_second)
        
        # 确保索引在有效范围内
        start_token_idx = max(0, min(start_token_idx, motion_tokens_count - 1))
        end_token_idx = max(0, min(end_token_idx, motion_tokens_count))
        
        # 获取对应的label值
        label_value = map_motion_name_to_label(motion_name)
        
        # 在对应的时间范围内设置label
        for idx in range(start_token_idx, end_token_idx):
            if idx < motion_tokens_count:
                # 如果当前位置已经有非0的label，保留较大的值（或者可以根据需要选择其他策略）
                if label_tokens[idx] == 0 or label_value > label_tokens[idx]:
                    label_tokens[idx] = label_value
    
    return label_tokens


def create_jsonl_entry(body_tokens, hand_tokens, audio_tokens, label_tokens, base_name, data_dir, dataset_name, user_prompt, audio_file_path=None, code_num=512):
    """创建jsonl条目"""
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
    # hand tokens: [code_num, 2*code_num-1] = [512, 1023] (已在process_motion_to_tokens中添加偏移)
    # 最终motion tokens: body[0], hand[0], body[1], hand[1], ... (逐对交替)
    motion_tokens = []
    max_len = max(len(body_tokens), len(hand_tokens))
    for i in range(max_len):
        if i < len(body_tokens):
            motion_tokens.append(int(body_tokens[i]))  # body token: [0, 511]
        if i < len(hand_tokens):
            motion_tokens.append(int(hand_tokens[i]))  # hand token: [512, 1023]
    
    # 确保label_tokens长度与motion_tokens一致
    if len(label_tokens) != len(motion_tokens):
        # 如果长度不一致，调整label_tokens
        if len(label_tokens) < len(motion_tokens):
            # 如果label_tokens较短，用0填充
            label_tokens.extend([0] * (len(motion_tokens) - len(label_tokens)))
        else:
            # 如果label_tokens较长，截断
            label_tokens = label_tokens[:len(motion_tokens)]
    
    # 查找音频文件路径
    if audio_file_path and os.path.exists(audio_file_path):
        # 使用实际找到的音频文件路径
        wav_path = audio_file_path
    else:
        # 尝试查找音频文件
        wav_path = find_audio_file_for_jsonl(base_name, data_dir, dataset_name)
        if wav_path is None:
            # 使用placeholder
            wav_path = f"<audio_path_for_{base_name}>"
    
    # 转换为list
    audio_tokens = audio_tokens.tolist()
    
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
                'content': wav_path,
                'audio_tokens': audio_tokens
            },
            {
                "role": "assistant",
                "message_type": "audio_motion",
                'audio_tokens': audio_tokens,
                'content': wav_path,
                'motion_tokens': motion_tokens,
                'label_tokens': label_tokens
            },
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate motion and audio tokens from data with label tokens')
    parser.add_argument("--vqvae_config", type=str, default="g1_vqvae_arbitrary_length_balanced.yaml", 
                       help="VQ-VAE config file")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                       help="Path to VQ-VAE checkpoint (default: from config)")
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B")
    parser.add_argument("--data_dirs", type=str, nargs='+', 
                       default=["single_motion_for_tokenizer_1108_joint_vecs"],
                       help="List of data directories to process.")
    parser.add_argument("--motion_subdir", type=str, default="npy")
    parser.add_argument("--audio_subdir", type=str, default=None)
    parser.add_argument("--output_subdir", type=str, default="tokens")
    parser.add_argument("--window_size", type=int, default=None, help="Window size (deprecated, not used - processing full sequences)")
    parser.add_argument("--stride", type=int, default=None, help="Not used, kept for compatibility")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_mode", action="store_true")
    
    # JSONL相关参数
    parser.add_argument("--create_jsonl", action="store_true", default=False,
                       help="Create jsonl file after generating tokens")
    parser.add_argument("--jsonl_output_dir", type=str, default=None,
                       help="Output directory for jsonl files (default: DATA_ROOT)")
    parser.add_argument("--jsonl_output_name", type=str, default=None,
                       help="Output jsonl file name")
    parser.add_argument("--user_prompt", type=str, 
                       default="Please repeat the following spoken content with a corresponding full-body motion sequence.",
                       help="User prompt for jsonl entries")
    
    # Label相关参数
    parser.add_argument("--csv_dir", type=str, default=None,
                       help="Directory containing CSV files with motion_start and motion_end (optional)")
    parser.add_argument("--use_json_labels", action="store_true", default=True,
                       help="Use JSON files for labels (default: True)")
    parser.add_argument("--use_csv_labels", action="store_true", default=False,
                       help="Use CSV files for labels (default: False)")
    
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
        if "single_motion_for_tokenizer" in data_dir_name:
            # single_motion_for_tokenizer数据集，音频在数据目录的wav文件夹下
            audio_dirs.append(os.path.join(data_dir, "wav"))
        
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
        jsonl_entries = []  # 存储jsonl条目
        
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
                    
                    # 🔧 整段处理：不使用window_size，直接处理完整序列
                    # 编码动作数据（整段处理）
                    motion_data = np.load(motion_path)
                    body_tokens, hand_tokens = process_motion_to_tokens(
                        motion_data, motion_vae, mean_t, std_t, 
                        args.device, target_length,
                        window_size=None,  # 不使用window_size
                        stride=args.stride,
                        code_num=motion_config.get("code_num", 512)
                    )
                    
                    # 合并body和hand tokens以计算motion_tokens长度
                    body_tokens_array = body_tokens.cpu().numpy().reshape(-1) if isinstance(body_tokens, torch.Tensor) else np.array(body_tokens).reshape(-1)
                    hand_tokens_array = hand_tokens.cpu().numpy().reshape(-1) if isinstance(hand_tokens, torch.Tensor) else np.array(hand_tokens).reshape(-1)
                    max_len = max(len(body_tokens_array), len(hand_tokens_array))
                    # motion_tokens是body和hand交替排列，所以总长度是len(body) + len(hand)
                    motion_tokens_count = len(body_tokens_array) + len(hand_tokens_array)
                    
                    # 加载标签信息
                    labels = []
                    total_duration = None
                    
                    # 优先使用JSON文件
                    if args.use_json_labels:
                        json_path = find_json_file(base_name, data_dir, data_dir_name)
                        if json_path:
                            labels, total_duration = load_motion_labels_from_json(json_path, motion_tokens_count, total_duration)
                    
                    # 如果JSON中没有找到，尝试CSV文件
                    if (not labels or total_duration is None) and args.use_csv_labels:
                        csv_path = find_csv_file(base_name, data_dir, data_dir_name, args.csv_dir)
                        if csv_path:
                            csv_labels, csv_duration = load_motion_labels_from_csv(csv_path, motion_tokens_count, total_duration)
                            if csv_labels:
                                labels = csv_labels
                            if csv_duration:
                                total_duration = csv_duration
                    
                    # 如果仍然没有total_duration，尝试从音频文件获取
                    if total_duration is None:
                        try:
                            import librosa
                            audio_data, sr = librosa.load(audio_file, sr=None)
                            total_duration = len(audio_data) / sr
                        except:
                            # 如果无法获取音频时长，使用motion token数量估算
                            total_duration = motion_tokens_count / TOKENS_PER_SECOND
                            print(f"Warning: Cannot determine duration for {base_name}, using estimated duration: {total_duration:.2f}s")
                    
                    # 生成label tokens
                    # 首先需要构建完整的motion_tokens列表来计算长度
                    motion_tokens_list = []
                    for i in range(max_len):
                        if i < len(body_tokens_array):
                            motion_tokens_list.append(int(body_tokens_array[i]))
                        if i < len(hand_tokens_array):
                            motion_tokens_list.append(int(hand_tokens_array[i]))
                    
                    label_tokens = generate_label_tokens(motion_tokens_list, labels, total_duration)
                    
                    # 保存tokens
                    body_motion_output_path = os.path.join(output_dir, base_name + '_body_tokens.pt')
                    hand_motion_output_path = os.path.join(output_dir, base_name + '_hand_tokens.pt')
                    audio_output_path = os.path.join(output_dir, base_name + '_audio_tokens.pt')
                    
                    torch.save(body_tokens, body_motion_output_path)
                    torch.save(hand_tokens, hand_motion_output_path)
                    torch.save(audio_tokens, audio_output_path)
                    
                    # 🔧 创建jsonl条目（如果启用了jsonl输出）
                    if args.create_jsonl:
                        jsonl_entry = create_jsonl_entry(
                            body_tokens, hand_tokens, audio_tokens, label_tokens,
                            base_name, data_dir, data_dir_name,
                            args.user_prompt,
                            audio_file_path=audio_file,  # 传递实际找到的音频文件路径
                            code_num=motion_config.get("code_num", 512)
                        )
                        jsonl_entries.append(jsonl_entry)
                    
                    success_count += 1
                else:
                    print(f"Audio file not found for: {motion_file}")
                    
            except Exception as e:
                print(f"\n❌ Error processing {motion_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ Completed processing {data_dir_name}: {success_count}/{len(motion_files)} successful")
        
        # 🔧 保存jsonl文件（如果启用了jsonl输出）
        if args.create_jsonl and jsonl_entries:
            jsonl_output_dir = args.jsonl_output_dir if args.jsonl_output_dir else DATA_ROOT
            os.makedirs(jsonl_output_dir, exist_ok=True)
            
            # 确定输出文件名
            if args.jsonl_output_name:
                # 使用指定的输出文件名
                jsonl_output_filename = args.jsonl_output_name
                if not jsonl_output_filename.endswith('.jsonl'):
                    jsonl_output_filename += '.jsonl'
            else:
                # 默认使用数据集名称
                jsonl_output_filename = f"{data_dir_name}_tokens_with_labels.jsonl"
            
            jsonl_output_file = os.path.join(jsonl_output_dir, jsonl_output_filename)
            
            with open(jsonl_output_file, 'w', encoding='utf-8') as f:
                for entry in jsonl_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"✅ JSONL file saved: {jsonl_output_file} ({len(jsonl_entries)} entries)")
    
    print(f"\n{'='*80}")
    print(f"🎉 All processing completed!")
    if args.create_jsonl:
        print(f"✅ JSONL files with labels have been created")
    print(f"{'='*80}")

