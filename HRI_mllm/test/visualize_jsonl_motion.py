"""
可视化JSONL文件中的motion tokens

功能：
1. 从jsonl文件中读取motion tokens
2. 使用VQ-VAE解码tokens为motion features
3. 使用vis_audio_motion可视化动作
4. 可选：加载原始motion数据并对比（如果找到）

使用方法：
    python HRI_mllm/test/visualize_jsonl_motion.py \
      --jsonl_path data/BEAT_v2_1110_tokens.jsonl \
      --vqvae_config g1_vqvae_arbitrary_length_balanced.yaml \
      --vqvae_checkpoint output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt \
      --num_samples 5 \
      --output_dir jsonl_visualization
"""
import os
import argparse
import json
import pickle
import numpy as np
import torch
import yaml
import random
import subprocess
from pathlib import Path
from tqdm import tqdm

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats, feats2datapkl
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, device='cuda', code_num=512):
    """
    解码motion tokens为motion features
    
    Args:
        motion_tokens: List of motion tokens [body1, hand1, body2, hand2, ...]
        motion_vae: The motion VQ-VAE model
        mean_t: Mean tensor for denormalization
        std_t: Std tensor for denormalization
        device: 计算设备
        code_num: codebook大小
    
    Returns:
        decoded_motion: numpy array, shape (T, 491) - 解码后的motion features（未归一化）
    """
    # 过滤无效的tokens
    # motion_tokens的范围应该是[0, code_num*2-1] = [0, 1023]
    # body tokens: [0, code_num-1] = [0, 511]
    # hand tokens: [code_num, 2*code_num-1] = [512, 1023]
    valid_tokens = []
    for t in motion_tokens:
        if isinstance(t, (int, np.integer, np.int64, np.int32)):
            t_int = int(t)
            if 0 <= t_int < code_num * 2:
                valid_tokens.append(t_int)
    
    if len(valid_tokens) == 0:
        print(f"Error: No valid motion tokens (total: {len(motion_tokens)})")
        return None
    
    # 分离body和hand tokens
    # motion_tokens格式: [body1, hand1, body2, hand2, ...]
    if len(valid_tokens) % 2 != 0:
        print(f"Warning: motion_tokens length ({len(valid_tokens)}) is odd, dropping last token")
        valid_tokens = valid_tokens[:-1]
    
    if len(valid_tokens) == 0:
        print("Error: No motion tokens after processing")
        return None
    
    body_tokens = valid_tokens[0::2]  # even indices: [body1, body2, ...]
    hand_tokens = valid_tokens[1::2]  # odd indices: [hand1, hand2, ...]
    
    # 验证和修正token范围
    # Body tokens应该在[0, code_num-1]范围内
    body_tokens = [max(0, min(code_num - 1, int(t))) for t in body_tokens]
    # Hand tokens应该在[code_num, 2*code_num-1]范围内，需要减去code_num偏移
    hand_tokens_original = [int(t) for t in hand_tokens]
    hand_tokens = [max(code_num, min(2 * code_num - 1, int(t))) - code_num for t in hand_tokens]
    
    # 检查hand tokens是否在有效范围内
    invalid_hand_count = sum(1 for orig, corrected in zip(hand_tokens_original, hand_tokens) 
                             if orig != corrected + code_num)
    if invalid_hand_count > 0:
        print(f"Warning: {invalid_hand_count} hand tokens were out of range and corrected")
    
    # 转换为tensors
    body_tokens_t = torch.tensor(body_tokens, dtype=torch.long).unsqueeze(0).to(device)
    hand_tokens_t = torch.tensor(hand_tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # 解码
    with torch.no_grad():
        decoded = motion_vae.decode((body_tokens_t, hand_tokens_t))
        # decoded: (1, T, 491) - 这是归一化的features
    
    # 反归一化
    decoded_motion = (decoded * std_t + mean_t).cpu().numpy()[0]  # (T, 491)
    
    return decoded_motion


def find_original_motion_file(audio_path, dataset_name=None):
    """查找原始motion文件（.npy）"""
    if audio_path is None or audio_path.startswith('<'):
        return None
    
    # 从audio_path提取文件名
    audio_filename = Path(audio_path).stem
    
    # 尝试多个可能的位置
    possible_paths = []
    
    # 根据数据集名称尝试不同的路径
    if dataset_name:
        if "beat" in dataset_name.lower() or "BEAT" in dataset_name:
            possible_paths.append(os.path.join(DATA_ROOT, "BEAT_v2_kimi", "new_joint_vecs", f"{audio_filename}.npy"))
        elif "single_motion" in dataset_name.lower():
            possible_paths.append(os.path.join(DATA_ROOT, "single_motion_for_tokenizer_1108_joint_vecs", "npy", f"{audio_filename}.npy"))
        elif "seg" in dataset_name.lower():
            possible_paths.append(os.path.join(DATA_ROOT, "seg_finger", "new_joint_vecs", f"{audio_filename}.npy"))
        elif "synthetic" in dataset_name.lower():
            possible_paths.append(os.path.join(DATA_ROOT, "synthetic_data", "SG_2_or_3_long_sentence_1030_en_joint_vecs", "npy", f"{audio_filename}.npy"))
    
    # 通用路径（按优先级排序）
    possible_paths.extend([
        os.path.join(DATA_ROOT, "BEAT_v2_kimi", "new_joint_vecs", f"{audio_filename}.npy"),
        os.path.join(DATA_ROOT, "single_motion_for_tokenizer_1108_processed", "new_joint_vecs", f"{audio_filename}.npy"),
        os.path.join(DATA_ROOT, "single_motion_sentence_version2_kimi", "new_joint_vecs", f"{audio_filename}.npy"),
        os.path.join(DATA_ROOT, "seg_finger", "new_joint_vecs", f"{audio_filename}.npy"),
    ])
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def concat_side_by_side(video_left, video_right, output_path, target_height=720):
    """使用 ffmpeg 横向拼接两个视频"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_left,
        "-i", video_right,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1[left];[1:v]scale=-2:{target_height},setsar=1[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def visualize_motion_from_features(motion_features, output_path, mean, std, device='cuda'):
    """从motion features可视化动作"""
    # motion_features应该是未归一化的原始数据
    # 转换为tensor并归一化（feats2datapkl需要归一化的输入）
    motion_tensor = torch.from_numpy(motion_features).unsqueeze(0).to(device).float()
    mean_t = torch.tensor(mean, dtype=torch.float32).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).to(device)
    
    # 归一化
    normalized_motion = (motion_tensor - mean_t) / std_t
    
    # 转换为pkl（不使用IK优化）
    motion_pkl = feats2datapkl(normalized_motion, mean=mean, std=std)
    
    # 保存pkl
    pkl_path = output_path.replace('.csv', '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(motion_pkl, f)
    
    # 转换为csv
    motion_csv = load_motion_pkl_as_csv_data(pkl_path)
    csv_path = output_path
    np.savetxt(csv_path, motion_csv, delimiter=',', fmt='%.8f')
    
    # 生成视频
    video_path = output_path.replace('.csv', '.mp4')
    vis_audio_motion(
        csv_path,
        output_path=video_path,
        audio_path=None,
        robot_type="g1_brainco",
        rate_limit=False,
        motion_fps=25
    )
    
    return video_path


def main():
    parser = argparse.ArgumentParser(description='Visualize motion tokens from jsonl file')
    parser.add_argument('--jsonl_path', type=str, required=True,
                       help='Path to the jsonl file containing tokens')
    parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='VQ-VAE config file name')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                       help='VQ-VAE checkpoint path. If not provided, will use the one in config file')
    parser.add_argument('--output_dir', type=str, default='./jsonl_visualization',
                       help='Output directory for visualization files')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (randomly selected)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--compare_with_gt', action='store_true',
                       help='Compare with ground truth motion if available')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name for finding original motion files (e.g., BEAT_v2_kimi)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载VQ-VAE模型
    print(f"Loading VQ-VAE model...")
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        exit(1)
    
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
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
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
    
    code_num = motion_config.get("code_num", 512)
    
    # 读取jsonl文件
    print(f"\nReading jsonl file: {args.jsonl_path}")
    if not os.path.exists(args.jsonl_path):
        print(f"❌ JSONL file not found: {args.jsonl_path}")
        exit(1)
    
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} samples in jsonl file")
    
    # 选择样本
    if args.num_samples > len(lines):
        print(f"⚠️  Requested {args.num_samples} samples, but only {len(lines)} available. Processing all samples.")
        sample_indices = list(range(len(lines)))
    else:
        sample_indices = random.sample(range(len(lines)), args.num_samples)
    
    print(f"Selected samples: {sample_indices}")
    
    # 处理每个样本
    success_count = 0
    for idx, line_idx in enumerate(tqdm(sample_indices, desc="Processing samples")):
        try:
            data = json.loads(lines[line_idx])
            
            # 提取motion tokens和audio路径
            audio_path = None
            motion_tokens = None
            audio_tokens = None
            
            for msg in data.get('conversation', []):
                if msg.get('message_type') == 'audio' and msg.get('audio_tokens'):
                    audio_path = msg.get('content')
                    audio_tokens = msg.get('audio_tokens')
                if msg.get('message_type') == 'audio_motion' and msg.get('motion_tokens'):
                    motion_tokens = msg.get('motion_tokens')
            
            if motion_tokens is None:
                print(f"⚠️  Sample {idx} (line {line_idx}): No motion tokens found")
                continue
            
            # 提取样本名字（从audio_path提取文件名，如果没有则使用索引）
            sample_name = None
            if audio_path:
                sample_name = Path(audio_path).stem  # 获取不带扩展名的文件名
            else:
                sample_name = f"sample_{idx}"
            
            print(f"\n{'='*80}")
            print(f"Sample {idx} (line {line_idx})")
            print(f"Sample name: {sample_name}")
            print(f"Motion tokens: {len(motion_tokens)} tokens")
            if audio_path:
                print(f"Audio path: {audio_path}")
            print(f"{'='*80}")
            
            # 解码motion tokens
            decoded_motion = decode_motion_tokens(
                motion_tokens, motion_vae, mean_t, std_t, 
                device=args.device, code_num=code_num
            )
            
            if decoded_motion is None:
                print(f"⚠️  Sample {idx}: Failed to decode motion tokens")
                continue
            
            print(f"Decoded motion shape: {decoded_motion.shape}")
            
            # 可视化解码后的motion
            sample_output_dir = os.path.join(args.output_dir, f"sample_{idx}")
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # 生成解码后的motion视频
            decoded_csv_path = os.path.join(sample_output_dir, "decoded_motion.csv")
            decoded_video_path = visualize_motion_from_features(
                decoded_motion, decoded_csv_path, test_mean, test_std, args.device
            )
            print(f"✅ Decoded motion video: {decoded_video_path}")
            
            # 如果启用GT对比，尝试加载原始motion
            if args.compare_with_gt:
                original_motion_path = find_original_motion_file(audio_path, args.dataset_name)
                
                if original_motion_path and os.path.exists(original_motion_path):
                    print(f"Found original motion: {original_motion_path}")
                    original_motion = np.load(original_motion_path)
                    
                    # 可视化原始motion（GT）
                    gt_csv_path = os.path.join(sample_output_dir, "ground_truth_motion.csv")
                    gt_video_path = visualize_motion_from_features(
                        original_motion, gt_csv_path, test_mean, test_std, args.device
                    )
                    print(f"✅ Ground truth motion video: {gt_video_path}")
                    
                    # 创建GT vs decoded的对比视频
                    comparison_video_path = os.path.join(sample_output_dir, "comparison_gt_vs_decoded.mp4")
                    concat_side_by_side(gt_video_path, decoded_video_path, comparison_video_path)
                    print(f"✅ Comparison video: {comparison_video_path}")
                else:
                    print(f"⚠️  Original motion file not found for sample {idx}")
            
            # 保存样本信息到单独的JSON文件
            sample_info = {
                "sample_index": idx,
                "line_index": line_idx,
                "sample_name": sample_name,
                "audio_path": audio_path
            }
            sample_info_json_path = os.path.join(sample_output_dir, "sample_info.json")
            with open(sample_info_json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_info, f, ensure_ascii=False, indent=2)
            print(f"✅ Sample info saved to: {sample_info_json_path}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing sample {idx} (line {line_idx}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Visualization completed!")
    print(f"Successfully processed: {success_count}/{len(sample_indices)} samples")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

