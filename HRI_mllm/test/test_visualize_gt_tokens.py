### Visualize ground truth tokens from BEAT_v2_kimi_tokens.jsonl
### Directly decode motion tokens without using the adaptor

import os
import argparse
import json
import pickle
import numpy as np
import torch
import shutil
import yaml
import random

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats, feats2datapkl
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
import wave
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion
from HRI_mllm import ROOT, DATA_ROOT
import subprocess


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """
    Decode motion tokens to motion features
    
    Args:
        motion_tokens: List of motion tokens [body1, hand1, body2, hand2, ...]
        motion_vae: The motion VQ-VAE model
        mean_t: Mean tensor for denormalization
        std_t: Std tensor for denormalization
        expected_frames: Expected number of motion frames (optional, used for truncation)
    
    Returns:
        data_dict: Decoded motion data in pkl format
    """
    # Separate body and hand tokens
    # motion_tokens are alternating: [body, hand, body, hand, ...]
    if len(motion_tokens) % 2 != 0:
        print(f"Warning: motion_tokens length ({len(motion_tokens)}) is odd, dropping last token")
        motion_tokens = motion_tokens[:-1]
    
    body_tokens = motion_tokens[0::2]  # even indices
    hand_tokens = motion_tokens[1::2]  # odd indices
    
    # Convert to tensors
    body_tokens = torch.tensor(body_tokens).unsqueeze(0).to("cuda")
    # hand tokens need to subtract 512 offset
    hand_tokens = torch.tensor(hand_tokens).unsqueeze(0).to("cuda") - 512
    
    print(f"Body tokens shape: {body_tokens.shape}, range: [{body_tokens.min()}, {body_tokens.max()}]")
    print(f"Hand tokens shape: {hand_tokens.shape}, range: [{hand_tokens.min()}, {hand_tokens.max()}]")
    
    # Decode using VAE - this returns normalized features
    decoded = motion_vae.decode((body_tokens, hand_tokens))
    
    print(f"Decoded shape before truncation: {decoded.shape}")
    
    # If expected_frames is provided and decoded is longer, truncate to expected length
    if expected_frames is not None and decoded.shape[1] > expected_frames:
        decoded = decoded[:, :expected_frames, :]
        print(f"Truncated to expected length: {decoded.shape}")
    
    # Convert to data format (feats2datapkl will handle denormalization with provided stats)
    data_dict = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())
    
    return data_dict


# Parse arguments
parser = argparse.ArgumentParser(description='Visualize ground truth tokens from jsonl')
parser.add_argument('--jsonl_path', type=str, default='/root/workspace/HRI_MLLM/data/SG_2_or_3_long_sentence_1030_en_kimi_tokens.jsonl',
                   help='Path to the jsonl file containing tokens')
parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                   help='VQ-VAE config file name')
parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='VQ-VAE checkpoint path. If not provided, will use the one in config file')
parser.add_argument('--output_dir', type=str, default='./gt_visualization',
                   help='Output directory for visualization files')
parser.add_argument('--num_samples', type=int, default=3, help='Number of random samples to visualize')

args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load the first line from jsonl
print(f"Reading tokens from: {args.jsonl_path}")
with open(args.jsonl_path, 'r') as f:
    lines = f.readlines()

if args.num_samples > len(lines):
    print(f"可用样本少于num_samples，全部处理（{len(lines)}个）")
    sample_indices = list(range(len(lines)))
else:
    sample_indices = random.sample(range(len(lines)), args.num_samples)

print(f"随机选择的样本编号: {sample_indices}")

for idx, line_idx in enumerate(sample_indices):
    data = json.loads(lines[line_idx])
    # --- 与原单例流程逻辑相同，全部在output_dir编上idx ---
    audio_path = None
    motion_tokens = None
    audio_tokens = None
    for msg in data['conversation']:
        if msg.get('message_type') == 'audio' and msg.get('audio_tokens'):
            audio_path = msg['content']
            audio_tokens = msg['audio_tokens']
        if msg.get('message_type') == 'audio_motion' and msg.get('motion_tokens'):
            motion_tokens = msg['motion_tokens']
    if motion_tokens is None:
        print(f"Error: No motion tokens in entry #{line_idx}")
        continue
    if audio_path is None:
        print(f"Error: No audio path in entry #{line_idx}")
        continue
    print(f"==== 样本#{idx} 文件[{audio_path}] ====")
    # 载入VQ-VAE同一份参数
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
    motion_config = open_yaml(config_path)
    if args.vqvae_checkpoint:
        checkpoint_path = args.vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
    test_mean, test_std = load_normalization_stats(motion_config)
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
    std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device="cuda")
    # 解码重建
    motion_pkl = decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None)
    output_pkl_path = os.path.join(args.output_dir, f"motion_{idx}.pkl")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(motion_pkl, f)
    motion_csv = load_motion_pkl_as_csv_data(output_pkl_path)
    output_csv_path = os.path.join(args.output_dir, f"motion_{idx}.csv")
    np.savetxt(output_csv_path, motion_csv, delimiter=',', fmt='%.8f')
    output_video_path = os.path.join(args.output_dir, f"motion_{idx}.mp4")
    vis_audio_motion(output_csv_path, output_path=output_video_path, audio_path=None, robot_type="g1_brainco", rate_limit=False)
    print(f"  重建mp4: {output_video_path}")
    # ground truth motion查找和保存
    from pathlib import Path
    audio_filename = Path(audio_path).stem
    npy_path = f"/root/workspace/HRI_MLLM/data/SG_2_or_3_long_sentence_1030_en_kimi/new_joint_vecs/{audio_filename}.npy"
    found_original = False
    if os.path.exists(npy_path):
        orig_data = np.load(npy_path)
        data_tensor = torch.from_numpy(orig_data).unsqueeze(0).to("cuda").float()
        normalized_input = (data_tensor - mean_t) / std_t
        gt_pkl = feats2datapkl(normalized_input, mean=test_mean, std=test_std)
        gt_pkl_path = os.path.join(args.output_dir, f"ground_truth_{idx}.pkl")
        with open(gt_pkl_path, 'wb') as f:
            pickle.dump(gt_pkl, f)
        ground_truth_csv_path = os.path.join(args.output_dir, f"ground_truth_{idx}.csv")
        gt_csv = load_motion_pkl_as_csv_data(gt_pkl_path)
        np.savetxt(ground_truth_csv_path, gt_csv, delimiter=',', fmt='%.8f')
        gt_video_path = os.path.join(args.output_dir, f"ground_truth_{idx}.mp4")
        vis_audio_motion(ground_truth_csv_path, output_path=gt_video_path, audio_path=None, robot_type="g1_brainco", rate_limit=False)
        print(f"  gt mp4: {gt_video_path}")
        # 横向拼接对比
        def concat_side_by_side(video_left: str, video_right: str, output_path: str, target_height: int = 720):
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", video_left,
                "-i", video_right,
                "-filter_complex",
                f"[0:v]scale=-2:{target_height},setsar=1[left];[1:v]scale=-2:{target_height},setsar=1[right];[left][right]hstack=inputs=2[v]",
                "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-c:a", "aac", "-shortest", output_path
            ]
            subprocess.run(cmd, check=True)
        comparison_video_path = os.path.join(args.output_dir, f"gt_vs_recon_{idx}.mp4")
        concat_side_by_side(gt_video_path, output_video_path, comparison_video_path, target_height=720)
        print(f"  gt_vs_recon mp4: {comparison_video_path}")
    else:
        print(f"  ⚠️ 未找到ground truth npy: {npy_path}")
        print(f"  只输出重建mp4: {output_video_path}")

