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
parser.add_argument('--jsonl_path', type=str, default='/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl',
                   help='Path to the jsonl file containing tokens')
parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                   help='VQ-VAE config file name')
parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='VQ-VAE checkpoint path. If not provided, will use the one in config file')
parser.add_argument('--output_dir', type=str, default='./gt_visualization',
                   help='Output directory for visualization files')

args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load the first line from jsonl
print(f"Reading tokens from: {args.jsonl_path}")
with open(args.jsonl_path, 'r') as f:
    first_line = f.readline()
    first_line = f.readline()
    data = json.loads(first_line)

# Extract audio path, audio tokens, and motion tokens
audio_path = None
motion_tokens = None
audio_tokens = None

for msg in data['conversation']:
    if msg.get('message_type') == 'audio' and msg.get('audio_tokens'):
        audio_path = msg['content']
        audio_tokens = msg['audio_tokens']
        print(f"Found audio: {audio_path}")
        print(f"Audio tokens length: {len(audio_tokens)}")
    
    if msg.get('message_type') == 'audio_motion' and msg.get('motion_tokens'):
        motion_tokens = msg['motion_tokens']
        print(f"Motion tokens length: {len(motion_tokens)}")

if motion_tokens is None:
    print("Error: No motion tokens found in the first entry")
    exit(1)

if audio_path is None:
    print("Error: No audio path found in the first entry")
    exit(1)

# Load VQ-VAE configuration
config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
print(f"Loading VQ-VAE config from: {config_path}")
motion_config = open_yaml(config_path)

# Determine checkpoint path
if args.vqvae_checkpoint:
    checkpoint_path = args.vqvae_checkpoint
elif "ckpt" in motion_config and motion_config["ckpt"]:
    checkpoint_path = motion_config["ckpt"]
else:
    checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
    print(f"⚠️  No checkpoint specified in config, using default: {checkpoint_path}")

# Check if checkpoint exists
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")

# Load normalization statistics
test_mean, test_std = load_normalization_stats(motion_config)
mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")

# Load the VAE model
motion_vae = VQVaeBodyHand(**motion_config)
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")
print(f"✅ VQ-VAE model loaded successfully!")

# Calculate expected frames based on audio duration
expected_frames = None

# Decode motion tokens
print("\nDecoding motion tokens...")
motion_pkl = decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=expected_frames)

# Save motion pkl
output_pkl_path = os.path.join(args.output_dir, "motion.pkl")
with open(output_pkl_path, 'wb') as f:
    pickle.dump(motion_pkl, f)
print(f"Saved motion pkl to: {output_pkl_path}")

# Convert to CSV
motion_csv = load_motion_pkl_as_csv_data(output_pkl_path)
output_csv_path = os.path.join(args.output_dir, "motion.csv")
np.savetxt(output_csv_path, motion_csv, delimiter=',', fmt='%.8f')
print(f"Saved motion CSV to: {output_csv_path}")



# Load original motion data for ground truth comparison
ground_truth_csv_path = None

# Try to find original motion data
from pathlib import Path

# Extract base name from audio path
audio_filename = Path(audio_path).stem  # e.g., "10_kieks_0_103_103"

# Try to load the original motion data from potential locations
potential_paths = [
    os.path.join(DATA_ROOT, "BEAT_v2_kimi", "new_joint_vecs", f"{audio_filename}.npy"),
    os.path.join(DATA_ROOT, "BEAT_v2_kimi", f"{audio_filename}.npy"),
    os.path.join(DATA_ROOT, "SeG_kimi", "new_joint_vecs", f"{audio_filename}.npy"),
]

found_original = False
for npy_path in potential_paths:
    if os.path.exists(npy_path):
        print(f"\nFound original motion data: {npy_path}")
        # Load and encode-decode original data as ground truth
        orig_data = np.load(npy_path)
        print(f"Original motion shape: {orig_data.shape}")
        
        # Normalize
        data_tensor = torch.from_numpy(orig_data).unsqueeze(0).to("cuda").float()
        normalized_input = (data_tensor - mean_t) / std_t
        

        
        # Convert to pkl
        gt_pkl = feats2datapkl(normalized_input, mean=test_mean, std=test_std)
        with open(os.path.join(args.output_dir, "ground_truth.pkl"), 'wb') as f:
            pickle.dump(gt_pkl, f)
        
        # Convert to CSV
        ground_truth_csv_path = os.path.join(args.output_dir, "ground_truth.csv")
        gt_csv = load_motion_pkl_as_csv_data(os.path.join(args.output_dir, "ground_truth.pkl"))
        np.savetxt(ground_truth_csv_path, gt_csv, delimiter=',', fmt='%.8f')
        print(f"Saved ground truth CSV to: {ground_truth_csv_path}")
        found_original = True
        break

if not found_original:
    print(f"\n⚠️  Original motion data not found for {audio_filename}")
    print("This is expected if the data was pre-encoded. The decoded result shows ground truth tokens.")

# Visualize
output_video_path = os.path.join(args.output_dir, "motion.mp4")
vis_audio_motion(
    output_csv_path,
    output_path=output_video_path,
    audio_path=None,
    robot_type="g1_brainco",
    rate_limit=False
)
print(f"✅ Visualization saved to: {output_video_path}")

# If ground truth exists, create comparison video
if ground_truth_csv_path and os.path.exists(ground_truth_csv_path):
    print("\nCreating comparison video...")
    
    # Visualize ground truth
    gt_video_path = os.path.join(args.output_dir, "ground_truth.mp4")
    vis_audio_motion(
        ground_truth_csv_path,
        output_path=gt_video_path,
        audio_path=None,
        robot_type="g1_brainco",
        rate_limit=False
    )
    
    # Create side-by-side comparison
    def concat_side_by_side(video_left: str, video_right: str, output_path: str, target_height: int = 720):
        """使用 ffmpeg 横向拼接两个视频，统一高度为 target_height，保持宽高比。"""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_left,
            "-i", video_right,
            "-filter_complex",
            f"[0:v]scale=-2:{target_height},setsar=1[left];[1:v]scale=-2:{target_height},setsar=1[right];[left][right]hstack=inputs=2[v]",
            "-map", "[v]",
            "-map", "0:a?",  # 使用左视频音频（若存在）
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, check=True)
    
    comparison_video_path = os.path.join(args.output_dir, "comparison.mp4")
    concat_side_by_side(gt_video_path, output_video_path, comparison_video_path, target_height=720)
    print(f"✅ Comparison video saved to: {comparison_video_path}")

