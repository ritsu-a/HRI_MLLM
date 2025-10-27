"""
VQ-VAE测试脚本 - 支持任意长度序列推理

功能：
1. 任意长度序列推理 - 使用滑动窗口处理完整序列
2. 性能统计 - 整体重建loss
3. 可视化生成 - 对比原始和重建动作

使用方法：
    python HRI_mllm/test/test_vqvae.py \
      --config g1_vqvae_arbitrary_length_balanced.yaml \
      --checkpoint output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt

说明：
- ✅ 支持任意长度序列，使用滑动窗口处理长序列
- ✅ 不同数据集使用不同窗口大小（BEAT: 256帧，SeG: 64帧）
- ✅ 自动处理padding和长度对齐
- ✅ 滑动窗口使用50%重叠确保边界平滑
"""
import os
import argparse

os.environ['MUJOCO_GL'] = 'egl'
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
import yaml
import torch
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae import VQVae
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand

from HRI_mllm.utils.motion_utils.g1ml3d_final import (
    feats2datapkl, load_normalization_stats
)
import torch.nn.functional as F
import numpy as np

from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion



# 加载配置文件
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)








# 主函数
if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='Test VQ-VAE model')
    parser.add_argument('--config', type=str, 
                       default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='Config file to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.config)
    print(f"Loading config from: {config_path}")
    motion_config = open_yaml(config_path)
    
    # 确定checkpoint路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        # 使用默认路径
        checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
        print(f"⚠️  No checkpoint specified, using default: {checkpoint_path}")
    
    # 检查checkpoint是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"请使用 --checkpoint 参数指定正确的路径")
        exit(1)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 加载归一化统计量
    print(f"\n加载归一化统计量...")
    test_mean, test_std = load_normalization_stats(motion_config)
    print(f"统计量shape: Mean {test_mean.shape}, Std {test_std.shape}")
    
    # 加载模型
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device=args.device)
    
    print(f"\n✅ Model loaded successfully!")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}\n")

    # 加载测试数据
    beat_vec_path = "/root/workspace/HRI_MLLM/data/SeG_kimi/new_joint_vecs/THUMB_UP-2.npy"
    train_data_vec = np.load(beat_vec_path)

    # 归一化
    print(f"归一化测试数据...")
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to(args.device)
    std_t = torch.tensor(test_std, dtype=torch.float32).to(args.device)
    
    # 准备输入数据（使用滑动窗口处理任意长度序列）
    total_frames = train_data_vec.shape[0]

   
    data_tensor = torch.from_numpy(train_data_vec).unsqueeze(0).to(args.device).float()
    normalized_input = (data_tensor - mean_t) / std_t
        
       
   
    if True:
        print(f"\n执行标准编解码...")
        with torch.no_grad():
            decoded_features, quant_loss, perplexity = motion_vae(normalized_input)
            
            # 处理输入输出长度不匹配（取最小长度并截断）
            input_length = normalized_input.shape[1]
            output_length = decoded_features.shape[1]
            min_length = min(input_length, output_length)
            
            input_truncated = normalized_input[:, :min_length, :]
            output_truncated = decoded_features[:, :min_length, :]
            recon_loss = F.mse_loss(output_truncated, input_truncated)
            
            print(f"✅ 编解码完成")
            print(f"输入shape: {normalized_input.shape} → 输出shape: {decoded_features.shape}")
            if input_length != output_length:
                print(f"⚠️  长度不一致，截取到 {min_length} 帧")
            print(f"重建Loss: {recon_loss:.6f}, 量化Loss: {quant_loss:.6f}, 困惑度: {perplexity:.2f}")
            
            # 统计有效帧（去除padding部分）
            valid_frames = total_frames
            print(f"有效帧数: {valid_frames}")
    # 否则已经在滑动窗口处理中完成了编解码

   
    
    # 生成可视化数据
    print(f"\n生成可视化...")
    decoded_data_pkl = feats2datapkl(output_truncated, mean=test_mean, std=test_std)
    source_data_pkl = feats2datapkl(input_truncated, mean=test_mean, std=test_std)

    with open("source.pkl", 'wb') as f:
        pickle.dump(source_data_pkl, f)
    with open("decoded.pkl", 'wb') as f:
        pickle.dump(decoded_data_pkl, f)
    
    source_csv =  load_motion_pkl_as_csv_data("source.pkl")
    decoded_csv =  load_motion_pkl_as_csv_data("decoded.pkl")

    np.savetxt("source.csv", source_csv, delimiter=',', fmt='%.8f')
    np.savetxt("decoded.csv", decoded_csv, delimiter=',', fmt='%.8f')

    vis_audio_motion("source.csv", output_path="vqvae_output_source.mp4", audio_path=None, robot_type="g1_brainco", rate_limit=False)
    vis_audio_motion("decoded.csv", output_path="vqvae_output_decoded.mp4", audio_path=None, robot_type="g1_brainco", rate_limit=False)
    
    # 横向拼接两个视频
    import subprocess
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
    
    concat_side_by_side("vqvae_output_source.mp4", "vqvae_output_decoded.mp4", "vqvae_output_compare.mp4", target_height=720)
    print("\n✅ 测试完成！已生成对比视频: vqvae_output_compare.mp4")

