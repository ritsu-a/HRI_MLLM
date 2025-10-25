"""
VQ-VAE测试脚本 - 滑动窗口测试（带重叠平滑）

功能：
1. 滑动窗口测试 - 使用50%重叠的32帧窗口
2. 加权平滑混合 - 使用汉宁窗对重叠区域进行加权平均
3. 详细性能统计 - 每个窗口的loss
4. 可视化生成 - 对比原始和重建动作

使用方法：
    python HRI_mllm/test/test_vqvae.py \
      --config g1_vqvae_final.yaml \
      --checkpoint output/vqvae_final/checkpoints/vqvae_final.pt

说明：
- ✅ 32帧窗口，与训练保持一致
- ✅ 50%重叠（stride=16），可调整为75%获得更平滑效果
- ✅ 汉宁窗加权混合，窗口边界平滑过渡
- ✅ 生成更连贯的动作序列
"""
import os
import argparse

from HRI_retarget.utils.motion_lib.qpose_denoiser import low_pass_filter

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

from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
# 使用支持自定义统计量的版本
from HRI_mllm.utils.motion_utils.g1ml3d_final import (
    feats2datapkl, feats2joints, load_normalization_stats
)
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion
import joblib



# 加载配置文件
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)



def collate_fn(batch):
    motions = torch.stack([torch.from_numpy(item[1]) for item in batch])
    if motions.isnan().any():
        print("Found NaN in motion data")
        import ipdb;ipdb.set_trace()
    return motions


# 在验证集上测试 MPJPE 和 PA-MPJPE
def validate(model, val_loader, device):
    model.eval()
    mpjpe_list, pa_mpjpe_list = [], []
    
    with torch.no_grad():
        for motions in val_loader:
            motions = motions.to(device)
            code = model.encode(motions)

            decoded = model.decode(code[0]).reshape(motions.shape)  # 解码生成运动
            
            # 计算关节位置
            
            joints_gt = feats2joints(motions)
            joints_pred = feats2joints(decoded)
            # 计算 MPJPE 和 PA-MPJPE
            for i in range(motions.shape[0]):
                mpjpe = calc_mpjpe(joints_gt[i], joints_pred[i]).mean()
                pa_mpjpe = calc_pampjpe(joints_gt[i], joints_pred[i]).mean()
                mpjpe_list.append(mpjpe.item())
                pa_mpjpe_list.append(pa_mpjpe.item())
    
    # 返回平均指标
    return np.mean(mpjpe_list), np.mean(pa_mpjpe_list)




# 主函数
if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='Test VQ-VAE model - Final Version')
    parser.add_argument('--config', type=str, 
                       default='g1_vqvae_final.yaml',
                       help='Config file to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file. If not provided, will use the one in config file')
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
        # 如果配置文件中没有指定checkpoint，使用默认路径
        checkpoint_path = "output/vqvae_final/checkpoints/vqvae_final.pt"
        print(f"⚠️  No checkpoint specified in config, using default: {checkpoint_path}")
    
    # 检查checkpoint是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\n默认checkpoint路径: output/vqvae_final/checkpoints/vqvae_final.pt")
        print(f"请使用 --checkpoint 参数指定正确的路径，或先运行训练")
        exit(1)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 🔧 关键修复：加载与训练时相同的归一化统计量
    print(f"\n" + "="*80)
    print(f"🔧 加载归一化统计量（与训练时保持一致）")
    print(f"="*80)
    
    # 使用新的函数加载统计量
    test_mean, test_std = load_normalization_stats(motion_config)
    
    print(f"统计量shape: Mean {test_mean.shape}, Std {test_std.shape}")
    print(f"Mean范围: [{test_mean.min():.4f}, {test_mean.max():.4f}]")
    print(f"Std范围: [{test_std.min():.4f}, {test_std.max():.4f}]")
    print(f"="*80 + "\n")
    
    # 加载模型
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device=args.device)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Config: {args.config}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Device: {args.device}")
    print(f"   Use Mixed Stats: {motion_config.get('use_mixed_stats', True)}")

    ### encode&decode for specific motion

    beat_vec_path = f"/root/workspace/HRI_MLLM/data/BEAT_v2_kimi/new_joint_vecs/1_wayne_0_1_1.npy"
    train_data_vec = np.load(beat_vec_path)

    beat_vec_path = f"/root/workspace/HRI_MLLM/data/BEAT_v2_kimi/new_joint_vecs/1_wayne_0_1_1.npy"
    beat_filename = beat_vec_path.split("/")[-1]
    audio_path = os.path.join("/root/workspace/HRI_MLLM/data/BEAT_v2", beat_filename.split("_")[0], beat_filename.replace(".npy", ".wav"))

    # 🔧 关键修复：使用与训练时相同的统计量进行归一化
    print(f"\n归一化测试数据...")
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to(args.device)
    std_t = torch.tensor(test_std, dtype=torch.float32).to(args.device)
    
    print(f"原始数据shape: {train_data_vec.shape}")
    print(f"原始数据范围: [{train_data_vec.min():.4f}, {train_data_vec.max():.4f}]")
    
    # 🔧 使用滑动窗口测试（带重叠，更连贯）
    window_size = 32  # 与训练时保持一致
    stride = 16  # 50%重叠，可调整为8（75%重叠）获得更平滑效果
    total_frames = train_data_vec.shape[0]
    
    print(f"\n{'='*80}")
    print(f"🔧 滑动窗口测试（带重叠，更连贯）")
    print(f"{'='*80}")
    print(f"总帧数: {total_frames} ({total_frames/20:.1f}秒 @20fps)")
    print(f"窗口大小: {window_size} 帧 ({window_size/20:.1f}秒)")
    print(f"滑动步长: {stride} 帧 (重叠率: {(1-stride/window_size)*100:.0f}%)")
    print(f"窗口数: {(total_frames - window_size) // stride + 1}")
    
    # 测试多个窗口并收集结果
    all_decoded_windows = []
    window_positions = []  # 记录每个窗口的起始位置
    window_losses = []
    
    print(f"\n执行滑动窗口编解码...")
    # 滑动窗口测试（stride<window_size，有重叠）
    for start_idx in range(0, total_frames - window_size + 1, stride):
        window_data = train_data_vec[start_idx:start_idx + window_size]
        
        data_tensor = torch.from_numpy(window_data).unsqueeze(0).to(args.device).float()
        normalized_input = (data_tensor - mean_t) / std_t
        normalized_input = normalized_input.float()
        
        # 编码-解码
        with torch.no_grad():
            encode_result = motion_vae.encode(normalized_input)
            motion_tokens = encode_result[0] if isinstance(encode_result, tuple) else encode_result
            decoded_features = motion_vae.decode(motion_tokens).detach().cpu()
        
        # 计算窗口loss
        window_loss = F.mse_loss(decoded_features, normalized_input.cpu()).item()
        window_losses.append(window_loss)
        
        all_decoded_windows.append(decoded_features.squeeze(0))  # [32, F]
        window_positions.append(start_idx)
    
    # 🌟 使用加权平滑混合重叠区域
    print(f"\n融合重叠区域...")
    output_length = window_positions[-1] + window_size  # 最后一个窗口的结束位置
    feature_dim = all_decoded_windows[0].shape[-1]
    
    # 初始化累加器
    decoded_accumulated = torch.zeros(output_length, feature_dim)
    weight_accumulated = torch.zeros(output_length, 1)
    source_accumulated = torch.zeros(output_length, feature_dim)
    source_weight = torch.zeros(output_length, 1)
    
    # 创建窗口权重（使用汉宁窗使边缘权重较小，中心权重较大）
    window_weight = torch.hann_window(window_size).unsqueeze(-1)  # [32, 1]
    
    # 累加所有窗口（加权）
    for i, start_idx in enumerate(window_positions):
        end_idx = start_idx + window_size
        decoded_accumulated[start_idx:end_idx] += all_decoded_windows[i] * window_weight
        weight_accumulated[start_idx:end_idx] += window_weight
        
        # 同样处理source数据
        source_window = train_data_vec[start_idx:start_idx + window_size]
        source_tensor = torch.from_numpy(source_window).float()
        normalized_source = (source_tensor - torch.tensor(test_mean).float()) / torch.tensor(test_std).float()
        source_accumulated[start_idx:end_idx] += normalized_source * window_weight
        source_weight[start_idx:end_idx] += window_weight
    
    # 归一化（除以累积权重）
    decoded_features = (decoded_accumulated / weight_accumulated.clamp(min=1e-8)).unsqueeze(0)  # [1, T, F]
    normalized_input_full = (source_accumulated / source_weight.clamp(min=1e-8)).unsqueeze(0)  # [1, T, F]
    
    # 统计结果
    avg_window_loss = np.mean(window_losses)
    min_window_loss = np.min(window_losses)
    max_window_loss = np.max(window_losses)
    
    print(f"\n📊 窗口测试结果:")
    print(f"  测试了 {len(window_losses)} 个窗口")
    print(f"  平均Loss: {avg_window_loss:.6f}")
    print(f"  最小Loss: {min_window_loss:.6f}")
    print(f"  最大Loss: {max_window_loss:.6f}")
    print(f"  Loss标准差: {np.std(window_losses):.6f}")
    
    # 整体重建loss（拼接后的）
    recon_loss = F.mse_loss(decoded_features, normalized_input_full)
    print(f"\n📊 整体重建Loss (拼接后): {recon_loss:.6f}")
    print(f"   ✅ 接近窗口平均loss: {avg_window_loss:.6f}")
    
    # 评估
    if avg_window_loss < 0.01:
        status = "✅ 优秀！"
    elif avg_window_loss < 0.02:
        status = "✅ 良好"
    elif avg_window_loss < 0.05:
        status = "⚠️  一般"
    else:
        status = "❌ 需要优化"
    
    print(f"  状态: {status}")
    print(f"{'='*80}\n")

    # 🔧 关键修复：反归一化时也使用相同的统计量
    print(f"反归一化...")
    decoded_data_pkl = feats2datapkl(decoded_features, mean=test_mean, std=test_std)
    source_data_pkl = feats2datapkl(normalized_input_full, mean=test_mean, std=test_std)

    with open("source.pkl", 'wb') as f:
        pickle.dump(source_data_pkl, f)
    with open("decoded.pkl", 'wb') as f:
        pickle.dump(decoded_data_pkl, f)
    
    source_csv =  load_motion_pkl_as_csv_data("source.pkl")
    decoded_csv =  load_motion_pkl_as_csv_data("decoded.pkl")

    np.savetxt("source.csv", source_csv, delimiter=',', fmt='%.8f')
    np.savetxt("decoded.csv", decoded_csv, delimiter=',', fmt='%.8f')

    print(f"\n生成可视化...")
    vis_audio_motion(audio_path, "source.csv", output_path="vqvae_output_source.mp4", robot_type="g1_brainco", rate_limit=False)
    vis_audio_motion(audio_path, "decoded.csv", output_path="vqvae_output_decoded.mp4", robot_type="g1_brainco", rate_limit=False)
    
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
    print("\n✅ 已生成对比视频: vqvae_output_compare.mp4")
    print(f"\n{'='*80}")
    print(f"🎉 测试完成！（使用修复后的归一化统计量）")
    print(f"{'='*80}")

