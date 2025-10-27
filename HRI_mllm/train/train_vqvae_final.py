"""
Amplitude-Preserved VQ-VAE训练脚本
解决训练后期动作幅度变小的over-smoothing问题

主要改进:
1. Amplitude Preservation Loss - 保持动作幅度
2. Velocity Loss - 保持动作动态性
3. 学习率衰减 - 防止后期过度优化
4. 幅度实时监控
"""
import os
import sys
import yaml
import torch
import wandb
import pickle
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae import VQVae
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.datasets.MixedMotionDatasetVQ import MixedMotionDatasetVQ
from HRI_mllm.datasets.MixedMotionDatasetVQGrouped import MixedMotionDatasetVQGrouped, GroupedBatchSampler
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def collate_fn(batch):
    """
    原始collate函数：固定窗口大小或变长窗口（支持padding）
    """
    motions_list = []
    dataset_indices = []
    
    for item in batch:
        motion = torch.from_numpy(item[1]).float()  # [T, F]
        dataset_idx = item[6]
        motions_list.append(motion)
        dataset_indices.append(dataset_idx)
    
    # Padding到batch内最大长度
    max_len_in_batch = max(m.shape[0] for m in motions_list)
    padded_motions = []
    
    for motion in motions_list:
        if motion.shape[0] < max_len_in_batch:
            # 用0 padding
            padding = torch.zeros(max_len_in_batch - motion.shape[0], motion.shape[1])
            padded_motion = torch.cat([motion, padding], dim=0)
        else:
            padded_motion = motion
        
        padded_motions.append(padded_motion)
    
    motions = torch.stack(padded_motions)
    dataset_indices = torch.tensor(dataset_indices, dtype=torch.long)
    
    if motions.isnan().any():
        print("Found NaN in motion data")
        import ipdb;ipdb.set_trace()
    return motions, dataset_indices

def collate_fn_variable_length(batch, min_length=32, max_length=128):
    """
    🔧 支持变长窗口的collate函数
    从每个样本中随机截取不同长度的窗口，让模型学习处理多种序列长度
    
    Args:
        batch: 数据批次
        min_length: 最小窗口长度
        max_length: 最大窗口长度
    """
    motions_list = []
    dataset_indices = []
    
    for item in batch:
        motion = torch.from_numpy(item[1]).float()  # [T, F]
        dataset_idx = item[6]
        
        # 随机选择窗口长度（必须是8的倍数，因为下采样2次，每次/2）
        available_lengths = [l for l in range(min_length, max_length + 1, 8) 
                           if l <= motion.shape[0]]
        
        if len(available_lengths) == 0:
            # 如果motion太短，使用原长度
            window_length = motion.shape[0]
        else:
            # 随机选择一个长度
            window_length = np.random.choice(available_lengths)
        
        # 随机截取窗口
        if motion.shape[0] > window_length:
            start_idx = np.random.randint(0, motion.shape[0] - window_length + 1)
            motion_window = motion[start_idx:start_idx + window_length]
        else:
            motion_window = motion
        
        motions_list.append(motion_window)
        dataset_indices.append(dataset_idx)
    
    # Padding到batch内最大长度
    max_len_in_batch = max(m.shape[0] for m in motions_list)
    padded_motions = []
    masks = []
    
    for motion in motions_list:
        if motion.shape[0] < max_len_in_batch:
            # 用0 padding
            padding = torch.zeros(max_len_in_batch - motion.shape[0], motion.shape[1])
            padded_motion = torch.cat([motion, padding], dim=0)
            mask = torch.cat([torch.ones(motion.shape[0]), torch.zeros(max_len_in_batch - motion.shape[0])])
        else:
            padded_motion = motion
            mask = torch.ones(motion.shape[0])
        
        padded_motions.append(padded_motion)
        masks.append(mask)
    
    motions = torch.stack(padded_motions)
    masks = torch.stack(masks)
    dataset_indices = torch.tensor(dataset_indices, dtype=torch.long)
    
    if motions.isnan().any():
        print("Found NaN in motion data")
        import ipdb;ipdb.set_trace()
    
    return motions, dataset_indices, masks


def collate_fn_arbitrary_length(batch, max_length=512, min_length=8):
    """
    🆕 支持任意长度动作序列的collate函数
    保持原始序列长度，只进行必要的padding
    
    Args:
        batch: 数据批次
        max_length: 最大允许长度（防止内存溢出）
        min_length: 最小允许长度（确保卷积层正常工作）
    """
    motions_list = []
    dataset_indices = []
    original_lengths = []
    
    for item in batch:
        motion = torch.from_numpy(item[1]).float()  # [T, F]
        dataset_idx = item[6]
        original_length = motion.shape[0]
        
        # 如果序列太短，padding到最小长度
        if motion.shape[0] < min_length:
            padding = torch.zeros(min_length - motion.shape[0], motion.shape[1])
            motion = torch.cat([motion, padding], dim=0)
            original_length = min_length
        
        # 如果序列太长，截断到最大长度
        if motion.shape[0] > max_length:
            motion = motion[:max_length]
            original_length = max_length
        
        motions_list.append(motion)
        dataset_indices.append(dataset_idx)
        original_lengths.append(original_length)
    
    # Padding到batch内最大长度
    max_len_in_batch = max(m.shape[0] for m in motions_list)
    padded_motions = []
    masks = []
    
    for i, motion in enumerate(motions_list):
        if motion.shape[0] < max_len_in_batch:
            # 用0 padding
            padding = torch.zeros(max_len_in_batch - motion.shape[0], motion.shape[1])
            padded_motion = torch.cat([motion, padding], dim=0)
            mask = torch.cat([torch.ones(motion.shape[0]), torch.zeros(max_len_in_batch - motion.shape[0])])
        else:
            padded_motion = motion
            mask = torch.ones(motion.shape[0])
        
        padded_motions.append(padded_motion)
        masks.append(mask)
    
    motions = torch.stack(padded_motions)
    masks = torch.stack(masks)
    dataset_indices = torch.tensor(dataset_indices, dtype=torch.long)
    original_lengths = torch.tensor(original_lengths, dtype=torch.long)
    
    if motions.isnan().any():
        print("Found NaN in motion data")
        import ipdb;ipdb.set_trace()
    
    return motions, dataset_indices, masks, original_lengths
    
def load_dataset(use_seg_only=False, seg_weight=10.0, beat_weight=1.0, internet_weight=1.0, 
                 use_mixed_stats=True, use_variable_length=False, min_length=32, max_length=128,
                 use_arbitrary_length=False, max_arbitrary_length=512, min_arbitrary_length=8,
                 use_grouped_batches=False):
    """
    加载数据集
    Args:
        use_mixed_stats: 是否使用混合数据集的统计信息（推荐True）
        use_variable_length: 是否使用变长窗口训练（推荐True，提高泛化能力）
        min_length: 变长窗口最小长度
        max_length: 变长窗口最大长度
        use_arbitrary_length: 是否使用任意长度序列（保持原始长度）
        max_arbitrary_length: 任意长度模式下的最大允许长度
        min_arbitrary_length: 任意长度模式下的最小允许长度（确保卷积层正常工作）
    """
    if use_seg_only:
        data_root_list = [os.path.join(DATA_ROOT, "SeG_kimi")]
        dataset_weights = [1.0]
    else:
        data_root_list = [
            os.path.join(DATA_ROOT, "BEAT_v2_kimi"),
            os.path.join(DATA_ROOT, "internet_data"),
            os.path.join(DATA_ROOT, "SeG_kimi")
        ]
        dataset_weights = [beat_weight, internet_weight, seg_weight]
    
    # 选择统计信息来源
    if use_mixed_stats:
        # 🔧 修复：优先使用Fixed版本（修复了std接近0的问题）
        mixed_stats_paths = [
            os.path.join(DATA_ROOT, "Mixed_Statistics_Equal_Fixed"),  # 优先
            os.path.join(DATA_ROOT, "Mixed_Statistics_Equal"),
            os.path.join(DATA_ROOT, "Mixed_Statistics"),
        ]
        mean_std_root = None
        for path in mixed_stats_paths:
            if os.path.exists(os.path.join(path, "Mean.npy")):
                mean_std_root = path
                print(f"✅ Using mixed dataset statistics from: {mean_std_root}")
                break
        
        if mean_std_root is None:
            print(f"⚠️  Mixed statistics not found, falling back to BEAT_v2_kimi")
            print(f"   Run 'bash scripts/compute_mixed_stats.sh' to compute mixed statistics")
            mean_std_root = os.path.join(DATA_ROOT, "BEAT_v2_kimi")
    else:
        # 使用BEAT统计信息（原方法）
        mean_std_root = os.path.join(DATA_ROOT, "BEAT_v2_kimi")
        print(f"ℹ️  Using BEAT-only statistics from: {mean_std_root}")
    
    mean = np.load(os.path.join(mean_std_root, "Mean.npy"))
    std = np.load(os.path.join(mean_std_root, "Std.npy"))
    
    # 🔧 根据训练模式调整数据集参数
    if use_arbitrary_length:
        # 任意长度模式：不限制窗口大小，保持原始长度
        win_size = 1  # 最小窗口，实际会被collate函数处理
        max_motion_length = max_arbitrary_length
        min_motion_length = min_arbitrary_length
        dataset_window_sizes = None  # 不使用特定窗口大小
    elif use_variable_length:
        # 变长窗口模式：使用最大窗口大小
        win_size = max_length
        max_motion_length = max_length
        min_motion_length = min_length
        dataset_window_sizes = None  # 不使用特定窗口大小
    else:
        # 固定窗口模式：不同数据集使用不同窗口大小
        # BEAT(0): 256帧，internet(1): 256帧，SeG(2): 64帧
        win_size = 256  # 默认窗口大小
        max_motion_length = 300
        min_motion_length = 256
        dataset_window_sizes = [256, 256, 64]  # 🔧 为SeG使用更短的64帧窗口
    
    # 🔧 选择数据集类
    if use_grouped_batches:
        # 使用分组数据集
        train_dataset = MixedMotionDatasetVQGrouped(
            data_root_list=data_root_list,
            split="train",
            mean=mean,
            std=std,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            win_size=win_size,
            unit_length=4,
            fps=20,
            tmpFile=True,
            tiny=False,
            debug=False,
            dataset_weights=dataset_weights,
            nfeats=491,
            dataset_window_sizes=dataset_window_sizes,  # 🔧 传递每个数据集的窗口大小
            dataset_name="Mixed_BEAT_v2_kimi_internet_data_SeG_kimi" if not use_seg_only else "SeG_kimi_only"
        )
    else:
        # 使用标准混合数据集
        train_dataset = MixedMotionDatasetVQ(
            data_root_list=data_root_list,
            split="train",
            mean=mean,
            std=std,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            win_size=win_size,
            unit_length=4,
            fps=20,
            tmpFile=True,
            tiny=False,
            debug=False,
            dataset_weights=dataset_weights,
            nfeats=491,
            dataset_name="Mixed_BEAT_v2_kimi_internet_data_SeG_kimi" if not use_seg_only else "SeG_kimi_only"
        )
    
    if is_dist_avail_and_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    # 🔧 选择collate函数
    if use_arbitrary_length:
        # 任意长度模式：保持原始序列长度
        collate_func = lambda batch: collate_fn_arbitrary_length(batch, max_arbitrary_length, min_arbitrary_length)
        if is_main_process():
            print(f"✅ 使用任意长度训练: 最小 {min_arbitrary_length} 帧，最大 {max_arbitrary_length} 帧")
    elif use_variable_length:
        # 变长窗口模式：随机选择窗口长度
        collate_func = lambda batch: collate_fn_variable_length(batch, min_length, max_length)
        if is_main_process():
            print(f"✅ 使用变长窗口训练: [{min_length}, {max_length}] 帧")
    else:
        # 固定窗口模式：256帧窗口
        collate_func = collate_fn
        if is_main_process():
            print(f"ℹ️  使用固定窗口训练: 256 帧")

    # 🔧 如果使用分组批次，需要特殊的sampler
    if use_grouped_batches and train_sampler is None:
        # 创建分组批次采样器
        grouped_sampler = GroupedBatchSampler(
            dataset=train_dataset,
            batch_size=32,
            dataset_weights=dataset_weights,
            num_batches_per_epoch=None  # 自动计算
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=grouped_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_func
        )
        if is_main_process():
            print(f"✅ 使用分组批次采样器（按数据集分组）")
    else:
        # 标准DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_func
        )
    
    return train_loader, mean, std


def compute_amplitude_loss(features, x_out, qpos_indices, loss_type="std", mean=None, std=None, mask=None):
    """
    计算幅度保持损失
    Args:
        features: 输入数据 [B, T, F]
        x_out: 重建输出 [B, T, F]
        qpos_indices: qpos维度索引 dict {'body': [...], 'hand': [...]}
        loss_type: 'std', 'range', or 'both'
        mean, std: 用于denormalize
        mask: 可选的mask张量 [B, T]
    Returns:
        amplitude_loss: 幅度保持损失
        stats: 统计信息
    """
    device = features.device
    stats = {}
    total_loss = 0.0
    
    # Denormalize到原始尺度
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).to(device)
        std_t = torch.tensor(std).to(device)
        features_denorm = features * std_t + mean_t
        x_out_denorm = x_out * std_t + mean_t
    else:
        features_denorm = features
        x_out_denorm = x_out
    
    # 计算body和hand的幅度损失
    for part, indices in qpos_indices.items():
        feat_qpos = features_denorm[:, :, indices]  # [B, T, D]
        recon_qpos = x_out_denorm[:, :, indices]
        
        if loss_type in ["std", "both"]:
            # 标准差损失 - 保持动作的方差
            # 🔧 支持mask：只在有效帧上计算std
            if mask is not None:
                feat_std_list = []
                recon_std_list = []
                for b in range(feat_qpos.shape[0]):
                    valid_mask = mask[b].bool()
                    if valid_mask.sum() > 1:  # 至少需要2帧才能计算std
                        feat_valid = feat_qpos[b, valid_mask, :]  # [T_valid, D]
                        recon_valid = recon_qpos[b, valid_mask, :]
                        feat_std_list.append(feat_valid.std(dim=0).mean())
                        recon_std_list.append(recon_valid.std(dim=0).mean())
                
                if feat_std_list:
                    feat_std = torch.stack(feat_std_list)
                    recon_std = torch.stack(recon_std_list)
                    std_loss = F.mse_loss(recon_std, feat_std)
                    stats[f'{part}_std_ratio'] = (recon_std.mean() / (feat_std.mean() + 1e-8)).item()
                else:
                    std_loss = torch.tensor(0.0, device=device)
            else:
                feat_std = feat_qpos.std(dim=1).mean(dim=1)  # [B]
                recon_std = recon_qpos.std(dim=1).mean(dim=1)  # [B]
                std_loss = F.mse_loss(recon_std, feat_std)
                stats[f'{part}_std_ratio'] = (recon_std.mean() / (feat_std.mean() + 1e-8)).item()
            
            total_loss += std_loss
            stats[f'{part}_std_loss'] = std_loss.item()
        
        if loss_type in ["range", "both"]:
            # 范围损失 - 保持动作的最大幅度
            # 🔧 支持mask：只在有效帧上计算range
            if mask is not None:
                feat_range_list = []
                recon_range_list = []
                for b in range(feat_qpos.shape[0]):
                    valid_mask = mask[b].bool()
                    if valid_mask.sum() > 0:
                        feat_valid = feat_qpos[b, valid_mask, :]  # [T_valid, D]
                        recon_valid = recon_qpos[b, valid_mask, :]
                        feat_range_list.append((feat_valid.max(dim=0)[0] - feat_valid.min(dim=0)[0]).mean())
                        recon_range_list.append((recon_valid.max(dim=0)[0] - recon_valid.min(dim=0)[0]).mean())
                
                if feat_range_list:
                    feat_range = torch.stack(feat_range_list)
                    recon_range = torch.stack(recon_range_list)
                    range_loss = F.mse_loss(recon_range, feat_range)
                    stats[f'{part}_range_ratio'] = (recon_range.mean() / (feat_range.mean() + 1e-8)).item()
                else:
                    range_loss = torch.tensor(0.0, device=device)
            else:
                feat_range = (feat_qpos.max(dim=1)[0] - feat_qpos.min(dim=1)[0]).mean(dim=1)  # [B]
                recon_range = (recon_qpos.max(dim=1)[0] - recon_qpos.min(dim=1)[0]).mean(dim=1)  # [B]
                range_loss = F.mse_loss(recon_range, feat_range)
                stats[f'{part}_range_ratio'] = (recon_range.mean() / (feat_range.mean() + 1e-8)).item()
            
            total_loss += range_loss
            stats[f'{part}_range_loss'] = range_loss.item()
    
    return total_loss, stats


def compute_velocity_loss(features, x_out, mask=None):
    """
    计算速度损失 - 保持动作的动态性
    Args:
        features: 输入数据 [B, T, F]
        x_out: 重建输出 [B, T, F]
        mask: 可选的mask张量 [B, T]
    Returns:
        velocity_loss: 速度损失
    """
    # 计算帧间差分（速度）
    feat_velocity = features[:, 1:, :] - features[:, :-1, :]  # [B, T-1, F]
    recon_velocity = x_out[:, 1:, :] - x_out[:, :-1, :]  # [B, T-1, F]
    
    # 🔧 应用mask（如果提供）
    if mask is not None:
        # velocity的mask：前后两帧都有效才有效
        velocity_mask = (mask[:, :-1] * mask[:, 1:]).to(feat_velocity.device)  # [B, T-1]
        
        # 速度的MSE损失
        loss_per_frame = F.mse_loss(feat_velocity, recon_velocity, reduction='none').mean(dim=2)  # [B, T-1]
        velocity_loss = (loss_per_frame * velocity_mask).sum() / (velocity_mask.sum() + 1e-8)
        
        # 速度幅度保持
        feat_vel_mag = feat_velocity.abs().mean(dim=2)  # [B, T-1]
        recon_vel_mag = recon_velocity.abs().mean(dim=2)  # [B, T-1]
        mag_loss_per_frame = (feat_vel_mag - recon_vel_mag).pow(2)
        vel_magnitude_loss = (mag_loss_per_frame * velocity_mask).sum() / (velocity_mask.sum() + 1e-8)
        
        vel_ratio = (recon_vel_mag * velocity_mask).sum() / ((feat_vel_mag * velocity_mask).sum() + 1e-8)
    else:
        # 速度的MSE损失
        velocity_loss = F.mse_loss(recon_velocity, feat_velocity)
        
        # 速度幅度保持
        feat_vel_magnitude = feat_velocity.abs().mean(dim=[1, 2])  # [B]
        recon_vel_magnitude = recon_velocity.abs().mean(dim=[1, 2])  # [B]
        vel_magnitude_loss = F.mse_loss(recon_vel_magnitude, feat_vel_magnitude)
        
        vel_ratio = (recon_vel_magnitude.mean() / (feat_vel_magnitude.mean() + 1e-8))
    
    total_vel_loss = velocity_loss + vel_magnitude_loss
    
    stats = {
        'velocity_loss': velocity_loss.item(),
        'vel_magnitude_loss': vel_magnitude_loss.item(),
        'vel_ratio': vel_ratio.item() if torch.is_tensor(vel_ratio) else vel_ratio
    }
    
    return total_vel_loss, stats


def compute_acceleration_loss(features, x_out, mask=None):
    """
    🆕 计算加速度损失 - 防止速度突变（二阶平滑）
    Args:
        features: 输入数据 [B, T, F]
        x_out: 重建输出 [B, T, F]
        mask: 可选的mask张量 [B, T]
    Returns:
        acceleration_loss: 加速度损失
        stats: 统计信息
    """
    # 计算速度（一阶导数）
    feat_velocity = features[:, 1:, :] - features[:, :-1, :]  # [B, T-1, F]
    recon_velocity = x_out[:, 1:, :] - x_out[:, :-1, :]  # [B, T-1, F]
    
    # 计算加速度（二阶导数）
    feat_accel = feat_velocity[:, 1:, :] - feat_velocity[:, :-1, :]  # [B, T-2, F]
    recon_accel = recon_velocity[:, 1:, :] - recon_velocity[:, :-1, :]  # [B, T-2, F]
    
    # 🔧 应用mask（如果提供）
    if mask is not None:
        # acceleration的mask：连续三帧都有效才有效
        accel_mask = (mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]).to(feat_accel.device)  # [B, T-2]
        
        # 加速度的MSE损失
        loss_per_frame = F.mse_loss(feat_accel, recon_accel, reduction='none').mean(dim=2)  # [B, T-2]
        accel_loss = (loss_per_frame * accel_mask).sum() / (accel_mask.sum() + 1e-8)
        
        # 加速度幅度保持
        feat_accel_mag = feat_accel.abs().mean(dim=2)  # [B, T-2]
        recon_accel_mag = recon_accel.abs().mean(dim=2)  # [B, T-2]
        mag_loss_per_frame = (feat_accel_mag - recon_accel_mag).pow(2)
        accel_magnitude_loss = (mag_loss_per_frame * accel_mask).sum() / (accel_mask.sum() + 1e-8)
        
        accel_ratio = (recon_accel_mag * accel_mask).sum() / ((feat_accel_mag * accel_mask).sum() + 1e-8)
    else:
        # 加速度的MSE损失
        accel_loss = F.mse_loss(recon_accel, feat_accel)
        
        # 加速度幅度保持
        feat_accel_magnitude = feat_accel.abs().mean(dim=[1, 2])  # [B]
        recon_accel_magnitude = recon_accel.abs().mean(dim=[1, 2])  # [B]
        accel_magnitude_loss = F.mse_loss(recon_accel_magnitude, feat_accel_magnitude)
        
        accel_ratio = (recon_accel_magnitude.mean() / (feat_accel_magnitude.mean() + 1e-8))
    
    total_accel_loss = accel_loss + accel_magnitude_loss
    
    stats = {
        'acceleration_loss': accel_loss.item(),
        'accel_magnitude_loss': accel_magnitude_loss.item(),
        'accel_ratio': accel_ratio.item() if torch.is_tensor(accel_ratio) else accel_ratio
    }
    
    return total_accel_loss, stats


def compute_loss_with_amplitude(features, x_out, quant_loss, config, mean=None, std=None,
                                dataset_indices=None, beta=0.25, mask=None):
    """
    包含幅度保持的完整损失函数
    Args:
        mask: 可选的mask张量 [B, T]，1表示有效帧，0表示padding
    """
    # 🔧 修复长度不匹配问题：取最小长度
    min_length = min(features.shape[1], x_out.shape[1])
    if features.shape[1] != x_out.shape[1]:
        features = features[:, :min_length, :]
        x_out = x_out[:, :min_length, :]
        if mask is not None:
            mask = mask[:, :min_length]
    
    # 1. 基础重建损失
    recon_loss_per_sample = F.mse_loss(x_out, features, reduction='none')  # [B, T, F]
    
    # 🔧 修复mask应用：先在特征维度平均，再在时间维度应用mask
    if mask is not None:
        # 先在特征维度求平均
        recon_loss_per_frame = recon_loss_per_sample.mean(dim=2)  # [B, T]
        
        # 然后在时间维度应用mask
        mask = mask.to(recon_loss_per_frame.device)
        recon_loss_per_sample = (recon_loss_per_frame * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B]
    else:
        recon_loss_per_sample = recon_loss_per_sample.mean(dim=[1, 2])  # [B]
    
    # 2. 语义加权
    stats = {}
    semantic_weight = config.get("semantic_weight", 10.0)
    if dataset_indices is not None:
        beat_mask = (dataset_indices == 0)
        internet_mask = (dataset_indices == 1)
        seg_mask = (dataset_indices == 2)
        
        if beat_mask.any():
            stats['beat_recon_loss'] = recon_loss_per_sample[beat_mask].mean().item()
        if internet_mask.any():
            stats['internet_recon_loss'] = recon_loss_per_sample[internet_mask].mean().item()
        if seg_mask.any():
            stats['seg_recon_loss'] = recon_loss_per_sample[seg_mask].mean().item()
        
        sample_weights = torch.ones_like(recon_loss_per_sample)
        semantic_mask = (dataset_indices == 2).float()
        sample_weights = sample_weights + semantic_mask * (semantic_weight - 1.0)
        recon_loss = (recon_loss_per_sample * sample_weights).sum() / sample_weights.sum()
    else:
        recon_loss = recon_loss_per_sample.mean()
    
    # 3. Amplitude preservation loss（支持mask）
    amplitude_loss = 0.0
    if config.get("use_amplitude_loss", False):
        amp_weight = config.get("amplitude_loss_weight", 0.1)
        amp_type = config.get("amplitude_loss_type", "std")
        qpos_indices = config.get("qpos_indices", {'body': list(range(246, 263)), 'hand': list(range(467, 491))})
        
        amp_loss, amp_stats = compute_amplitude_loss(features, x_out, qpos_indices, amp_type, mean, std, mask=mask)
        amplitude_loss = amp_weight * amp_loss
        stats.update(amp_stats)
        stats['amplitude_loss'] = amplitude_loss.item()
    
    # 4. Velocity loss（支持mask）
    velocity_loss = 0.0
    if config.get("use_velocity_loss", False):
        vel_weight = config.get("velocity_loss_weight", 0.05)
        vel_loss, vel_stats = compute_velocity_loss(features, x_out, mask=mask)
        velocity_loss = vel_weight * vel_loss
        stats.update(vel_stats)
        stats['velocity_loss'] = velocity_loss.item()
    
    # 5. 🆕 Acceleration loss（二阶平滑，防止跳变，支持mask）
    acceleration_loss = 0.0
    if config.get("use_acceleration_loss", False):
        accel_weight = config.get("acceleration_loss_weight", 0.2)
        accel_loss, accel_stats = compute_acceleration_loss(features, x_out, mask=mask)
        acceleration_loss = accel_weight * accel_loss
        stats.update(accel_stats)
        stats['acceleration_loss_weighted'] = acceleration_loss.item()
    
    # 6. 总损失
    total_loss = recon_loss + beta * quant_loss + amplitude_loss + velocity_loss + acceleration_loss
    
    return total_loss, recon_loss, quant_loss, stats


def train_vqvae(config, train_loader=None):
    # 设备初始化
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 加载mean和std（用于amplitude loss的denormalize）
    use_mixed_stats = config.get("use_mixed_stats", True)
    if use_mixed_stats:
        # 🔧 修复：优先使用Fixed版本（修复了std接近0的问题）
        mixed_stats_paths = [
            os.path.join(DATA_ROOT, "Mixed_Statistics_Equal_Fixed"),  # 优先
            os.path.join(DATA_ROOT, "Mixed_Statistics_Equal"),
            os.path.join(DATA_ROOT, "Mixed_Statistics"),
        ]
        mean_std_root = None
        for path in mixed_stats_paths:
            if os.path.exists(os.path.join(path, "Mean.npy")):
                mean_std_root = path
                if is_main_process():
                    print(f"✅ Using mixed dataset statistics from: {mean_std_root}")
                break
        
        if mean_std_root is None:
            if is_main_process():
                print(f"⚠️  Mixed statistics not found, falling back to BEAT_v2_kimi")
                print(f"   Run 'bash scripts/compute_mixed_stats.sh' to compute mixed statistics")
            mean_std_root = os.path.join(DATA_ROOT, "BEAT_v2_kimi")
    else:
        mean_std_root = os.path.join(DATA_ROOT, "BEAT_v2_kimi")
        if is_main_process():
            print(f"ℹ️  Using BEAT-only statistics from: {mean_std_root}")
    
    mean = np.load(os.path.join(mean_std_root, "Mean.npy"))
    std = np.load(os.path.join(mean_std_root, "Std.npy"))
    
    # 初始化模型
    motion_vae = VQVaeBodyHand(**config).to(device)
    if "ckpt" in config and config["ckpt"]:
        state_dict = torch.load(config["ckpt"], map_location=device, weights_only=False)
        motion_vae.load_state_dict(state_dict, strict=True)
        print(f"Loaded pre-trained model from {config['ckpt']}")
    
    # 训练参数
    base_lr = float(config.get("lr", 1e-4))
    base_beta = float(config.get("beta", 0.25))
    warmup_steps = int(config.get("warmup_steps", 1000))
    clip_grad_norm = float(config.get("clip_grad_norm", 1.0))
    
    # 学习率衰减
    use_lr_decay = config.get("use_lr_decay", False)
    lr_decay_start = int(config.get("lr_decay_start", 400))
    lr_decay_rate = float(config.get("lr_decay_rate", 0.98))
    lr_min = float(config.get("lr_min", 1e-5))
    
    # 两阶段训练
    use_two_stage = config.get("use_two_stage_training", True)
    stage1_epochs = int(config.get("stage1_epochs", 400))
    
    # 数据集权重
    seg_weight = float(config.get("seg_weight", 20.0))
    beat_weight = float(config.get("beat_weight", 1.0))
    internet_weight = float(config.get("internet_weight", 1.0))
    
    optimizer = torch.optim.Adam(motion_vae.parameters(), lr=0.0)
    
    # DDP
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        motion_vae = DDP(motion_vae, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    
    # 输出目录
    output_dir = config.get("output_dir", "output/vqvae_amplitude_preserved")
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    # 训练循环
    global_step = 0
    current_stage = 1 if use_two_stage else 2
    current_train_loader = None
    
    for epoch in tqdm(range(config["epochs"])):
        # 两阶段切换
        if use_two_stage and epoch == stage1_epochs:
            if is_main_process():
                print(f"\n{'='*60}")
                print(f"🔄 Switching to Stage 2: Mixed Training")
                print(f"{'='*60}\n")
            current_stage = 2
            # 同步所有进程，确保都切换到Stage 2
            if is_dist_avail_and_initialized():
                dist.barrier()
            if is_main_process():
                wandb.log({"training_stage": 2, "epoch": epoch})
        
        # 加载数据集
        # 🔧 如果使用256帧固定窗口，强制使用混合数据（不分阶段）
        # 因为SeG数据序列太短，无法满足256帧窗口要求
        use_seg_only = (current_stage == 1) and (not int(config.get("max_window_length", 256)) >= 256)
        use_mixed_stats = config.get("use_mixed_stats", True)
        use_variable_length = config.get("use_variable_length", False)
        use_arbitrary_length = config.get("use_arbitrary_length", False)
        min_length = int(config.get("min_window_length", 32))
        max_length = int(config.get("max_window_length", 128))
        max_arbitrary_length = int(config.get("max_arbitrary_length", 512))
        min_arbitrary_length = int(config.get("min_arbitrary_length", 8))
        use_grouped_batches = config.get("use_grouped_batches", False)  # 🔧 新增：分组批次
        
        if current_train_loader is None or (use_two_stage and epoch == stage1_epochs):
            current_train_loader, _, _ = load_dataset(
                use_seg_only=use_seg_only,
                seg_weight=seg_weight,
                beat_weight=beat_weight,
                internet_weight=internet_weight,
                use_mixed_stats=use_mixed_stats,
                use_variable_length=use_variable_length,
                min_length=min_length,
                max_length=max_length,
                use_arbitrary_length=use_arbitrary_length,
                max_arbitrary_length=max_arbitrary_length,
                min_arbitrary_length=min_arbitrary_length,
                use_grouped_batches=use_grouped_batches
            )
        
        total_loss = 0.0
        total_perplexity = 0.0
        epoch_stats = {}
        
        # 🔧 检查训练集是否为空
        if len(current_train_loader) == 0:
            if is_main_process():
                print(f"⚠️  Warning: Train loader is empty for epoch {epoch}. Skipping this epoch.")
            continue
        
        # 分布式sampler设置
        if hasattr(current_train_loader, 'sampler') and isinstance(current_train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            current_train_loader.sampler.set_epoch(epoch)
            
        for batch_idx, batch_data in enumerate(current_train_loader):
            # 🔧 处理不同collate_fn的返回值
            if len(batch_data) == 4:
                # 任意长度模式：(motions, dataset_indices, masks, original_lengths)
                motions, dataset_indices, masks, original_lengths = batch_data
                motions = motions.to(device, non_blocking=True)
                dataset_indices = dataset_indices.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                original_lengths = original_lengths.to(device, non_blocking=True)
            elif len(batch_data) == 3:
                # 变长窗口模式：(motions, dataset_indices, masks)
                motions, dataset_indices, masks = batch_data
                motions = motions.to(device, non_blocking=True)
                dataset_indices = dataset_indices.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                original_lengths = None
            else:
                # 固定窗口模式：(motions, dataset_indices)
                motions, dataset_indices = batch_data
                motions = motions.to(device, non_blocking=True)
                dataset_indices = dataset_indices.to(device, non_blocking=True)
                masks = None
                original_lengths = None
            
            # 前向传播
            x_out, quant_loss, perplexity = motion_vae(motions)
            
            # Warmup
            warmup_factor = 1.0 if warmup_steps <= 0 else min(1.0, global_step / warmup_steps)
            current_lr = base_lr * warmup_factor
            current_beta = base_beta * warmup_factor
            
            # 学习率衰减
            if use_lr_decay and epoch >= lr_decay_start:
                decay_factor = lr_decay_rate ** (epoch - lr_decay_start)
                current_lr = max(current_lr * decay_factor, lr_min)
            
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            
            # 计算损失（包含amplitude loss，支持mask）
            loss, recon_loss, quant_loss, batch_stats = compute_loss_with_amplitude(
                motions, x_out, quant_loss, config, mean, std,
                dataset_indices=dataset_indices,
                beta=current_beta,
                mask=masks
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(motion_vae.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            global_step += 1
            
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            
            # 累积统计
            for key, val in batch_stats.items():
                if key not in epoch_stats:
                    epoch_stats[key] = []
                epoch_stats[key].append(val)
            
            # 记录batch数据（每10个batch记录一次）
            if is_main_process() and batch_idx % 10 == 0:
                log_dict = {
                    "epoch": epoch,
                    "batch_loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "quant_loss": quant_loss.item(),
                    "perplexity": perplexity.item(),
                    "lr": current_lr,
                    "beta": current_beta,
                    "global_step": global_step,
                    "training_stage": current_stage,
                }
                log_dict.update(batch_stats)
                wandb.log(log_dict)
                
                # 🔧 新增：详细的console日志（每50个batch打印一次）
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx:>4}: "
                          f"Loss={loss.item():.4f}, "
                          f"Recon={recon_loss.item():.4f}, "
                          f"Quant={quant_loss.item():.4f}, "
                          f"Perp={perplexity.item():.1f}")
                    
                    # 如果有分数据集的损失，打印出来
                    if 'beat_recon_loss' in batch_stats:
                        print(f"    BEAT Loss: {batch_stats['beat_recon_loss']:.4f}")
                    if 'seg_recon_loss' in batch_stats:
                        print(f"    SeG Loss: {batch_stats['seg_recon_loss']:.4f}")
                    if 'internet_recon_loss' in batch_stats:
                        print(f"    Internet Loss: {batch_stats['internet_recon_loss']:.4f}")
                    
                    # 打印amplitude相关信息
                    if 'body_std_ratio' in batch_stats:
                        print(f"    Body Amp: {batch_stats['body_std_ratio']:.4f}, "
                              f"Hand Amp: {batch_stats.get('hand_std_ratio', 0):.4f}")
        
        # Epoch统计
        num_batches = max(len(current_train_loader), 1)  # 防止除零
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        # 计算平均幅度比例
        avg_amplitude_ratio = 0.0
        if 'body_std_ratio' in epoch_stats and 'hand_std_ratio' in epoch_stats:
            avg_amplitude_ratio = (np.mean(epoch_stats['body_std_ratio']) + np.mean(epoch_stats['hand_std_ratio'])) / 2.0
        
        if is_main_process():
            log_dict = {
                "epoch_loss": avg_loss,
                "epoch_perplexity": avg_perplexity,
                "epoch": epoch,
                "avg_amplitude_ratio": avg_amplitude_ratio,
            }
            
            # 添加epoch级别统计
            for key, values in epoch_stats.items():
                if values:
                    log_dict[f'epoch_{key}'] = np.mean(values)
            
            wandb.log(log_dict)
            
            # 🔧 新增：详细的epoch总结日志
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} | Stage {current_stage}")
            print(f"{'='*80}")
            print(f"📉 Overall Loss: {avg_loss:.6f} | Perplexity: {avg_perplexity:.2f}")
            
            # 分数据集的损失统计
            if 'beat_recon_loss' in epoch_stats and epoch_stats['beat_recon_loss']:
                beat_loss = np.mean(epoch_stats['beat_recon_loss'])
                print(f"📊 BEAT Recon Loss: {beat_loss:.6f}")
            
            if 'seg_recon_loss' in epoch_stats and epoch_stats['seg_recon_loss']:
                seg_loss = np.mean(epoch_stats['seg_recon_loss'])
                print(f"📊 SeG Recon Loss: {seg_loss:.6f}")
            
            if 'internet_recon_loss' in epoch_stats and epoch_stats['internet_recon_loss']:
                internet_loss = np.mean(epoch_stats['internet_recon_loss'])
                print(f"📊 Internet Recon Loss: {internet_loss:.6f}")
            
            # 计算损失比（诊断关键指标）
            if ('beat_recon_loss' in epoch_stats and epoch_stats['beat_recon_loss'] and 
                'seg_recon_loss' in epoch_stats and epoch_stats['seg_recon_loss']):
                beat_loss = np.mean(epoch_stats['beat_recon_loss'])
                seg_loss = np.mean(epoch_stats['seg_recon_loss'])
                loss_ratio = beat_loss / (seg_loss + 1e-8)
                print(f"⚖️  BEAT/SeG Loss Ratio: {loss_ratio:.3f}", end="")
                if loss_ratio > 2.0:
                    print(f" ❌ (目标<2.0, 严重不平衡)")
                elif loss_ratio > 1.5:
                    print(f" ⚠️  (目标<1.5, 轻微不平衡)")
                else:
                    print(f" ✅ (良好)")
            
            # Amplitude统计
            if avg_amplitude_ratio > 0:
                print(f"🎯 Avg Amplitude Ratio: {avg_amplitude_ratio:.4f} ({avg_amplitude_ratio*100:.1f}%)", end="")
                if avg_amplitude_ratio >= 0.95:
                    print(f" ✅ 优秀")
                elif avg_amplitude_ratio >= 0.90:
                    print(f" ✅ 良好")
                elif avg_amplitude_ratio >= 0.85:
                    print(f" ⚠️  一般")
                else:
                    print(f" ❌ 需改进")
            
            # 分body和hand的amplitude
            if 'body_std_ratio' in epoch_stats and epoch_stats['body_std_ratio']:
                body_amp = np.mean(epoch_stats['body_std_ratio'])
                print(f"   Body Amplitude: {body_amp:.4f} ({body_amp*100:.1f}%)")
            
            if 'hand_std_ratio' in epoch_stats and epoch_stats['hand_std_ratio']:
                hand_amp = np.mean(epoch_stats['hand_std_ratio'])
                print(f"   Hand Amplitude: {hand_amp:.4f} ({hand_amp*100:.1f}%)")
            
            # Loss组成详情
            if 'amplitude_loss' in epoch_stats and epoch_stats['amplitude_loss']:
                amp_loss = np.mean(epoch_stats['amplitude_loss'])
                print(f"📊 Amplitude Loss: {amp_loss:.6f}")
            
            if 'velocity_loss' in epoch_stats and epoch_stats['velocity_loss']:
                vel_loss = np.mean(epoch_stats['velocity_loss'])
                print(f"📊 Velocity Loss: {vel_loss:.6f}")
            
            print(f"{'='*80}\n")
        
        # 定期保存
        save_interval = int(config.get("save_interval", 50))
        if is_main_process() and epoch % save_interval == 0:
            target_state_dict = motion_vae.module.state_dict() if hasattr(motion_vae, 'module') else motion_vae.state_dict()
            save_path = f"{output_dir}/checkpoints/vqvae_epoch_{epoch}.pt"
            torch.save(target_state_dict, save_path)
            print(f"  ✅ Checkpoint saved: {save_path}")
        
        # 🔧 修复：在epoch结束时同步所有进程，避免有的进程快有的慢
        if is_dist_avail_and_initialized():
            dist.barrier()
    
    return motion_vae


class TeeLogger:
    """将输出同时写入终端和文件，完全兼容文件对象接口"""
    def __init__(self, file_path, mode='a', terminal=None):
        self.terminal = terminal if terminal else sys.stdout
        self.log = open(file_path, mode, encoding='utf-8', buffering=1)  # 行缓冲
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即刷新到文件
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        """检查终端是否是tty，WandB需要此方法"""
        return self.terminal.isatty() if hasattr(self.terminal, 'isatty') else False
    
    def fileno(self):
        """返回文件描述符，某些库可能需要"""
        return self.terminal.fileno() if hasattr(self.terminal, 'fileno') else -1
    
    def close(self):
        self.log.close()


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    
    # 支持通过环境变量指定配置文件
    config_name = os.environ.get("VQVAE_CONFIG", "g1_vqvae_amplitude_preserved.yaml")
    config_path = os.path.join(ROOT, "model", "motion_encoder", config_name)
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        exit(1)
    
    motion_config = open_yaml(config_path)
    
    # 🔧 新增：设置日志输出到文件（只在主进程）
    if is_main_process():
        # 创建日志目录
        output_dir = motion_config.get("output_dir", "output/vqvae_comprehensive")
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建带时间戳的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # 保存原始的stdout和stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # 重定向stdout和stderr到文件（同时保留终端输出）
        sys.stdout = TeeLogger(log_file, mode='w', terminal=original_stdout)
        sys.stderr = TeeLogger(log_file.replace('.log', '_error.log'), mode='w', terminal=original_stderr)
        
        print(f"=" * 80)
        print(f"📝 训练日志将同时保存到: {log_file}")
        print(f"=" * 80)
    
    print(f"\n{'='*60}")
    print(f"🚀 Amplitude-Preserved VQ-VAE Training")
    print(f"{'='*60}")
    print(f"✅ Amplitude Loss: {motion_config.get('use_amplitude_loss', False)}")
    print(f"✅ Velocity Loss: {motion_config.get('use_velocity_loss', False)}")
    print(f"✅ LR Decay: {motion_config.get('use_lr_decay', False)}")
    print(f"{'='*60}\n")
    
    if is_main_process():
        wandb.init(
            mode='offline',
            project="motion-vqvae-amplitude-preserved",
            entity="ritsu",
            name=f"amplitude_preserved_epochs{motion_config['epochs']}"
        )
        wandb.config.update(motion_config)
    
    trained_vae = train_vqvae(motion_config)
    
    if is_main_process():
        output_dir = motion_config.get("output_dir", "output/vqvae_amplitude_preserved")
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        target_state_dict = trained_vae.module.state_dict() if hasattr(trained_vae, 'module') else trained_vae.state_dict()
        final_path = f"{output_dir}/checkpoints/vqvae_final.pt"
        torch.save(target_state_dict, final_path)
        print(f"✅ Training completed! Final model saved to {final_path}")
        
        # 🔧 修复：正确关闭wandb，确保数据被保存
        wandb.finish()
        print(f"✅ WandB run finished and data saved")
        
        # 关闭日志文件
        if isinstance(sys.stdout, TeeLogger):
            print(f"\n{'='*80}")
            print(f"📝 训练日志已保存到: {sys.stdout.log.name}")
            print(f"{'='*80}")
            sys.stdout.close()
            sys.stderr.close()
    
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

