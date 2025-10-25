"""
语义增强的VQ-VAE训练脚本
支持：
1. 语义感知的加权损失
2. 分数据集的详细监控
3. 两阶段训练策略
4. 灵活的超参数配置
"""
import os
import yaml
import torch
import wandb
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae import VQVae
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.datasets.MixedMotionDatasetVQ import MixedMotionDatasetVQ
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

qpos_mask = torch.zeros(491)
qpos_mask[263-17:263] = 1
qpos_mask[491-24:491] = 1

# 加载配置文件
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

# 加载数据集

def collate_fn(batch):
    motions = torch.stack([torch.from_numpy(item[1]).float() for item in batch])
    dataset_indices = torch.tensor([item[6] for item in batch], dtype=torch.long)
    if motions.isnan().any():
        print("Found NaN in motion data")
        import ipdb;ipdb.set_trace()
    
    return motions, dataset_indices
    
def load_dataset(use_seg_only=False, seg_weight=10.0, beat_weight=1.0):
    """
    加载数据集
    Args:
        use_seg_only: 是否只使用SeG数据集（用于两阶段训练的第一阶段）
        seg_weight: SeG数据集的采样权重
        beat_weight: BEAT数据集的采样权重
    """
    # 定义混合数据集的配置
    if use_seg_only:
        # 第一阶段：只使用SeG数据集
        data_root_list = [
            os.path.join(DATA_ROOT, "SeG_kimi")
        ]
        dataset_weights = [1.0]
    else:
        # 第二阶段或标准训练：使用混合数据集
        data_root_list = [
            os.path.join(DATA_ROOT, "BEAT_v2_kimi"),
            os.path.join(DATA_ROOT, "SeG_kimi")
        ]
        dataset_weights = [beat_weight, seg_weight]
    
    # 加载mean和std（使用beat_v2_kimi的统计信息作为基准）
    mean_std_root = os.path.join(DATA_ROOT, "BEAT_v2_kimi")
    mean = np.load(os.path.join(mean_std_root, "Mean.npy"))
    std = np.load(os.path.join(mean_std_root, "Std.npy"))
    
    # 创建混合训练数据集
    train_dataset = MixedMotionDatasetVQ(
        data_root_list=data_root_list,
        split="train",
        mean=mean,
        std=std,
        max_motion_length=196,
        min_motion_length=32,  # 降低以适应SeG数据集
        win_size=32,  # 降低到32以保留几乎所有SeG样本（544个中535个）
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        dataset_weights=dataset_weights,
        nfeats=491,
        dataset_name="Mixed_BEAT_v2_kimi_SeG_kimi" if not use_seg_only else "SeG_kimi_only"
    )
    
    # 分布式采样器
    if is_dist_avail_and_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=collate_fn)
    
    return train_loader



def compute_loss_with_stats(features, x_out, quant_loss, beta: float = 0.25, 
                            dataset_indices=None, semantic_weight: float = 5.0):
    """
    语义感知的损失函数，并返回详细的统计信息
    Args:
        features: 输入数据 [B, T, F]
        x_out: 重建输出 [B, T, F]
        quant_loss: 量化器返回的loss
        beta: commitment loss 的权重系数
        dataset_indices: 数据集索引 [B], 0=BEAT, 1=SeG
        semantic_weight: 语义动作（SeG数据集）的额外权重
    Returns:
        total_loss: 总损失
        recon_loss: 加权的重建损失
        quant_loss: 量化损失
        stats: 详细统计信息字典
    """
    # 1. 计算逐样本的重建损失
    mask = qpos_mask.to(features.device)
    recon_loss_per_sample = F.mse_loss(x_out * mask, features * mask, reduction='none')
    recon_loss_per_sample = recon_loss_per_sample.mean(dim=[1, 2])  # [B]
    
    # 2. 收集统计信息
    stats = {}
    if dataset_indices is not None:
        beat_mask = (dataset_indices == 0)
        seg_mask = (dataset_indices == 1)
        
        if beat_mask.any():
            stats['beat_recon_loss'] = recon_loss_per_sample[beat_mask].mean().item()
            stats['beat_count'] = beat_mask.sum().item()
        else:
            stats['beat_recon_loss'] = 0.0
            stats['beat_count'] = 0
            
        if seg_mask.any():
            stats['seg_recon_loss'] = recon_loss_per_sample[seg_mask].mean().item()
            stats['seg_count'] = seg_mask.sum().item()
        else:
            stats['seg_recon_loss'] = 0.0
            stats['seg_count'] = 0
    
    # 3. 应用语义感知权重
    if dataset_indices is not None:
        # dataset_idx=1 表示SeG数据集（语义动作），给予更高权重
        sample_weights = torch.ones_like(recon_loss_per_sample)
        semantic_mask = (dataset_indices == 1).float()
        sample_weights = sample_weights + semantic_mask * (semantic_weight - 1.0)
        
        # 加权平均
        recon_loss = (recon_loss_per_sample * sample_weights).sum() / sample_weights.sum()
    else:
        recon_loss = recon_loss_per_sample.mean()
    
    # 4. 总损失（加权求和）
    total_loss = recon_loss + beta * quant_loss
    
    return total_loss, recon_loss, quant_loss, stats


# 训练函数
def train_vqvae(config, train_loader=None):
    # DDP: 设备与进程
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 初始化标准VQ-VAE模型
    motion_vae = VQVaeBodyHand(**config).to(device)
    # 加载预训练模型（如果有的话）
    if "ckpt" in config and config["ckpt"]:
        state_dict = torch.load(config["ckpt"], map_location=device, weights_only=False)
        motion_vae.load_state_dict(state_dict, strict=True)
        print(f"Loaded pre-trained model from {config['ckpt']}")

    
    
    base_lr = float(config.get("lr", 1e-4))
    base_beta = float(config.get("beta", 0.25))
    warmup_steps = int(config.get("warmup_steps", 1000))
    clip_grad_norm = float(config.get("clip_grad_norm", 1.0))
    semantic_weight = float(config.get("semantic_weight", 5.0))
    
    # 两阶段训练配置
    use_two_stage = config.get("use_two_stage_training", False)
    stage1_epochs = config.get("stage1_epochs", 500)
    
    # 数据集权重配置
    seg_weight = config.get("seg_weight", 10.0)
    beat_weight = config.get("beat_weight", 1.0)

    # 优化器以 0 学习率启动，逐步 warmup 到 base_lr
    optimizer = torch.optim.Adam(motion_vae.parameters(), lr=0.0)

    # 包裹 DDP（需在优化器之后/之前均可，这里在之后）
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        motion_vae = DDP(motion_vae, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    
    # 输出目录
    output_dir = config.get("output_dir", "output/vqvae_semantic_enhanced")
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    # 训练循环
    global_step = 0
    current_stage = 1 if use_two_stage else 2  # 1=SeG only, 2=Mixed
    current_train_loader = None
    
    for epoch in tqdm(range(config["epochs"])):
        # 两阶段训练：在stage1_epochs时切换
        if use_two_stage and epoch == stage1_epochs:
            print(f"\n{'='*60}")
            print(f"Switching to Stage 2: Mixed Training (BEAT + SeG)")
            print(f"{'='*60}\n")
            current_stage = 2
            # 重新加载数据集
            if is_main_process():
                wandb.log({"training_stage": 2, "epoch": epoch})
        
        # 根据当前阶段加载数据集
        use_seg_only = (current_stage == 1)
        if current_train_loader is None or (use_two_stage and epoch == stage1_epochs):
            current_train_loader = load_dataset(
                use_seg_only=use_seg_only,
                seg_weight=seg_weight,
                beat_weight=beat_weight
            )
        
        total_loss = 0.0
        total_stats = {'beat_recon_loss': 0.0, 'seg_recon_loss': 0.0, 
                      'beat_count': 0, 'seg_count': 0}
        
        # 分布式 sampler 设定 epoch，确保各卡 shuffle 一致
        if hasattr(current_train_loader, 'sampler') and isinstance(current_train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            current_train_loader.sampler.set_epoch(epoch)
            
        for batch_idx, batch_data in enumerate(current_train_loader):
            # 处理批次数据
            motions, dataset_indices = batch_data
            motions = motions.to(device, non_blocking=True)
            dataset_indices = dataset_indices.to(device, non_blocking=True)
            
            # 前向传播
            x_out, quant_loss, perplexity = motion_vae(motions)
            
            # Warmup: 线性从 0 -> base_lr 与 0 -> base_beta
            warmup_factor = 1.0 if warmup_steps <= 0 else min(1.0, global_step / warmup_steps)
            current_lr = base_lr * warmup_factor
            current_beta = base_beta * warmup_factor
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # 计算损失（使用语义感知的加权）
            loss, recon_loss, quant_loss, batch_stats = compute_loss_with_stats(
                motions, x_out, quant_loss, 
                beta=current_beta,
                dataset_indices=dataset_indices,
                semantic_weight=semantic_weight
            )
            
            if loss.isnan().any():
                import ipdb;ipdb.set_trace()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(motion_vae.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            global_step += 1
            
            total_loss += loss.item()
            
            # 累积统计信息
            for key in total_stats:
                if key in batch_stats:
                    total_stats[key] += batch_stats[key]
            
            # 记录 batch 数据到 wandb
            if is_main_process() and batch_idx % 10 == 0:  # 每10个batch记录一次
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
                # 添加分数据集的统计
                if batch_stats.get('beat_count', 0) > 0:
                    log_dict['batch_beat_recon_loss'] = batch_stats['beat_recon_loss']
                if batch_stats.get('seg_count', 0) > 0:
                    log_dict['batch_seg_recon_loss'] = batch_stats['seg_recon_loss']
                
                wandb.log(log_dict)
            
        
        # 计算并记录 epoch 数据
        avg_loss = total_loss / len(current_train_loader)
        
        if is_main_process():
            log_dict = {
                "epoch_loss": avg_loss,
                "epoch": epoch,
            }
            
            # 计算epoch级别的统计
            if total_stats['beat_count'] > 0:
                avg_beat_loss = total_stats['beat_recon_loss'] / len(current_train_loader)
                log_dict['epoch_beat_recon_loss'] = avg_beat_loss
            
            if total_stats['seg_count'] > 0:
                avg_seg_loss = total_stats['seg_recon_loss'] / len(current_train_loader)
                log_dict['epoch_seg_recon_loss'] = avg_seg_loss
            
            # 如果两个数据集都有，计算比值
            if total_stats['beat_count'] > 0 and total_stats['seg_count'] > 0:
                ratio = avg_seg_loss / (avg_beat_loss + 1e-8)
                log_dict['seg_to_beat_loss_ratio'] = ratio
            
            wandb.log(log_dict)
            
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
            if total_stats['beat_count'] > 0:
                print(f"  BEAT Recon Loss: {avg_beat_loss:.4f}")
            if total_stats['seg_count'] > 0:
                print(f"  SeG Recon Loss: {avg_seg_loss:.4f}")
        
        # 保存模型 checkpoint
        save_interval = config.get("save_interval", 50)
        if is_main_process() and epoch % save_interval == 0:
            target_state_dict = motion_vae.module.state_dict() if hasattr(motion_vae, 'module') else motion_vae.state_dict()
            save_path = f"{output_dir}/checkpoints/vqvae_epoch_{epoch}.pt"
            torch.save(target_state_dict, save_path)
            wandb.save(save_path)
    
    return motion_vae

# 主函数
if __name__ == "__main__":
    # 分布式初始化（torchrun 启动时）
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    # 加载配置
    config_path = os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae_semantic_enhanced.yaml")
    if not os.path.exists(config_path):
        # 如果增强配置不存在，使用原配置
        config_path = os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae_full_qpos.yaml")
    
    motion_config = open_yaml(config_path)
    
    # 确保必要的参数存在
    motion_config["epochs"] = motion_config.get("epochs", 3000)
    motion_config["lr"] = motion_config.get("lr", 1e-4)
    motion_config["beta"] = motion_config.get("beta", 0.25)
    motion_config["warmup_steps"] = motion_config.get("warmup_steps", 1000)
    motion_config["clip_grad_norm"] = motion_config.get("clip_grad_norm", 1.0)
    motion_config["semantic_weight"] = motion_config.get("semantic_weight", 5.0)
    motion_config["seg_weight"] = motion_config.get("seg_weight", 10.0)
    motion_config["beat_weight"] = motion_config.get("beat_weight", 1.0)
    
    # 仅主进程初始化 wandb
    if is_main_process():
        wandb.init(
            mode='offline', 
            project="motion-vqvae-semantic", 
            entity="ritsu",
            name=f"semantic_enhanced_sw{motion_config['semantic_weight']}_dw{motion_config['seg_weight']}"
        )
        wandb.config.update(motion_config)
    
    # 训练模型
    trained_vae = train_vqvae(motion_config)
    
    # 保存最终模型（仅主进程）
    if is_main_process():
        output_dir = motion_config.get("output_dir", "output/VQVAE_semantic_enhanced")
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        target_state_dict = trained_vae.module.state_dict() if hasattr(trained_vae, 'module') else trained_vae.state_dict()
        final_path = f"{output_dir}/checkpoints/vqvae_final.pt"
        torch.save(target_state_dict, final_path)
        wandb.save(final_path)
        print(f"Training completed! Final model saved to {final_path}")

    # 结束分布式
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

