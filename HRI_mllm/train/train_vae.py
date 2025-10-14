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
from HRI_mllm.utils.motion_utils.g1ml3d import feats2joints
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    if motions.isnan().any():
        print("Found NaN in motion data")
        import ipdb;ipdb.set_trace()
    return motions
    
def load_dataset():
    dataset = G1ML3DDataModule(stage="vae", split="train", 
                               nfeats=491,
                               data_root=os.path.join(DATA_ROOT, "BEAT_v2_kimi"),
                               dis_data_root=os.path.join(DATA_ROOT, "BEAT_v2_kimi"), ### mean and std
                               dataset_name="BEAT_v2_kimi",
                               )
    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset

    # 分布式采样器
    if is_dist_avail_and_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=collate_fn)
    val_dataset = dataset.val_dataset
    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=collate_fn)
    return train_loader, val_loader

# 在验证集上测试 MPJPE 和 PA-MPJPE
def validate(model, val_loader, device):
    target_model = model.module if hasattr(model, "module") else model
    target_model.eval()
    mpjpe_list, pa_mpjpe_list = [], []
    
    with torch.no_grad():
        for motions in val_loader:
            motions = motions.to(device, non_blocking=True)
            code = target_model.encode(motions)

            decoded = target_model.decode(code[0]).reshape(motions.shape)  # 解码生成运动
            
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
    # 分布式聚合
    mpjpe_mean = np.mean(mpjpe_list) if len(mpjpe_list) > 0 else 0.0
    pampjpe_mean = np.mean(pa_mpjpe_list) if len(pa_mpjpe_list) > 0 else 0.0
    mpjpe_tensor = torch.tensor([mpjpe_mean], device=device, dtype=torch.float32)
    pampjpe_tensor = torch.tensor([pampjpe_mean], device=device, dtype=torch.float32)
    if is_dist_avail_and_initialized():
        dist.all_reduce(mpjpe_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(pampjpe_tensor, op=dist.ReduceOp.AVG)
    return mpjpe_tensor.item(), pampjpe_tensor.item()


def compute_loss(features, x_out, quant_loss, beta: float = 0.25):
    """
    Args:
        features: 输入数据 [B, ...]
        x_out: 重建输出 [B, ...]
        quant_loss: 量化器返回的loss（如 commitment loss）
        beta: commitment loss 的权重系数（默认参考 VQ-VAE 论文）
    Returns:
        total_loss: 总损失
        recon_loss: 重建损失
        quant_loss: 量化损失
    """
    # 1. 重建损失（假设输入是图像像素值，范围 [0,1]）
    recon_loss = F.mse_loss(x_out, features, reduction='mean')  # 或用 F.binary_cross_entropy
    
    # 2. 量化损失（直接使用 quantizer 返回的 loss）
    #    通常包含 codebook 的 L2 损失和 commitment loss
    
    # 3. 总损失（加权求和）
    total_loss = recon_loss + beta * quant_loss
    
    return total_loss, recon_loss, quant_loss


# 训练函数
def train_vqvae(config, train_loader, val_loader):
    # DDP: 设备与进程
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 初始化模型
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

    # 优化器以 0 学习率启动，逐步 warmup 到 base_lr
    optimizer = torch.optim.Adam(motion_vae.parameters(), lr=0.0)

    # 包裹 DDP（需在优化器之后/之前均可，这里在之后）
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        motion_vae = DDP(motion_vae, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
    
    # 训练循环
    global_step = 0
    for epoch in tqdm(range(config["epochs"])):
        total_loss = 0.0
        # 分布式 sampler 设定 epoch，确保各卡 shuffle 一致
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, motions in enumerate(train_loader):
            motions = motions.to(device, non_blocking=True)
            
            # 前向传播
            # Forward pass
            x_out, quant_loss, perplexity = motion_vae(motions)
            
            # Warmup: 线性从 0 -> base_lr 与 0 -> base_beta
            warmup_factor = 1.0 if warmup_steps <= 0 else min(1.0, global_step / warmup_steps)
            current_lr = base_lr * warmup_factor
            current_beta = base_beta * warmup_factor
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # 计算损失
            loss, recon_loss, quant_loss = compute_loss(motions, x_out, quant_loss, beta=current_beta)
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
            
            # 记录 batch 数据到 wandb
            if is_main_process():
                wandb.log({
                    "epoch": epoch,
                    "batch_loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "quant_loss": quant_loss.item(),
                    "perplexity": perplexity.item(),
                    "lr": current_lr,
                    "beta": current_beta,
                    "global_step": global_step,
                })
            
            # if is_main_process() and batch_idx == 0 and epoch % 50 == 0:  # 周期性打印
            #     print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            #     mpjpe, pampjpe = validate(motion_vae, val_loader, device)
            #     print(f"Validation MPJPE: {mpjpe:.4f}, PA-MPJPE: {pampjpe:.4f}")
            #     wandb.log({
            #         "validation_mpjpe": mpjpe,
            #         "validation_pampjpe": pampjpe,
            #     })
        
        # 计算并记录 epoch 数据
        avg_loss = total_loss / len(train_loader)
        if is_main_process():
            wandb.log({"epoch_loss": avg_loss})
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
        
        # 保存模型 checkpoint
        if is_main_process() and epoch % 50 == 0:
            os.makedirs("output/VQVAE_full/checkpoints", exist_ok=True)
            target_state_dict = motion_vae.module.state_dict() if hasattr(motion_vae, 'module') else motion_vae.state_dict()
            torch.save(target_state_dict, f"output/VQVAE_full/checkpoints/vqvae_epoch_{epoch}.pt")
            wandb.save(f"vqvae_epoch_{epoch}.pt")  # 上传到 wandb
    
    return motion_vae

# 主函数
if __name__ == "__main__":
    # 分布式初始化（torchrun 启动时）
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    # 加载配置
    motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae_full.yaml"))
    motion_config["epochs"] = motion_config.get("epochs", 1000)
    motion_config["lr"] = motion_config.get("lr", 1e-4)
    motion_config["beta"] = motion_config.get("beta", 0.25)
    motion_config["warmup_steps"] = motion_config.get("warmup_steps", 1000)
    motion_config["clip_grad_norm"] = motion_config.get("clip_grad_norm", 1.0)
    
    # 仅主进程初始化 wandb
    if is_main_process():
        wandb.init(mode='online', project="motion-vqvae", entity="ritsu")
        wandb.config.update(motion_config)
    
    # 加载数据
    train_loader, val_loader = load_dataset()
    
    # 训练模型
    trained_vae = train_vqvae(motion_config, train_loader, val_loader)
    
    # 保存最终模型（仅主进程）
    if is_main_process():
        os.makedirs("output/VQVAE_full/checkpoints", exist_ok=True)
        target_state_dict = trained_vae.module.state_dict() if hasattr(trained_vae, 'module') else trained_vae.state_dict()
        torch.save(target_state_dict, "output/VQVAE_full/checkpoints/vqvae_final_v2.pt")
        wandb.save("vqvae_final_v2.pt")

    # 结束分布式
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()