import os
import yaml
import torch
import wandb
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from HRI_mllm import ROOT
from HRI_mllm.model.motion_encoder.vqvae import VQVae
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.utils.motion_utils.g1ml3d import feats2joints
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.g1_29_humanml3d_representation import vec_to_data_pkl

import numpy as np

# 加载配置文件
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)

# 初始化 wandb
wandb.init(project="motion-vqvae", entity="ritsu")  # 替换为你的 wandb 用户名

# 加载数据集
def load_dataset():
    dataset = G1ML3DDataModule(stage="vae", split="train")
    train_dataset = dataset.train_dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataset = dataset.val_dataset
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    return train_loader, val_dataloader

# 在验证集上测试 MPJPE 和 PA-MPJPE
def validate(model, val_loader, device):
    model.eval()
    mpjpe_list, pa_mpjpe_list = [], []
    
    with torch.no_grad():
        for texts, motions in val_loader:
            motions = motions.to(device)
            code = model.encode(motions)
            decoded = model.decode(code[0])  # 解码生成运动
            
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


# 训练函数
def train_vqvae(config, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    motion_vae = VQVae(**config).to(device)
    optimizer = torch.optim.Adam(motion_vae.parameters(), lr=config.get("lr", 1e-4))
    
    # 训练循环
    for epoch in range(config["epochs"]):
        total_loss = 0.0
        for batch_idx, (texts, motions) in enumerate(train_loader):
            motions = motions.to(device)
            
            # 前向传播
            code, vq_loss, recon_loss = motion_vae(motions)
            loss = recon_loss + vq_loss  # 总损失 = 重构损失 + VQ 损失
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 记录 batch 数据到 wandb
            wandb.log({
                "epoch": epoch,
                "batch_loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "vq_loss": vq_loss.item(),
            })
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 计算并记录 epoch 数据
        avg_loss = total_loss / len(train_loader)
        wandb.log({"epoch_loss": avg_loss})
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
        
        # 保存模型 checkpoint
        if epoch % 5 == 0:
            torch.save(motion_vae.state_dict(), f"vqvae_epoch_{epoch}.pt")
            wandb.save(f"vqvae_epoch_{epoch}.pt")  # 上传到 wandb
    
    return motion_vae

# 主函数
if __name__ == "__main__":
    # 加载配置
    motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae.yaml"))
    motion_config["epochs"] = 50  # 训练 epoch 数
    motion_config["lr"] = 1e-4   # 学习率
    
    # 记录超参数到 wandb
    wandb.config.update(motion_config)
    
    # 加载数据
    train_loader = load_dataset()
    
    # 训练模型
    trained_vae = train_vqvae(motion_config, train_loader)
    
    # 保存最终模型
    torch.save(trained_vae.state_dict(), "vqvae_final.pt")
    wandb.save("vqvae_final.pt")  # 上传到 wandb