import os
import yaml
import torch
import wandb
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae import VQVae
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.utils.motion_utils.g1ml3d import feats2joints
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 加载配置文件
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)

# 初始化 wandb
wandb.init(mode='offline', project="motion-vqvae", entity="ritsu")  # 替换为你的 wandb 用户名

# 加载数据集

def collate_fn(batch):
    motions = torch.stack([torch.from_numpy(item[1]) for item in batch])
    if motions.isnan().any():
        print("Found NaN in motion data")
        import ipdb;ipdb.set_trace()
    return motions
    
def load_dataset():
    dataset = G1ML3DDataModule(stage="vae", split="train", 
                               nfeats=179,
                               data_root=os.path.join(DATA_ROOT, "BEAT_v1_kimi"),
                               dis_data_root=os.path.join(DATA_ROOT, "BEAT_v1_kimi"), ### mean and std
                               dataset_name="BEAT_v1_kimi",
                               )
    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset

    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataset = dataset.val_dataset
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader

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
    total_loss = recon_loss + quant_loss  # 如果 quant_loss 已包含 beta 权重
    
    return total_loss, recon_loss, quant_loss


# 训练函数
def train_vqvae(config, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    motion_vae = VQVae(**config).to(device)
    # 加载预训练模型（如果有的话）
    if "ckpt" in config and config["ckpt"]:
        state_dict = torch.load(config["ckpt"], map_location=device, weights_only=False)
        motion_vae.load_state_dict(state_dict, strict=True)
        print(f"Loaded pre-trained model from {config['ckpt']}")



    optimizer = torch.optim.Adam(motion_vae.parameters(), lr=config.get("lr", 1e-4))
    
    # 训练循环
    for epoch in tqdm(range(config["epochs"])):
        total_loss = 0.0
        for batch_idx, motions in enumerate(train_loader):
            motions = motions.to(device)
            
            # 前向传播
            # Forward pass
            x_out, quant_loss, perplexity = motion_vae(motions)
    
            # 计算损失
            loss, recon_loss, quant_loss = compute_loss(motions, x_out, quant_loss, beta=0.25)
            if loss.isnan().any():
                import ipdb;ipdb.set_trace()
            
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
                "quant_loss": quant_loss.item(),
                "perplexity": perplexity.item(),
            })
            
            if batch_idx == 0 and epoch % 10 == 0:  # 每10个epoch打印一次
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                mpjpe, pampjpe = validate(motion_vae, val_loader, device)
                print(f"Validation MPJPE: {mpjpe:.4f}, PA-MPJPE: {pampjpe:.4f}")
                wandb.log({
                    "validation_mpjpe": mpjpe,
                    "validation_pampjpe": pampjpe,
                })
        
        # 计算并记录 epoch 数据
        avg_loss = total_loss / len(train_loader)
        wandb.log({"epoch_loss": avg_loss})
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
        
        # 保存模型 checkpoint
        if epoch % 5 == 0:
            torch.save(motion_vae.state_dict(), f"output/VQVAE_body/checkpoints/vqvae_epoch_{epoch}.pt")
            wandb.save(f"vqvae_epoch_{epoch}.pt")  # 上传到 wandb
    
    return motion_vae

# 主函数
if __name__ == "__main__":
    # 加载配置
    motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae_body.yaml"))
    motion_config["epochs"] = 500  # 训练 epoch 数
    motion_config["lr"] = 1e-4   # 学习率
    
    # 记录超参数到 wandb
    wandb.config.update(motion_config)
    
    # 加载数据
    train_loader, val_loader = load_dataset()
    
    # 训练模型
    trained_vae = train_vqvae(motion_config, train_loader, val_loader)
    
    # 保存最终模型
    torch.save(trained_vae.state_dict(), "vqvae_final.pt")
    wandb.save("vqvae_final.pt")  # 上传到 wandb