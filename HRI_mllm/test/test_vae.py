import os
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
import yaml
import torch
import wandb
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae import VQVae
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.utils.motion_utils.g1ml3d import feats2datapkl, feats2joints, normalize_features
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import joblib


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
                               nfeats=280,
                               data_root=os.path.join(DATA_ROOT, "BEAT_TTS"),
                               dis_data_root=os.path.join(DATA_ROOT, "G1ML3D_v1"), ### mean and std
                               dataset_name="BEAT_TTS",
                               )
    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset
    # for _, motions, length, _, _, _, _, name, idx in train_dataset:
    #     motions = torch.from_numpy(motions)
    #     if motions.isnan().any():
    #         print("Found NaN in motion data")
    #         import ipdb;ipdb.set_trace()
    #     if motions.shape[0] < 64:
    #         print(f"Skipping motion with length {motions.shape[0]}")
    #         continue
    #     if motions.shape[1] != 280:  # 确保关节数量正确
    #         print(f"Skipping motion with incorrect joint count: {motions.shape[1]}")
    #         continue
    

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




# 主函数
if __name__ == "__main__":
    
    # 加载数据
    train_loader, val_loader = load_dataset()
    

    ### loading motion vae 
    def open_yaml(path):
        with open(path, 'r', encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data    
    motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae.yaml"))
    motion_vae = VQVae(**motion_config)
    state_dict = torch.load(motion_config["ckpt"], map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device="cuda")

    ### compute mpjpe
    mpjpe, pampjpe = validate(motion_vae, train_loader, device="cuda")
    print(f"Validation MPJPE: {mpjpe:.4f}, PA-MPJPE: {pampjpe:.4f}")

    ### encode&decode for specific motion

    train_data_feature = np.load(f"/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS/new_joint_vecs/6_carla_0_63_63_joint_vecs.npy")

    import ipdb;ipdb.set_trace()

    motion_tokens = motion_vae.encode(normalize_features(torch.from_numpy(train_data_feature).unsqueeze(0).to("cuda:0")))[0].detach().cpu()

    decoded_features = motion_vae.decode(motion_tokens.to("cuda:0")).detach().cpu()

    decoded_data_pkl = feats2datapkl(decoded_features)
    source_data_pkl = feats2datapkl(torch.from_numpy(train_data_feature).unsqueeze(0))


    with open("source.pkl", 'wb') as f:
        pickle.dump(source_data_pkl, f)
    with open("decoded.pkl", 'wb') as f:
        pickle.dump(decoded_data_pkl, f)
    

    source_csv =  load_motion_pkl_as_csv_data("source.pkl")
    decoded_csv =  load_motion_pkl_as_csv_data("decoded.pkl")

    np.savetxt("source.csv", source_csv, delimiter=',', fmt='%.8f')
    np.savetxt("decoded.csv", decoded_csv, delimiter=',', fmt='%.8f')
    