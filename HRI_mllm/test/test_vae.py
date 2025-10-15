import os

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
from HRI_mllm.utils.motion_utils.g1ml3d import feats2datapkl, feats2joints, normalize_vec
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe
from HRI_mllm.train.train_vae import load_dataset
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

    

    ### loading motion vae 
    def open_yaml(path):
        with open(path, 'r', encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data    
    motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae_full.yaml"))
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(motion_config["ckpt"], map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device="cuda")

    # ### compute mpjpe
        
    # # 加载数据
    # train_loader, val_loader = load_dataset()
    # mpjpe, pampjpe = validate(motion_vae, train_loader, device="cuda")
    # print(f"Validation MPJPE: {mpjpe:.4f}, PA-MPJPE: {pampjpe:.4f}")

    ### encode&decode for specific motion


    beat_vec_path = f"/root/workspace/HRI_MLLM/data/BEAT_v2_kimi/new_joint_vecs/1_wayne_0_1_1.npy"
    train_data_vec = np.load(beat_vec_path)
    beat_filename = beat_vec_path.split("/")[-1]
    audio_path = os.path.join("/root/workspace/HRI_MLLM/data/BEAT_v2", beat_filename.split("_")[0], beat_filename.replace(".npy", ".wav"))



    motion_tokens = motion_vae.encode(normalize_vec(torch.from_numpy(train_data_vec).unsqueeze(0).to("cuda:0")))[0]

    decoded_features = motion_vae.decode(motion_tokens).detach().cpu()

    decoded_data_pkl = feats2datapkl(decoded_features)
    source_data_pkl = feats2datapkl(normalize_vec(torch.from_numpy(train_data_vec).unsqueeze(0).to("cuda:0")))


    with open("source.pkl", 'wb') as f:
        pickle.dump(source_data_pkl, f)
    with open("decoded.pkl", 'wb') as f:
        pickle.dump(decoded_data_pkl, f)
    

    source_csv =  load_motion_pkl_as_csv_data("source.pkl")
    decoded_csv =  load_motion_pkl_as_csv_data("decoded.pkl")


    np.savetxt("source.csv", source_csv, delimiter=',', fmt='%.8f')
    np.savetxt("decoded.csv", decoded_csv, delimiter=',', fmt='%.8f')

    vis_audio_motion(audio_path, "source.csv", output_path="final_output_source.mp4", robot_type="g1_brainco", rate_limit=False)
    vis_audio_motion(audio_path, "decoded.csv", output_path="final_output_decoded.mp4", robot_type="g1_brainco", rate_limit=False)
    