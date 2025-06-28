from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm import ROOT

from collections import OrderedDict
from HRI_mllm.utils.motion_utils.g1ml3d import feats2joints
import yaml
import os
import torch
import sys
import pickle
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data        

### loading motion VQVAE
motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae.yaml"))
motion_vae = VQVae(**motion_config)
state_dict = torch.load(motion_config["ckpt"], map_location="cpu", weights_only=False)
motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")



### loading G1ML3D dataset
dataset = G1ML3DDataModule(stage="vae", split="train")
train_dataset = dataset.train_dataset




### decode 280 dim feature into g1 motion pickle
import os
from pathlib import Path 
from HRI_mllm import ROOT

OUTPUT_DIR = os.path.join(Path(ROOT).parent,"output", "g1_motion")


### testing the motion VQVAE
from HRI_mllm.utils.motion_utils.g1ml3d import feats2joints
with torch.no_grad():
        
    for idx in range(10):
        test_sample = torch.from_numpy(train_dataset[idx][1]).unsqueeze(0).cuda()
        text = train_dataset[idx][0]
        code = motion_vae.encode(test_sample)
        decoded = motion_vae.decode(code[0])


        print(decoded.shape, text, "mpjpe:", calc_mpjpe(feats2joints(test_sample)[0], feats2joints(decoded)[0]).mean(),
            "pampjpe:", calc_pampjpe(feats2joints(test_sample)[0], feats2joints(decoded)[0]).mean())
        

        data_dict_gt = vec_to_data_pkl(test_sample)
        data_dict_decoded = vec_to_data_pkl(decoded)

        data_dict_gt['text'] = text
        data_dict_decoded['text'] = text

        with open(os.path.join(OUTPUT_DIR, f"g1ml3d_{idx}_gt.pkl"), 'wb') as f:
            pickle.dump(data_dict_gt, f)
        with open(os.path.join(OUTPUT_DIR, f"g1ml3d_{idx}_decoded.pkl"), 'wb') as f:
            pickle.dump(data_dict_decoded, f)




