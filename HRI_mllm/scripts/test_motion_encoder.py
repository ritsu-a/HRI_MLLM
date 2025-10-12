from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm import DATA_ROOT, ROOT

from collections import OrderedDict
from HRI_mllm.utils.motion_utils.g1ml3d import feats2joints
import yaml
import os
import torch
import sys
import pickle
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl, feats2joints


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data        

### loading motion VQVAE
motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae_full.yaml"))
motion_vae = VQVae(**motion_config)
state_dict = torch.load(motion_config["ckpt"], map_location="cpu", weights_only=False)
motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")



### loading G1ML3D dataset
dataset = G1ML3DDataModule(stage="vae", split="train", 
                            nfeats=383,
                            data_root=os.path.join(DATA_ROOT, "BEAT_v2_kimi"),
                            dis_data_root=os.path.join(DATA_ROOT, "BEAT_v2_kimi"), ### mean and std
                            dataset_name="BEAT_v2_kimi",
                            )
train_dataset = dataset.train_dataset




### decode 383 dim feature into g1 motion pickle
import os
from pathlib import Path 
from HRI_mllm import ROOT

OUTPUT_DIR = os.path.join(Path(ROOT).parent,"output", "g1_motion")


### testing the motion VQVAE
from HRI_mllm.utils.motion_utils.g1ml3d import feats2datapkl


def merge_data_list(data_dict_list):
    merged_data = {
        'fps': data_dict_list[0]['fps'],
        'robot_name': data_dict_list[0]['robot_name'],
        'angles': torch.cat([data_dict['angles'] for data_dict in data_dict_list]),
        'global_rotation': torch.cat([data_dict['global_rotation'] for data_dict in data_dict_list]),
        'global_translation': torch.cat([data_dict['global_translation'] for data_dict in data_dict_list]),
        'scale': data_dict_list[0]['scale'],
        'text': [data_dict['text'] for data_dict in data_dict_list],
    }

   
    return merged_data


gt_data_dict_list = []
decoded_data_dict_list = []
with torch.no_grad():
        
    for idx in range(10):
        ### TODO: seems lack normalization?
        test_sample = torch.from_numpy(train_dataset[idx][1]).unsqueeze(0).cuda()
        text = train_dataset[idx][0]
        code = motion_vae.encode(test_sample)
        decoded = motion_vae.decode(code[0])


        print(decoded.shape, text, "mpjpe:", calc_mpjpe(feats2joints(test_sample)[0], feats2joints(decoded)[0]).mean(),
            "pampjpe:", calc_pampjpe(feats2joints(test_sample)[0], feats2joints(decoded)[0]).mean())
        

        data_dict_gt = feats2datapkl(test_sample)
        data_dict_decoded = feats2datapkl(decoded)

        data_dict_gt['text'] = text
        data_dict_decoded['text'] = text

        gt_data_dict_list.append(data_dict_gt)
        decoded_data_dict_list.append(data_dict_decoded)

        with open(os.path.join(OUTPUT_DIR, f"g1ml3d_{idx}_gt.pkl"), 'wb') as f:
            pickle.dump(data_dict_gt, f)
        with open(os.path.join(OUTPUT_DIR, f"g1ml3d_{idx}_decoded.pkl"), 'wb') as f:
            pickle.dump(data_dict_decoded, f)


gt_data_dict = merge_data_list(gt_data_dict_list)
decoded_data_dict = merge_data_list(decoded_data_dict_list)

with open(os.path.join(OUTPUT_DIR, "g1ml3d_gt.pkl"), 'wb') as f:
    pickle.dump(gt_data_dict, f)
with open(os.path.join(OUTPUT_DIR, "g1ml3d_decoded.pkl"), 'wb') as f:
    pickle.dump(decoded_data_dict, f)




