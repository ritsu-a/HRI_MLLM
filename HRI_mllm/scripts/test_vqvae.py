from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm import ROOT

from collections import OrderedDict
from HRI_mllm.utils.motion_utils.humanml3d_g1 import feats2joints

import yaml
import os
import torch
import sys

import pickle




def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data        

motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "vqvae.yaml"))
import ipdb;ipdb.set_trace()


motion_vae = VQVae(**motion_config)



state_dict = torch.load(motion_config["ckpt"], map_location="cpu", weights_only=False)

motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")


motion_repre = motion_vae.decode(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to("cuda"))




joints = feats2joints(motion_repre)

ipdb.set_trace()