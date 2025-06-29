from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm import ROOT

from collections import OrderedDict
from HRI_mllm.utils.motion_utils.g1ml3d import feats2joints
import yaml
import os
import torch
import numpy as np
import sys
import pickle
from HRI_mllm.datasets.G1ML3D import G1ML3DDataModule
from HRI_mllm.utils.motion_utils.metrics import calc_mpjpe, calc_pampjpe

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.g1_29_humanml3d_representation import data_pkl_to_vec
from HRI_mllm import DATA_ROOT

motion_name = "000093"
npy_file = os.path.join(DATA_ROOT, "G1ML3D_v1", "new_joint_vecs", f"{motion_name}.npy")
pkl_file = os.path.join(DATA_ROOT, "G1ML3D_v1", "joints", f"{motion_name}.pickle")


np_vec = np.load(npy_file, allow_pickle=True)
with open(pkl_file, 'rb') as f:
    pkl_data = pickle.load(f)


vec = data_pkl_to_vec(pkl_data)

assert np.allclose(np_vec, vec), "The numpy vector and the vector from pickle data do not match."

data = vec_to_data_pkl(torch.from_numpy(np_vec).unsqueeze(0))

