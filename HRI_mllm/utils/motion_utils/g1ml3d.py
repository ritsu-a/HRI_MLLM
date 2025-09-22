### encoding: utf-8
### encoding and decoding of G1 280 dimension feature
import torch 
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.inspirehand_representation import vec_to_data_pkl
from HRI_retarget.utils.torch_utils.diff_quat import quat_to_matrix
from HRI_retarget.model.g1_29 import G1_29_Motion_Model
import numpy as np
import os
from HRI_mllm import DATA_ROOT

from .skeleton import Skeleton
import numpy as np
import os
from .quaternion import *
from .paramUtil import *

hparams_mean = np.load(os.path.join("{DATA_ROOT}/BEAT_v1_kimi", "Mean.npy"))
hparams_std = np.load(os.path.join("{DATA_ROOT}/BEAT_v1_kimi", "Std.npy"))

def normalize_vec(vec):
    """
    Normalize the features using the mean and standard deviation.
    :param features: Tensor of shape (batch_size, num_frames, num_features)
    :return: Normalized features
    """
    mean = torch.tensor(hparams_mean).to(vec)
    std = torch.tensor(hparams_std).to(vec)
    normalized_features = (vec - mean) / std
    return normalized_features

def feats2joints(features):
    mean = torch.tensor(hparams_mean).to(features)
    std = torch.tensor(hparams_std).to(features)
    vec = features * std + mean
    return recover_from_ric(vec)

def feats2datapkl(features):
    assert features.dim() == 3, "Input features must be a 3D tensor (batch_size, frame, num_features)"
    assert features.shape[0] == 1, "Batch size must be 1 for feats2datapkl"
    mean = torch.tensor(hparams_mean).to(features)
    std = torch.tensor(hparams_std).to(features)
    vec = features * std + mean
    return vec_to_data_pkl(vec.squeeze(0).detach().cpu().numpy())


def recover_from_ric(body_vec):
    assert body_vec.dim() == 3, "Input data must be a 3D tensor (batch_size, frame, num_features)"
    assert isinstance(body_vec, torch.Tensor), f"body_vec 应该是 torch.tensor，但实际类型是 {type(body_vec)}"


    joints_num = 29 
    links_num = 41
    device = body_vec.device
    batch_size = body_vec.shape[0]
    num_frames = body_vec.shape[1]

    total_frames = batch_size * num_frames


    body_vec_expanded = body_vec.view(total_frames, -1)
    dof_angles = torch.zeros((total_frames, 29))
    dof_angles[:, 12:22] = body_vec_expanded[:, 162:172]  # waist and left arm
    dof_angles[:, 22:29] = body_vec_expanded[:, 172:179]  # right arm

    


    model = G1_29_Motion_Model(total_frames, device=device)
    model.set_angles(torch.tensor(dof_angles))

    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3] 
    positions = link_to_root_pos.view(batch_size, num_frames, -1, 3)

    return positions


