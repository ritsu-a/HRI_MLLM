### encoding: utf-8
### encoding and decoding of G1 280 dimension feature
import torch 
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.brainco_representation import vec_to_data_pkl, vec_to_joints, hparams_mean, hparams_std

import numpy as np
import os



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
    return vec_to_joints(vec)

def feats2datapkl(features):
    assert features.dim() == 3, "Input features must be a 3D tensor (batch_size, frame, num_features)"
    assert features.shape[0] == 1, "Batch size must be 1 for feats2datapkl"
    mean = torch.tensor(hparams_mean).to(features)
    std = torch.tensor(hparams_std).to(features)
    vec = features * std + mean
    return vec_to_data_pkl(vec.squeeze(0).detach().cpu().numpy())




