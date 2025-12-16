### encoding: utf-8
### encoding and decoding of G1 280 dimension feature with flexible statistics
import torch 
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.brainco_representation import vec_to_data_pkl, vec_to_joints, hparams_mean, hparams_std
from HRI_mllm import DATA_ROOT
import numpy as np
import os


def normalize_vec(vec, mean=None, std=None):
    """
    Normalize the features using the mean and standard deviation.
    :param vec: Tensor of shape (batch_size, num_frames, num_features)
    :param mean: Mean array (optional, defaults to hparams_mean)
    :param std: Std array (optional, defaults to hparams_std)
    :return: Normalized features
    """
    if mean is None:
        mean = hparams_mean
    if std is None:
        std = hparams_std
    
    mean_t = torch.tensor(mean, dtype=torch.float32).to(vec)
    std_t = torch.tensor(std, dtype=torch.float32).to(vec)
    normalized_features = (vec - mean_t) / std_t
    return normalized_features

def denormalize_vec(vec, mean=None, std=None):
    """
    Denormalize the features.
    :param vec: Normalized tensor
    :param mean: Mean array (optional, defaults to hparams_mean)
    :param std: Std array (optional, defaults to hparams_std)
    :return: Denormalized features
    """
    if mean is None:
        mean = hparams_mean
    if std is None:
        std = hparams_std
    
    mean_t = torch.tensor(mean, dtype=torch.float32).to(vec)
    std_t = torch.tensor(std, dtype=torch.float32).to(vec)
    denormalized_features = vec * std_t + mean_t
    return denormalized_features

def feats2joints(features, mean=None, std=None):
    """
    Convert normalized features to joints.
    :param features: Normalized features
    :param mean: Mean array (optional, defaults to hparams_mean)
    :param std: Std array (optional, defaults to hparams_std)
    """
    vec = denormalize_vec(features, mean, std)
    return vec_to_joints(vec)

def feats2datapkl(features, mean=None, std=None):
    """
    Convert normalized features to data pickle.
    :param features: Normalized features
    :param mean: Mean array (optional, defaults to hparams_mean)
    :param std: Std array (optional, defaults to hparams_std)
    """
    assert features.dim() == 3, "Input features must be a 3D tensor (batch_size, frame, num_features)"
    assert features.shape[0] == 1, "Batch size must be 1 for feats2datapkl"
    
    vec = denormalize_vec(features, mean, std)
    return vec_to_data_pkl(vec.squeeze(0).detach().cpu().numpy())


def load_normalization_stats(config):
    """
    Load normalization statistics based on config.
    Returns (mean, std) arrays.
    """
    use_mixed_stats = config.get("use_mixed_stats", True)
    
    if use_mixed_stats:
        # 🔧 优先使用Fixed版本（修复了std接近0的问题）
        mixed_stats_paths = [
            os.path.join(DATA_ROOT, "Mixed_Statistics_Equal_Fixed"),  # 优先
            os.path.join(DATA_ROOT, "Mixed_Statistics_Equal"),
            os.path.join(DATA_ROOT, "Mixed_Statistics"),
        ]
        for path in mixed_stats_paths:
            mean_path = os.path.join(path, "Mean.npy")
            if os.path.exists(mean_path):
                mean = np.load(mean_path)
                std = np.load(os.path.join(path, "Std.npy"))
                print(f"✅ 使用混合统计量: {path}")
                return mean, std
        
        # Fallback to BEAT
        print(f"⚠️  混合统计量不存在，使用BEAT统计量")
    
    # Use BEAT statistics
    mean_path = os.path.join(DATA_ROOT, "BEAT_v2_kimi", "Mean.npy")
    mean = np.load(mean_path)
    std = np.load(os.path.join(DATA_ROOT, "BEAT_v2_kimi", "Std.npy"))
    print(f"使用BEAT统计量: {os.path.join(DATA_ROOT, 'BEAT_v2_kimi')}")
    return mean, std

