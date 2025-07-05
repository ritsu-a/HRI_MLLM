# 2025.07.05 HIT-xiaowangzi
# beat的辅助函数 我读一下有什么需要修改的地方

### encoding: utf-8
### encoding and decoding of G1_inspirehands 502 dimension feature
import torch 
from HRI_retarget.utils.torch_utils.diff_quat import quat_to_matrix
from HRI_retarget.model.g1_inspirehands import G1_Inspirehands_Motion_Model
import numpy as np
import os

from .skeleton import Skeleton
import numpy as np
import os
from .quaternion import *
from .paramUtil import *

hparams_mean = np.load(os.path.join("/root/pengyang/codebase/HRI_MLLM/data/G1_beat", "Mean.npy"))
hparams_std = np.load(os.path.join("/root/pengyang/codebase/HRI_MLLM/data/G1_beat", "Std.npy"))

def feats2joints(features):
    mean = torch.tensor(hparams_mean).to(features)
    std = torch.tensor(hparams_std).to(features)
    features = features * std + mean
    return recover_from_ric(features)

def feats2datapkl(features):
    mean = torch.tensor(hparams_mean).to(features)
    std = torch.tensor(hparams_std).to(features)
    features = features * std + mean
    return vec_to_data_pkl(features)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def recover_root_rot_pos(data):
    assert data.dim() == 3, "Input data must be a 3D tensor (batch_size, frame, num_features)"
    batch_size = data.shape[0]
    r_velocity = data[:, :, 0:4]  
    l_velocity = data[:, :, 4:6]  # XY速度
    root_z = data[:, :, 6:7]       # 高度

    
    # 还原translation
    restored_translation = torch.zeros((batch_size, data.shape[1]+1, 3)).to(data.device)
    restored_translation[:, 0, 2] = root_z[:, 0 ,0]
    restored_translation[:, 1:, 0:2] = torch.cumsum(l_velocity, axis=1) + restored_translation[:, :1, 0:2]
    restored_translation[:, 1:, 2] = root_z[:, :, 0]
    
    # 还原rotation
    restored_rotation = torch.zeros((batch_size, data.shape[1]+1, 4)).to(data.device)
    restored_rotation[:, 0, :] = torch.from_numpy(np.array([0, 0, 0, 1])).expand(batch_size, 4)  # 初始四元数 (w, x, y, z)
    

    for i in range(1, restored_rotation.shape[1]):
        delta_q = r_velocity[:, i-1]
        restored_rotation[:, i] = qmul(delta_q, restored_rotation[:, i-1])
    
    return restored_rotation[:, 1:], restored_translation[:, 1:]

def recover_from_ric(data):
    assert data.dim() == 3, "Input data must be a 3D tensor (batch_size, frame, num_features)"


    global_rotations_quat, global_positions = recover_root_rot_pos(data)

    device = data.device

    joints_num = 53
    links_num = 74
    batch_size = data.shape[0]
    num_frames = data.shape[1]

     
    global_positions = global_positions.reshape(-1, 3, 1)
    global_rotations = quat_to_matrix(global_rotations_quat)[..., :3, :2].reshape(-1, 3, 2)


    dof_angles = data[..., 7 + (links_num - 1) * 3: 7 + (links_num - 1) * 3 + joints_num].reshape(-1, joints_num)

    

    data_dict = {
        "angles": dof_angles,
        "global_rotation": global_rotations,
        "global_translation": global_positions,
        "scale":torch.ones(3).to(device),
    }

    # 折腾一大圈 计算出fk 每个关节点的位置
    model = G1_Inspirehands_Motion_Model(batch_size * num_frames, device=device)
    model.set_angles(dof_angles)
    model.set_global_matrix(data_dict)
    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3] 
    positions = link_to_root_pos.view(batch_size, num_frames, -1, 3)


    return positions

def vec_to_data_pkl(data, fps=20, reference_motion_pth=None, robot_name="g1_inspirehands", scale=np.ones(3)):
    """
    Convert the vec representation to data_dict
    :param vec: vec, the vec representation
    :return: data_dict, the data_dict
    """
    assert data.dim() == 3, "Input data must be a 3D tensor (batch_size, frame, num_features)"
    assert data.shape[0] == 1, "Input data must have batch size of 1"


    global_rotations_quat, global_positions = recover_root_rot_pos(data)

    device = data.device


    joints_num = 53 
    links_num = 74
    batch_size = data.shape[0]
    num_frames = data.shape[1]

     
    global_positions = global_positions.reshape(-1, 3, 1)
    global_rotations = quat_to_matrix(global_rotations_quat)[..., :3, :2].reshape(-1, 3, 2)


    dof_angles = data[..., 7 + (links_num - 1) * 3: 7 + (links_num - 1) * 3 + joints_num].reshape(-1, joints_num)


    data_dict = {
        "fps": fps,
        "reference_motion_pth": reference_motion_pth,
        "robot_name": robot_name,
        "angles": dof_angles,
        "global_rotation": global_rotations,
        "global_translation": global_positions,
        "scale": scale,
    }



    ### rot_data = data_dict["angles"]
    return data_dict
