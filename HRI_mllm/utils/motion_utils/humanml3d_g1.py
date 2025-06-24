### encoding: utf-8
### encoding and decoding of G1 280 dimension feature
import torch 
from HRI_retarget.utils.torch_utils.diff_quat import quat_to_matrix
from HRI_retarget.model.g1_29 import G1_29_Motion_Model
import numpy as np
import os

hparams_mean = np.load(os.path.join("/data0/data/G1ML3D", "Mean.npy"))
hparams_std = np.load(os.path.join("/data0/data/G1ML3D", "Std.npy"))

def feats2joints(features):
    mean = torch.tensor(hparams_mean).to(features)
    std = torch.tensor(hparams_std).to(features)
    features = features * std + mean
    return recover_from_ric(features)

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
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data):

    r_rot_quat, r_pos = recover_root_rot_pos(data)

    device = data.device

    rot = torch.tensor([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float).to(device)


    joints_num = 29 
    links_num = 41
    batch_size = data.shape[0]
    num_frames = data.shape[1]

    global_positions = r_pos.reshape(-1, 3, 1)
    global_rotations = quat_to_matrix(r_rot_quat)[..., :3, :2].reshape(-1, 3, 2)
    data_dict = {
        "global_rotation": global_rotations,
        "global_translation": global_positions,
        "scale": torch.ones(3).to(device),
    }


    dof_angles = data[..., 4 + (links_num - 1) * 3: 4 + (links_num - 1) * 3 + joints_num].reshape(-1, joints_num)
    model = G1_29_Motion_Model(batch_size * num_frames, device=device)
    model.set_angles(dof_angles)
    model.set_global_matrix(data_dict)
    link_to_root_dict = model.forward_kinematics()
    link_to_root_pos = link_to_root_dict[:, :, :3, 3] @ rot
    positions = link_to_root_pos.view(batch_size, num_frames, -1, 3)


    return positions