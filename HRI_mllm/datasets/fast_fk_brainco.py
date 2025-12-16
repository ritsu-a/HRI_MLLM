"""
快速前向运动学（FK）计算模块
专门用于推理场景，不使用梯度计算，大幅提升速度
"""
import numpy as np
import torch
from scipy import interpolate
import sys
import os

# 添加HRI_retarget到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../external/HRI_retarget'))

from HRI_retarget.config.joint_mapping import G1_BRAINCO_LINKS, G1_LINKS, G1_BRAINCO_LEFT_HAND_LINKS, G1_BRAINCO_RIGHT_HAND_LINKS
from HRI_retarget.model.g1_brainco import G1_Brainco_Motion_Model

target_fps = 25


def fast_data_pkl_to_vec_batch(data_dict_list):
    """
    批量处理多个样本，大幅提升GPU利用率
    
    Args:
        data_dict_list: list of data_dict, 每个data_dict包含:
            - angles: numpy array, shape (num_frames, 53)
            - robot_name: str, 默认 "g1_brainco"
            - fps: int, 帧率
    
    Returns:
        list of vec: 每个vec shape (num_frame - 1, 491)
    """
    if len(data_dict_list) == 0:
        return []
    
    # 按帧数分组，相同帧数的样本一起处理
    from collections import defaultdict
    grouped_by_frames = defaultdict(list)
    
    for idx, data_dict in enumerate(data_dict_list):
        angles = np.unwrap(data_dict["angles"])
        num_frames = angles.shape[0]
        source_fps = data_dict.get("fps", 60)
        
        # 计算目标帧数（考虑重采样）
        if source_fps != target_fps:
            num_target_frames = int(num_frames * target_fps / source_fps)
        else:
            num_target_frames = num_frames
        
        grouped_by_frames[num_target_frames].append((idx, data_dict))
    
    # 存储结果
    results = [None] * len(data_dict_list)
    
    # 对每组相同帧数的样本进行批量处理
    for num_frames, group in grouped_by_frames.items():
        if len(group) == 1:
            # 单个样本，直接处理
            idx, data_dict = group[0]
            results[idx] = fast_data_pkl_to_vec(data_dict)
        else:
            # 多个样本，批量处理
            batch_results = _process_batch_same_length(group, num_frames)
            for (idx, _), result in zip(group, batch_results):
                results[idx] = result
    
    return results


def _process_batch_same_length(group, num_frames):
    """
    处理相同长度的样本批次
    
    Args:
        group: list of (idx, data_dict)
        num_frames: int, 所有样本的帧数
    
    Returns:
        list of vec arrays
    """
    # 预处理所有样本
    angles_list = []
    for idx, data_dict in group:
        angles = np.unwrap(data_dict["angles"])
        source_fps = data_dict.get("fps", 60)
        
        # 重采样
        if source_fps != target_fps:
            num_target_frames = int(angles.shape[0] * target_fps / source_fps)
            interpolated_data = np.zeros((num_target_frames, angles.shape[1]))
            t_original = np.linspace(0, angles.shape[0] - 1, angles.shape[0]) 
            t_target = np.linspace(0, angles.shape[0] - 1, num_target_frames)
            
            for dof_idx in range(angles.shape[1]):
                y_original = angles[:, dof_idx]
                f = interpolate.interp1d(t_original, y_original, kind='cubic', fill_value='extrapolate')
                y_interpolated = f(t_target)
                interpolated_data[:, dof_idx] = y_interpolated
            angles = interpolated_data
        
        angles_list.append(angles)
    
    # 堆叠成批次 (batch_size, num_frames, 53)
    batch_angles = np.stack(angles_list, axis=0)
    batch_size = batch_angles.shape[0]
    
    # 创建模型（批量大小 = 批次中样本数 * 每个样本的帧数）
    total_frames = batch_size * num_frames
    model = G1_Brainco_Motion_Model(total_frames)
    model.eval()
    
    # 转换为tensor并处理
    batch_angles_tensor = torch.from_numpy(batch_angles).float()  # (batch_size, num_frames, 53)
    batch_angles_tensor[:, :, :12] *= 0  # set unused dofs to zero
    batch_angles_tensor = batch_angles_tensor.detach().requires_grad_(False)
    
    # 重塑为 (batch_size * num_frames, 53) 用于模型处理
    device = getattr(model, "device", torch.device("cpu"))
    angles_flat = batch_angles_tensor.view(total_frames, 53).to(device)
    model.set_angles(angles_flat)
    
    # 批量计算FK
    with torch.no_grad():
        link_to_root_dict = fast_forward_kinematics_split(model, angles_flat)
        link_to_root_pos = link_to_root_dict[:, :, :3, 3]  # (total_frames, 75, 3)
        local_positions = link_to_root_pos.cpu().numpy()  # (total_frames, 75, 3)
    
    # 重塑回批次形状
    local_positions = local_positions.reshape(batch_size, num_frames, 75, 3)
    rot_data = batch_angles_tensor.cpu().numpy()  # (batch_size, num_frames, 53)
    
    # 对每个样本分别处理
    results = []
    for i in range(batch_size):
        local_pos = local_positions[i]  # (num_frames, 75, 3)
        rot = rot_data[i]  # (num_frames, 53)
        
        # 计算速度
        local_vel = local_pos[1:] - local_pos[:-1]  # (num_frames-1, 75, 3)
        
        # 构建特征向量
        body_vec = np.concatenate([
            local_pos[:-1, :41, :].reshape(num_frames-1, -1),  # body_link
            local_vel[:, :41, :].reshape(num_frames-1, -1),  # body_vel
            rot[:-1, 12:22], # waist and left arm
            rot[:-1, 34:41], # right arm
        ], axis=-1)
        
        hand_vec = np.concatenate([
            local_pos[:-1, 41:58, :].reshape(num_frames-1, -1),  # left_hand_link
            local_pos[:-1, 58:75, :].reshape(num_frames-1, -1),  # right_hand_link
            local_vel[:, 41:58, :].reshape(num_frames-1, -1),  # left_hand_vel
            local_vel[:, 58:75, :].reshape(num_frames-1, -1),  # right_hand_vel
            rot[:-1, 22:34], # left hand
            rot[:-1, 41:53], # right hand
        ], axis=-1)
        
        vec = np.concatenate([body_vec, hand_vec], axis=-1)  # (num_frame - 1, 491)
        results.append(vec)
    
    return results


def fast_data_pkl_to_vec(data_dict):
    """
    快速版本的 data_pkl_to_vec，专门用于推理场景
    不使用梯度计算，大幅提升速度
    
    Convert the data_dict to vec representation (optimized for inference)
    :param data_dict: dict, contains the data
    :return: vec, the vec representation
    """
    angles = np.unwrap(data_dict["angles"])
    num_frames = angles.shape[0]
    source_fps = data_dict["fps"]

    ### resample if fps is not equal
    if source_fps != target_fps:
        print(f"source fps {source_fps} is not equal to target fps {target_fps}, resampling...")
        num_target_frames = int(num_frames * target_fps / source_fps)
        interpolated_data = np.zeros((num_target_frames, angles.shape[1]))
        t_original = np.linspace(0, num_frames - 1, num_frames) 
        t_target = np.linspace(0, num_frames - 1, num_target_frames)     

        # 对每个自由度（每一列）进行插值
        for dof_idx in range(angles.shape[1]):
            y_original = angles[:, dof_idx]
            f = interpolate.interp1d(t_original, y_original, kind='cubic', fill_value='extrapolate')
            y_interpolated = f(t_target)
            interpolated_data[:, dof_idx] = y_interpolated
        angles = interpolated_data
        num_frames = angles.shape[0]

    match data_dict["robot_name"]:
        case "g1_brainco":
            model = G1_Brainco_Motion_Model(num_frames)
            robot_link = G1_BRAINCO_LINKS
        case _:
            print("wrong robot name in kinematic vis")
            quit()

    # 设置为推理模式，禁用梯度计算以加速
    model.eval()
    
    ### set unused dofs to zero
    device = getattr(model, "device", torch.device("cpu"))
    angles = torch.from_numpy(angles).float().to(device)
    angles[:, :12] *= 0
    
    # 使用 detach() 和 requires_grad=False 来避免梯度计算
    angles = angles.detach().requires_grad_(False)
    model.set_angles(angles)
  
    # 使用 torch.no_grad() 禁用梯度计算，大幅提升速度
    with torch.no_grad():
        link_to_root_dict = fast_forward_kinematics_split(model, angles)
        link_to_root_pos = link_to_root_dict[:, :, :3, 3]
        local_positions = link_to_root_pos.cpu().numpy()

    '''Get Joint Rotation Representation'''
    # (seq_len, dof) dof for skeleton joints
    rot_data = angles.detach().cpu().numpy()

    '''Get Joint Velocity Representation'''
    # (seq_len-1, (link-1)*3)
    local_vel = local_positions[1:] - local_positions[:-1]

    body_vec = np.concatenate([
        local_positions[:-1, :41, :].reshape(num_frames-1, -1),  # body_link
        local_vel[:, :41, :].reshape(num_frames-1, -1),  # body_vel
        rot_data[:-1, 12:22], # waist and left arm
        rot_data[:-1, 34:41], # right arm
    ], axis=-1)

    hand_vec = np.concatenate([
        local_positions[:-1, 41:58, :].reshape(num_frames-1, -1),  # left_hand_link
        local_positions[:-1, 58:75, :].reshape(num_frames-1, -1),  # right_hand_link
        local_vel[:, 41:58, :].reshape(num_frames-1, -1),  # left_hand_vel
        local_vel[:, 58:75, :].reshape(num_frames-1, -1),  # right_hand_vel
        rot_data[:-1, 22:34], # left hand
        rot_data[:-1, 41:53], # right hand
    ], axis=-1)                                     

    return np.concatenate([body_vec, hand_vec], axis=-1)  # (num_frame - 1, 263 + 228 = 491)


def fast_forward_kinematics_split(model, joint_angles):
    """
    快速版本的 forward_kinematics_split，不使用梯度计算
    直接使用传入的角度值，避免从模型中获取（可能触发梯度计算）
    
    Args:
        model: G1_Brainco_Motion_Model 实例
        joint_angles: torch.Tensor, shape (num_frames, 53), 不需要梯度的角度值
    """
    # 确保角度值不需要梯度
    joint_angles = joint_angles.detach()
    
    # 使用 torch.no_grad() 确保在无梯度模式下运行
    with torch.no_grad():
        body_dict = model.body_chain.forward_kinematics(
            torch.cat([joint_angles[:, :22], joint_angles[:, 34:41]], dim=1)
        )
        left_hand_dict = model.left_hand_chain.forward_kinematics(
            torch.cat([joint_angles[:, 22:25], joint_angles[:, 26:34]], dim=1)
        )
        right_hand_dict = model.right_hand_chain.forward_kinematics(
            torch.cat([joint_angles[:, 41:44], joint_angles[:, 45:53]], dim=1)
        )
        
        link_to_root_dict = []

        for link_name in G1_LINKS:
            link_to_root_dict.append(body_dict[link_name].get_matrix())
        for link_name in G1_BRAINCO_LEFT_HAND_LINKS:
            link_to_root_dict.append(left_hand_dict[link_name].get_matrix())
        for link_name in G1_BRAINCO_RIGHT_HAND_LINKS:   
            link_to_root_dict.append(right_hand_dict[link_name].get_matrix())

        link_to_root_dict = torch.stack(link_to_root_dict, dim=1)  # (N_frame, 41+17+17=75, 4, 4)

    return link_to_root_dict

