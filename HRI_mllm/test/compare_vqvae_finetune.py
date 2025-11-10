"""
比较finetune前后VQ-VAE模型的效果

功能：
1. 加载finetune前后的两个模型
2. 在BEAT、seg_finger、single_motion三个数据集上测试
3. 计算重建loss、量化loss、困惑度等指标
4. 可视化结果并生成对比视频
5. 生成对比报告

特点：
- 整段测试：不使用窗口切分，直接处理完整序列
- 自动长度对齐：如果模型输出长度与输入不一致，使用线性插值恢复
- 最大长度限制：防止内存溢出（BEAT: 2048帧，seg_finger: 512帧，single_motion: 2048帧）

使用方法：
    python HRI_mllm/test/compare_vqvae_finetune.py \
      --pretrained_ckpt output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt \
      --finetuned_ckpt output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt \
      --config g1_vqvae_arbitrary_length_balanced.yaml \
      --num_samples 5
"""
import os
import argparse
import random
import subprocess
from pathlib import Path
from tqdm import tqdm

os.environ['MUJOCO_GL'] = 'egl'
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
import yaml
import torch
import pickle
import numpy as np
from scipy import interpolate
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import (
    feats2datapkl, load_normalization_stats
)
import torch.nn.functional as F
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion


def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(checkpoint_path, config_path, device='cuda'):
    """加载VQ-VAE模型"""
    motion_config = open_yaml(config_path)
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device=device)
    return motion_vae, motion_config


def evaluate_model(model, motion_data, mean, std, device='cuda', max_length=None):
    """
    评估模型在给定motion数据上的性能（整段测试，不使用窗口）
    
    Args:
        model: VQ-VAE模型
        motion_data: numpy array, shape [T, F]，原始未归一化的数据
        mean: 均值数组
        std: 标准差数组
        device: 设备
        max_length: 最大长度限制（None表示不限制，防止内存溢出）
    
    Returns:
        metrics: 包含重建loss、量化loss、困惑度等指标的字典
        reconstructed_motion: 重建的motion数据（未归一化，与原始数据长度一致）
    """
    model.eval()
    mean_t = torch.tensor(mean, dtype=torch.float32).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).to(device)
    
    # 🔧 整段测试：使用完整序列，不进行窗口切分
    original_length = motion_data.shape[0]
    original_motion_data = motion_data.copy()
    
    # 如果设置了最大长度限制，且序列超过最大长度，进行截断
    if max_length is not None and original_length > max_length:
        print(f"⚠️  Sequence length {original_length} exceeds max_length {max_length}, truncating...")
        motion_data = motion_data[:max_length]
        actual_length = max_length
    else:
        actual_length = original_length
    
    # 确保序列长度至少为最小长度（模型要求，通常为8帧）
    min_required_length = 8
    if actual_length < min_required_length:
        # 如果太短，padding到最小长度（但会在loss计算时忽略padding部分）
        padding = np.zeros((min_required_length - actual_length, motion_data.shape[1]), dtype=motion_data.dtype)
        motion_data_padded = np.concatenate([motion_data, padding], axis=0)
        has_padding = True
    else:
        motion_data_padded = motion_data
        has_padding = False
    
    # 归一化
    data_tensor = torch.from_numpy(motion_data_padded).unsqueeze(0).to(device).float()
    normalized_input = (data_tensor - mean_t) / std_t
    input_length = normalized_input.shape[1]
    
    with torch.no_grad():
        # 前向传播
        decoded_features, quant_loss, perplexity = model(normalized_input)
        output_length = decoded_features.shape[1]
        
        # 🔧 处理输入输出长度不匹配
        # Decoder使用Upsample上采样，理论上输出长度应该与输入长度一致
        # 但由于padding等原因可能会有细微差异（通常差异很小，1-2帧）
        # 我们取最小长度以确保对齐
        min_length = min(input_length, output_length)
        
        # 只在有效帧上计算loss（如果有padding，忽略padding部分）
        if has_padding:
            # 创建mask，只计算有效帧的loss
            mask = torch.zeros(min_length, device=device)
            mask[:actual_length] = 1.0
            mask = mask.unsqueeze(0).unsqueeze(2)  # [1, T, 1]
            
            input_for_loss = normalized_input[:, :min_length, :]
            output_for_loss = decoded_features[:, :min_length, :]
            
            recon_loss = F.mse_loss(output_for_loss * mask, input_for_loss * mask, reduction='sum')
            recon_loss = recon_loss / (mask.sum() + 1e-8)
        else:
            # 没有padding，直接计算loss（只在有效长度上）
            input_for_loss = normalized_input[:, :min_length, :]
            output_for_loss = decoded_features[:, :min_length, :]
            recon_loss = F.mse_loss(output_for_loss, input_for_loss)
        
        # 反归一化重建结果
        reconstructed_motion = (output_for_loss * std_t + mean_t).cpu().numpy()[0]
        
        # 🔧 处理长度对齐：确保重建结果长度与actual_length一致
        # 由于模型的下采样和上采样，输出长度可能与输入长度有细微差异
        # 我们使用线性插值或截断来确保长度一致
        if reconstructed_motion.shape[0] != actual_length:
            if reconstructed_motion.shape[0] < actual_length:
                # 如果重建结果较短，使用线性插值扩展到actual_length
                if reconstructed_motion.shape[0] > 1:
                    original_indices = np.linspace(0, reconstructed_motion.shape[0] - 1, reconstructed_motion.shape[0])
                    target_indices = np.linspace(0, reconstructed_motion.shape[0] - 1, actual_length)
                    
                    reconstructed_motion_interp = np.zeros((actual_length, reconstructed_motion.shape[1]))
                    for i in range(reconstructed_motion.shape[1]):
                        f = interpolate.interp1d(original_indices, reconstructed_motion[:, i], kind='linear', 
                                                bounds_error=False, fill_value='extrapolate')
                        reconstructed_motion_interp[:, i] = f(target_indices)
                    reconstructed_motion = reconstructed_motion_interp
                else:
                    # 如果只有1帧，直接复制
                    reconstructed_motion = np.tile(reconstructed_motion, (actual_length, 1))
            else:
                # 如果重建结果较长，截断到actual_length
                reconstructed_motion = reconstructed_motion[:actual_length]
        
        # 最终重建结果长度应该等于actual_length
        assert reconstructed_motion.shape[0] == actual_length, \
            f"重建结果长度 {reconstructed_motion.shape[0]} 与期望长度 {actual_length} 不一致"
    
    metrics = {
        'recon_loss': recon_loss.item(),
        'quant_loss': quant_loss.item(),
        'perplexity': perplexity.item(),
        'actual_frames': actual_length,
        'original_frames': original_length,
        'input_frames': input_length,
        'output_frames': output_length,
    }
    
    return metrics, reconstructed_motion


def get_dataset_files(dataset_name, num_samples=5):
    """
    获取数据集文件列表
    
    Args:
        dataset_name: 数据集名称 ('beat', 'seg_finger', 'single_motion')
        num_samples: 要采样的文件数量
    
    Returns:
        file_paths: 文件路径列表
    """
    if dataset_name == 'beat':
        data_dir = os.path.join(DATA_ROOT, "BEAT_v2_kimi", "new_joint_vecs")
        # 从train.txt读取文件列表
        train_txt = os.path.join(DATA_ROOT, "BEAT_v2_kimi", "train.txt")
        with open(train_txt, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]
        file_paths = [os.path.join(data_dir, f"{name}.npy") for name in file_names]
    elif dataset_name == 'seg_finger':
        data_dir = os.path.join(DATA_ROOT, "seg_finger", "new_joint_vecs")
        # 直接扫描目录
        file_names = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        file_paths = [os.path.join(data_dir, f) for f in file_names]
    elif dataset_name == 'single_motion':
        data_dir = "/root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_joint_vecs/npy"
        file_names = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        file_paths = [os.path.join(data_dir, f) for f in file_names]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 随机采样
    if len(file_paths) > num_samples:
        file_paths = random.sample(file_paths, num_samples)
    
    return file_paths


def concat_side_by_side(video_left, video_right, output_path, target_height=720):
    """使用 ffmpeg 横向拼接两个视频"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_left,
        "-i", video_right,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1[left];[1:v]scale=-2:{target_height},setsar=1[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def concat_three_videos(video_left, video_middle, video_right, output_path, target_height=720):
    """
    使用 ffmpeg 横向拼接三个视频（GT | Baseline | Finetune），并添加文字注释
    
    Args:
        video_left: GT视频路径
        video_middle: Baseline视频路径
        video_right: Finetune视频路径
        output_path: 输出视频路径
        target_height: 目标高度
    """
    # 字体大小和位置设置
    font_size = max(24, target_height // 30)  # 根据视频高度自适应字体大小
    text_x = 20  # 文字X位置（距离左边）
    text_y = 40  # 文字Y位置（距离顶部）
    
    # 使用 drawtext 滤镜添加文字标签
    # 注意：在filter_complex中，text参数中的单引号可能需要转义或使用双引号
    # box=1:boxcolor=black@0.5 添加半透明黑色背景，提高文字可读性
    # boxborderw=5 设置边框宽度
    # fontcolor=white 白色文字
    
    # 构建filter_complex字符串
    # 使用转义的单引号来包围文本，确保特殊字符被正确处理
    filter_complex = (
        f"[0:v]scale=-2:{target_height},setsar=1,"
        f"drawtext=text='GT':fontsize={font_size}:fontcolor=white:"
        f"x={text_x}:y={text_y}:box=1:boxcolor=black@0.5:boxborderw=5[left];"
        f"[1:v]scale=-2:{target_height},setsar=1,"
        f"drawtext=text='Baseline':fontsize={font_size}:fontcolor=white:"
        f"x={text_x}:y={text_y}:box=1:boxcolor=black@0.5:boxborderw=5[middle];"
        f"[2:v]scale=-2:{target_height},setsar=1,"
        f"drawtext=text='Finetune':fontsize={font_size}:fontcolor=white:"
        f"x={text_x}:y={text_y}:box=1:boxcolor=black@0.5:boxborderw=5[right];"
        f"[left][middle]hstack=inputs=2[tmp];[tmp][right]hstack=inputs=2[v]"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_left,
        "-i", video_middle,
        "-i", video_right,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def visualize_motion(motion_data, output_path, mean, std, device='cuda'):
    """
    可视化motion数据
    
    Args:
        motion_data: numpy array, shape [T, F]，应该是未归一化的原始数据
        output_path: 输出CSV文件路径
        mean: 均值数组
        std: 标准差数组
        device: 设备
    """
    # motion_data应该是未归一化的原始数据
    # 转换为tensor并归一化
    motion_tensor = torch.from_numpy(motion_data).unsqueeze(0).to(device).float()
    mean_t = torch.tensor(mean, dtype=torch.float32).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).to(device)
    
    # 归一化（feats2datapkl需要归一化的输入）
    normalized_motion = (motion_tensor - mean_t) / std_t
    
    # 转换为pkl（feats2datapkl内部会反归一化）
    motion_pkl = feats2datapkl(normalized_motion, mean=mean, std=std)
    
    # 保存pkl
    pkl_path = output_path.replace('.csv', '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(motion_pkl, f)
    
    # 转换为csv
    motion_csv = load_motion_pkl_as_csv_data(pkl_path)
    csv_path = output_path
    np.savetxt(csv_path, motion_csv, delimiter=',', fmt='%.8f')
    
    # 生成视频
    video_path = output_path.replace('.csv', '.mp4')
    vis_audio_motion(
        csv_path,
        output_path=video_path,
        audio_path=None,
        robot_type="g1_brainco",
        rate_limit=False,
        motion_fps=25
    )
    
    return video_path


def compare_models(pretrained_model, finetuned_model, pretrained_config, finetuned_config,
                   dataset_name, file_paths, mean, std, device='cuda', output_dir='comparison_results'):
    """
    比较两个模型在同一数据集上的表现（整段测试，不使用窗口）
    
    Args:
        pretrained_model: 预训练模型
        finetuned_model: finetune后的模型
        pretrained_config: 预训练模型配置
        finetuned_config: finetune模型配置
        dataset_name: 数据集名称
        file_paths: 文件路径列表
        mean: 均值数组
        std: 标准差数组
        device: 设备
        output_dir: 输出目录
    
    Returns:
        comparison_results: 对比结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 🔧 创建统一的对比视频目录
    comparison_videos_dir = os.path.join(output_dir, "comparison_videos")
    os.makedirs(comparison_videos_dir, exist_ok=True)
    
    # 🔧 整段测试：不限制窗口大小，但设置最大长度防止内存溢出
    # 根据数据集特点设置合理的最大长度
    if dataset_name == 'beat':
        max_length = 2048  # BEAT数据可能较长，设置较大的最大长度
    elif dataset_name == 'seg_finger':
        max_length = 512   # seg_finger数据通常较短
    elif dataset_name == 'single_motion':
        max_length = 2048  # single_motion数据可能较长
    else:
        max_length = 2048  # 默认最大长度
    
    pretrained_metrics_list = []
    finetuned_metrics_list = []
    
    print(f"\n{'='*80}")
    print(f"测试数据集: {dataset_name}")
    print(f"样本数量: {len(file_paths)}")
    print(f"测试模式: 整段测试（不使用窗口）")
    print(f"最大长度限制: {max_length} 帧")
    print(f"{'='*80}")
    
    for idx, file_path in enumerate(tqdm(file_paths, desc=f"Processing {dataset_name}")):
        file_name = os.path.basename(file_path).replace('.npy', '')
        
        # 加载motion数据
        original_motion_data = np.load(file_path)
        original_length = original_motion_data.shape[0]
        
        print(f"  Sample {idx}: {file_name}, 原始长度: {original_length} 帧")
        
        # 🔧 整段测试：直接使用完整序列（不进行窗口切分）
        # 评估预训练模型
        pretrained_metrics, pretrained_recon = evaluate_model(
            pretrained_model, original_motion_data.copy(), mean, std, device, max_length=max_length
        )
        pretrained_metrics_list.append(pretrained_metrics)
        
        # 评估finetune模型
        finetuned_metrics, finetuned_recon = evaluate_model(
            finetuned_model, original_motion_data.copy(), mean, std, device, max_length=max_length
        )
        finetuned_metrics_list.append(finetuned_metrics)
        
        # 🔧 evaluate_model已经处理了长度对齐，重建结果长度应该等于actual_frames
        actual_test_length = pretrained_metrics['actual_frames']
        
        # 检查长度一致性
        if pretrained_recon.shape[0] != actual_test_length:
            print(f"    ⚠️  预训练模型重建结果长度不匹配: {pretrained_recon.shape[0]} != {actual_test_length}")
        if finetuned_recon.shape[0] != actual_test_length:
            print(f"    ⚠️  Finetune模型重建结果长度不匹配: {finetuned_recon.shape[0]} != {actual_test_length}")
        
        # 可视化结果
        # 🔧 使用临时目录保存单个视频，然后生成三路对比视频
        temp_dir = os.path.join(dataset_output_dir, f"temp_{idx}_{file_name}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 🔧 使用实际测试的长度进行可视化
        # 原始motion（使用实际测试的长度，如果被截断则使用截断后的数据）
        original_motion_vis = original_motion_data[:actual_test_length]
        
        # 原始motion可视化（GT）
        original_video = visualize_motion(
            original_motion_vis,
            os.path.join(temp_dir, "original.csv"),
            mean, std, device
        )
        
        # 预训练模型重建（Baseline）
        pretrained_video = visualize_motion(
            pretrained_recon,
            os.path.join(temp_dir, "pretrained.csv"),
            mean, std, device
        )
        
        # Finetune模型重建
        finetuned_video = visualize_motion(
            finetuned_recon,
            os.path.join(temp_dir, "finetuned.csv"),
            mean, std, device
        )
        
        # 🔧 创建三路对比视频（GT | Baseline | Finetune）
        comparison_video_name = f"{dataset_name}_sample_{idx}_{file_name}_compare.mp4"
        comparison_video_path = os.path.join(comparison_videos_dir, comparison_video_name)
        concat_three_videos(original_video, pretrained_video, finetuned_video, comparison_video_path)
        
        # 清理临时文件（可选，保留单个视频文件以便后续查看）
        # import shutil
        # shutil.rmtree(temp_dir)
        
        print(f"  Sample {idx}: {file_name}")
        print(f"    Pretrained - Recon Loss: {pretrained_metrics['recon_loss']:.6f}, "
              f"Quant Loss: {pretrained_metrics['quant_loss']:.6f}, "
              f"Perplexity: {pretrained_metrics['perplexity']:.2f}")
        print(f"    Finetuned  - Recon Loss: {finetuned_metrics['recon_loss']:.6f}, "
              f"Quant Loss: {finetuned_metrics['quant_loss']:.6f}, "
              f"Perplexity: {finetuned_metrics['perplexity']:.2f}")
        print(f"    Improvement: Recon Loss: {pretrained_metrics['recon_loss'] - finetuned_metrics['recon_loss']:.6f}")
        print(f"    ✅ 对比视频已保存: {comparison_video_path}")
    
    # 计算平均指标
    avg_pretrained = {
        'recon_loss': np.mean([m['recon_loss'] for m in pretrained_metrics_list]),
        'quant_loss': np.mean([m['quant_loss'] for m in pretrained_metrics_list]),
        'perplexity': np.mean([m['perplexity'] for m in pretrained_metrics_list]),
    }
    
    avg_finetuned = {
        'recon_loss': np.mean([m['recon_loss'] for m in finetuned_metrics_list]),
        'quant_loss': np.mean([m['quant_loss'] for m in finetuned_metrics_list]),
        'perplexity': np.mean([m['perplexity'] for m in finetuned_metrics_list]),
    }
    
    improvement = {
        'recon_loss': avg_pretrained['recon_loss'] - avg_finetuned['recon_loss'],
        'quant_loss': avg_pretrained['quant_loss'] - avg_finetuned['quant_loss'],
        'perplexity': avg_pretrained['perplexity'] - avg_finetuned['perplexity'],
    }
    
    comparison_results = {
        'dataset': dataset_name,
        'pretrained': avg_pretrained,
        'finetuned': avg_finetuned,
        'improvement': improvement,
        'num_samples': len(file_paths),
    }
    
    print(f"\n平均指标 ({dataset_name}):")
    print(f"  Pretrained - Recon Loss: {avg_pretrained['recon_loss']:.6f}, "
          f"Quant Loss: {avg_pretrained['quant_loss']:.6f}, "
          f"Perplexity: {avg_pretrained['perplexity']:.2f}")
    print(f"  Finetuned  - Recon Loss: {avg_finetuned['recon_loss']:.6f}, "
          f"Quant Loss: {avg_finetuned['quant_loss']:.6f}, "
          f"Perplexity: {avg_finetuned['perplexity']:.2f}")
    print(f"  Improvement - Recon Loss: {improvement['recon_loss']:.6f}, "
          f"Quant Loss: {improvement['quant_loss']:.6f}, "
          f"Perplexity: {improvement['perplexity']:.2f}")
    
    return comparison_results


def generate_report(comparison_results_list, output_dir):
    """生成对比报告"""
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VQ-VAE Finetune前后对比报告\n")
        f.write("="*80 + "\n\n")
        
        for result in comparison_results_list:
            dataset = result['dataset']
            f.write(f"\n数据集: {dataset}\n")
            f.write("-"*80 + "\n")
            f.write(f"样本数量: {result['num_samples']}\n\n")
            
            f.write("预训练模型指标:\n")
            f.write(f"  重建Loss: {result['pretrained']['recon_loss']:.6f}\n")
            f.write(f"  量化Loss: {result['pretrained']['quant_loss']:.6f}\n")
            f.write(f"  困惑度: {result['pretrained']['perplexity']:.2f}\n\n")
            
            f.write("Finetune模型指标:\n")
            f.write(f"  重建Loss: {result['finetuned']['recon_loss']:.6f}\n")
            f.write(f"  量化Loss: {result['finetuned']['quant_loss']:.6f}\n")
            f.write(f"  困惑度: {result['finetuned']['perplexity']:.2f}\n\n")
            
            f.write("改进幅度:\n")
            f.write(f"  重建Loss改进: {result['improvement']['recon_loss']:.6f} "
                   f"({result['improvement']['recon_loss']/result['pretrained']['recon_loss']*100:.2f}%)\n")
            f.write(f"  量化Loss改进: {result['improvement']['quant_loss']:.6f} "
                   f"({result['improvement']['quant_loss']/result['pretrained']['quant_loss']*100:.2f}%)\n")
            f.write(f"  困惑度改进: {result['improvement']['perplexity']:.2f} "
                   f"({result['improvement']['perplexity']/result['pretrained']['perplexity']*100:.2f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("总结\n")
        f.write("="*80 + "\n")
        f.write("所有对比视频已保存在 comparison_videos/ 目录中\n")
        f.write("每个样本包含一个三路对比视频（GT | Baseline | Finetune）:\n")
        f.write("  - 格式: <dataset>_sample_<idx>_<filename>_compare.mp4\n")
        f.write("  - 内容: 原始动作 | 预训练模型重建 | Finetune模型重建\n")
    
    print(f"\n✅ 对比报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare VQ-VAE models before and after finetune')
    parser.add_argument('--pretrained_ckpt', type=str, required=True,
                       help='Path to pretrained checkpoint')
    parser.add_argument('--finetuned_ckpt', type=str, required=True,
                       help='Path to finetuned checkpoint')
    parser.add_argument('--config', type=str, 
                       default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='Config file to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test for each dataset')
    parser.add_argument('--output_dir', type=str, default='vqvae_comparison_results',
                       help='Output directory for comparison results')
    args = parser.parse_args()
    
    # 检查checkpoint文件
    if not os.path.exists(args.pretrained_ckpt):
        print(f"❌ Pretrained checkpoint not found: {args.pretrained_ckpt}")
        exit(1)
    if not os.path.exists(args.finetuned_ckpt):
        print(f"❌ Finetuned checkpoint not found: {args.finetuned_ckpt}")
        exit(1)
    
    # 加载配置文件
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.config)
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        exit(1)
    
    print(f"加载配置文件: {config_path}")
    motion_config = open_yaml(config_path)
    
    # 加载归一化统计量
    print(f"\n加载归一化统计量...")
    mean, std = load_normalization_stats(motion_config)
    print(f"统计量shape: Mean {mean.shape}, Std {std.shape}")
    
    # 加载模型
    print(f"\n加载预训练模型...")
    pretrained_model, pretrained_config = load_model(
        args.pretrained_ckpt, config_path, args.device
    )
    print(f"✅ 预训练模型加载成功")
    
    print(f"\n加载Finetune模型...")
    finetuned_model, finetuned_config = load_model(
        args.finetuned_ckpt, config_path, args.device
    )
    print(f"✅ Finetune模型加载成功")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 测试三个数据集
    datasets = ['beat', 'seg_finger', 'single_motion']
    comparison_results_list = []
    
    for dataset_name in datasets:
        try:
            # 获取数据集文件
            file_paths = get_dataset_files(dataset_name, args.num_samples)
            if len(file_paths) == 0:
                print(f"⚠️  No files found for dataset: {dataset_name}")
                continue
            
            # 比较模型
            comparison_results = compare_models(
                pretrained_model, finetuned_model,
                pretrained_config, finetuned_config,
                dataset_name, file_paths, mean, std,
                args.device, args.output_dir
            )
            comparison_results_list.append(comparison_results)
        except Exception as e:
            print(f"❌ Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成报告
    if comparison_results_list:
        generate_report(comparison_results_list, args.output_dir)
        comparison_videos_dir = os.path.join(args.output_dir, "comparison_videos")
        print(f"\n🎉 对比完成！")
        print(f"📊 对比报告: {os.path.join(args.output_dir, 'comparison_report.txt')}")
        print(f"📹 所有对比视频: {comparison_videos_dir}")
        if os.path.exists(comparison_videos_dir):
            video_count = len([f for f in os.listdir(comparison_videos_dir) if f.endswith('.mp4')])
            print(f"   共生成 {video_count} 个对比视频（GT | Baseline | Finetune）")
    else:
        print(f"\n❌ 没有成功处理任何数据集")


if __name__ == "__main__":
    main()

