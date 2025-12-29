#!/usr/bin/env python3
"""
测试训练好的模型在单个文件上的分类结果
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from model import MotionClassifier, MotionClassifierCNN
from motion_dataset import MotionClassificationDataset
import re


def extract_motion_class(motion_name):
    """从动作名称中提取类别（去掉末尾的数字）- 与prepare_dataset.py保持一致"""
    class_match = re.match(r'(.+?)_(\d+)$', motion_name)
    if class_match:
        return class_match.group(1)
    else:
        return motion_name


def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    
    # 创建模型
    if model_config['model_type'] == 'lstm':
        model = MotionClassifier(
            input_dim=model_config['input_dim'],
            num_frames=model_config['num_frames'],
            num_classes=model_config['num_classes'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
    else:
        model = MotionClassifierCNN(
            input_dim=model_config['input_dim'],
            num_frames=model_config['num_frames'],
            num_classes=model_config['num_classes'],
            hidden_dim=model_config['hidden_dim'],
            dropout=model_config['dropout']
        )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_config


def preprocess_data(npy_path, num_frames=50, mean=None, std=None, normalize=True):
    """
    预处理数据：加载、下采样、归一化
    
    Args:
        npy_path: NPY文件路径
        num_frames: 目标帧数
        mean: 均值（用于归一化）
        std: 标准差（用于归一化）
        normalize: 是否归一化
    
    Returns:
        预处理后的tensor: (1, num_frames, 491)
    """
    # 加载数据
    data = np.load(npy_path)
    print(f"原始数据形状: {data.shape}")
    
    # 下采样
    original_frames = data.shape[0]
    if original_frames > num_frames:
        indices = np.linspace(0, original_frames - 1, num_frames, dtype=int)
        data = data[indices]
    elif original_frames < num_frames:
        padding = np.zeros((num_frames - original_frames, data.shape[1]), dtype=data.dtype)
        data = np.vstack([data, padding])
    
    print(f"下采样后形状: {data.shape}")
    
    # 归一化
    if normalize and mean is not None and std is not None:
        data = (data - mean) / std
        print(f"归一化后形状: {data.shape}")
    
    # 转换为tensor并添加batch维度
    data = torch.FloatTensor(data).unsqueeze(0)  # (1, num_frames, 491)
    
    return data


def predict(model, data, id_to_class, device='cuda', top_k=5):
    """
    进行预测
    
    Args:
        model: 模型
        data: 输入数据 (1, num_frames, 491)
        id_to_class: 类别ID到类别名称的映射
        device: 设备
        top_k: 显示前k个预测结果
    
    Returns:
        预测结果字典
    """
    data = data.to(device)
    
    with torch.no_grad():
        logits = model(data)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)  # (1, num_classes)
        
        # 获取top-k预测
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(id_to_class)), dim=1)
        
        # 转换为numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # 构建结果
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = id_to_class.get(int(idx), f"Unknown_{idx}")
            predictions.append({
                'rank': i + 1,
                'class_id': int(idx),
                'class_name': class_name,
                'probability': float(prob)
            })
        
        # 获取最高预测
        pred_class_id = int(top_indices[0])
        pred_class_name = id_to_class.get(pred_class_id, f"Unknown_{pred_class_id}")
        pred_prob = float(top_probs[0])
    
    return {
        'predicted_class_id': pred_class_id,
        'predicted_class_name': pred_class_name,
        'predicted_probability': pred_prob,
        'top_k_predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径（best_model.pth或final_model.pth）')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据集目录（包含dataset_info.json）')
    parser.add_argument('--npy_file', type=str, required=True,
                       help='要测试的NPY文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备：cuda或cpu，默认cuda')
    parser.add_argument('--top_k', type=int, default=5,
                       help='显示前k个预测结果，默认5')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集信息
    data_dir = Path(args.data_dir)
    dataset_info_path = data_dir / 'dataset_info.json'
    
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"数据集信息文件不存在: {dataset_info_path}")
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    id_to_class = {int(k): v for k, v in dataset_info['id_to_class'].items()}
    num_classes = dataset_info['num_classes']
    
    print(f"\n数据集信息:")
    print(f"  类别数: {num_classes}")
    print(f"  文件: {args.npy_file}")
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    model, model_config = load_model(args.model_path, device)
    print(f"模型类型: {model_config['model_type']}")
    print(f"输入帧数: {model_config['num_frames']}")
    print(f"类别数: {model_config['num_classes']}")
    
    # 加载归一化参数（如果数据集已生成）
    mean = None
    std = None
    normalize = True
    
    # 尝试从训练数据集中获取归一化参数
    try:
        # 创建一个临时数据集来获取归一化参数
        train_list_path = data_dir / 'train_list.json'
        if train_list_path.exists():
            with open(train_list_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            temp_dataset = MotionClassificationDataset(
                train_data[:1],  # 只需要一个样本来初始化
                dataset_info_path,
                num_frames=model_config['num_frames'],
                normalize=normalize
            )
            mean = temp_dataset.mean
            std = temp_dataset.std
            print(f"\n归一化参数:")
            print(f"  均值形状: {mean.shape}")
            print(f"  标准差形状: {std.shape}")
    except Exception as e:
        print(f"\n警告: 无法加载归一化参数: {e}")
        print("将不使用归一化")
        normalize = False
    
    # 预处理数据
    print(f"\n预处理数据...")
    npy_path = Path(args.npy_file)
    if not npy_path.exists():
        raise FileNotFoundError(f"NPY文件不存在: {npy_path}")
    
    data = preprocess_data(
        npy_path,
        num_frames=model_config['num_frames'],
        mean=mean,
        std=std,
        normalize=normalize
    )
    print(f"最终输入形状: {data.shape}")
    
    # 进行预测
    print(f"\n进行预测...")
    results = predict(model, data, id_to_class, device, top_k=args.top_k)
    
    # 显示结果
    print("\n" + "=" * 80)
    print("预测结果")
    print("=" * 80)
    print(f"文件: {npy_path.name}")
    print(f"\n最高预测:")
    print(f"  类别: {results['predicted_class_name']} (ID: {results['predicted_class_id']})")
    print(f"  概率: {results['predicted_probability']:.4f} ({results['predicted_probability']*100:.2f}%)")
    
    print(f"\nTop-{args.top_k} 预测:")
    for pred in results['top_k_predictions']:
        print(f"  {pred['rank']}. {pred['class_name']} (ID: {pred['class_id']}): "
              f"{pred['probability']:.4f} ({pred['probability']*100:.2f}%)")
    
    # 查找真实类别（优先从数据集中查找，确保与训练时完全一致）
    true_class = None
    true_class_id = None
    
    # 首先尝试从训练集或测试集中查找（这是最准确的方式）
    train_list_path = data_dir / 'train_list.json'
    test_list_path = data_dir / 'test_list.json'
    
    for list_path in [train_list_path, test_list_path]:
        if list_path.exists():
            with open(list_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            
            # 查找匹配的文件（通过文件名匹配）
            for item in data_list:
                item_path = Path(item['file_path'])
                # 匹配文件名（不包含路径）
                if item_path.name == npy_path.name:
                    true_class = item['motion_class']
                    true_class_id = item['class_id']
                    print(f"\n从数据集找到真实类别: {true_class} (ID: {true_class_id})")
                    break
            
            if true_class:
                break
    
    # 如果数据集中没找到，从文件名提取（使用与prepare_dataset.py相同的函数）
    if not true_class:
        motion_pattern = re.compile(r'_motion_\d+_(.+)\.npy$')
        match = motion_pattern.search(npy_path.name)
        if match:
            motion_name = match.group(1)
            # 使用和prepare_dataset.py完全相同的提取逻辑
            true_class = extract_motion_class(motion_name)
            print(f"\n从文件名提取真实类别: {true_class}")
            
            # 尝试从id_to_class中找到对应的ID
            for class_id, class_name in id_to_class.items():
                if class_name == true_class:
                    true_class_id = class_id
                    break
    
    # 判断预测是否正确
    if true_class:
        print(f"真实类别: {true_class}")
        if true_class_id is not None:
            print(f"真实类别ID: {true_class_id}")
        
        if results['predicted_class_name'] == true_class:
            print("  ✅ 预测正确！")
        else:
            print(f"  ❌ 预测错误")
            print(f"  预测类别: {results['predicted_class_name']} (ID: {results['predicted_class_id']})")
            print(f"  真实类别: {true_class}" + (f" (ID: {true_class_id})" if true_class_id is not None else ""))
            
            # 检查预测的类别ID是否在top-k中
            if true_class_id is not None:
                found_in_topk = False
                for pred in results['top_k_predictions']:
                    if pred['class_id'] == true_class_id:
                        print(f"  ⚠️  真实类别在Top-{args.top_k}中，排名: {pred['rank']}, 概率: {pred['probability']:.4f}")
                        found_in_topk = True
                        break
                if not found_in_topk:
                    print(f"  ⚠️  真实类别不在Top-{args.top_k}预测中")
    
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    main()

