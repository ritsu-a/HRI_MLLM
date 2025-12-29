#!/usr/bin/env python3
"""
准备动作分类数据集：划分训练集和测试集
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
import argparse


def extract_motion_class(motion_name):
    """从动作名称中提取类别（去掉末尾的数字）"""
    class_match = re.match(r'(.+?)_(\d+)$', motion_name)
    if class_match:
        return class_match.group(1)
    else:
        return motion_name


def prepare_dataset(npy_dir, output_dir, test_ratio=0.2, random_seed=42):
    """
    准备数据集，划分训练集和测试集
    
    Args:
        npy_dir: NPY文件目录
        output_dir: 输出目录
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    npy_dir = Path(npy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有npy文件
    npy_files = sorted(npy_dir.glob("*.npy"))
    print(f"找到 {len(npy_files)} 个NPY文件")
    
    # 提取动作名称和类别
    motion_pattern = re.compile(r'_motion_\d+_(.+)\.npy$')
    data_list = []
    class_to_id = {}
    id_to_class = {}
    
    for npy_file in npy_files:
        match = motion_pattern.search(npy_file.name)
        if match:
            motion_name = match.group(1)
            motion_class = extract_motion_class(motion_name)
            
            # 构建类别ID映射
            if motion_class not in class_to_id:
                class_id = len(class_to_id)
                class_to_id[motion_class] = class_id
                id_to_class[class_id] = motion_class
            
            data_list.append({
                'file_path': str(npy_file),
                'motion_name': motion_name,
                'motion_class': motion_class,
                'class_id': class_to_id[motion_class]
            })
    
    print(f"找到 {len(class_to_id)} 个动作类别")
    print(f"总样本数: {len(data_list)}")
    
    # 统计每个类别的样本数
    class_counts = defaultdict(int)
    for item in data_list:
        class_counts[item['motion_class']] += 1
    
    print(f"\n各类别样本数统计:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes[:10]:
        print(f"  {cls}: {count} 个样本")
    if len(sorted_classes) > 10:
        print(f"  ... 还有 {len(sorted_classes) - 10} 个类别")
    
    # 按类别分组，确保每个类别在训练集和测试集中都有样本
    class_data = defaultdict(list)
    for item in data_list:
        class_data[item['motion_class']].append(item)
    
    train_data = []
    test_data = []
    
    # 对每个类别进行划分
    for motion_class, items in class_data.items():
        if len(items) == 1:
            # 如果只有1个样本，放入训练集
            train_data.extend(items)
        else:
            # 使用分层划分
            train_items, test_items = train_test_split(
                items,
                test_size=test_ratio,
                random_state=random_seed,
                shuffle=True
            )
            train_data.extend(train_items)
            test_data.extend(test_items)
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  测试集: {len(test_data)} 个样本")
    
    # 保存数据集信息
    dataset_info = {
        'num_classes': len(class_to_id),
        'num_train_samples': len(train_data),
        'num_test_samples': len(test_data),
        'class_to_id': class_to_id,
        'id_to_class': id_to_class,
        'class_counts': dict(class_counts),
        'test_ratio': test_ratio,
        'random_seed': random_seed
    }
    
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"\n数据集信息已保存到: {info_path}")
    
    # 保存训练集和测试集列表
    train_list_path = output_dir / 'train_list.json'
    test_list_path = output_dir / 'test_list.json'
    
    with open(train_list_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_list_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"训练集列表已保存到: {train_list_path}")
    print(f"测试集列表已保存到: {test_list_path}")
    
    # 统计训练集和测试集中各类别的分布
    train_class_counts = defaultdict(int)
    test_class_counts = defaultdict(int)
    
    for item in train_data:
        train_class_counts[item['motion_class']] += 1
    for item in test_data:
        test_class_counts[item['motion_class']] += 1
    
    print(f"\n训练集类别分布（前10个）:")
    sorted_train = sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_train[:10]:
        print(f"  {cls}: {count} 个样本")
    
    print(f"\n测试集类别分布（前10个）:")
    sorted_test = sorted(test_class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_test[:10]:
        print(f"  {cls}: {count} 个样本")
    
    return dataset_info, train_data, test_data


def main():
    parser = argparse.ArgumentParser(description='准备动作分类数据集')
    parser.add_argument('--npy_dir', type=str, required=True,
                       help='NPY文件目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='测试集比例，默认0.2')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子，默认42')
    
    args = parser.parse_args()
    
    prepare_dataset(
        npy_dir=args.npy_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()


