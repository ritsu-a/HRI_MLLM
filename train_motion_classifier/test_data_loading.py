#!/usr/bin/env python3
"""
测试数据加载是否正确
"""

import sys
import json
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from motion_dataset import MotionClassificationDataset

# 测试数据加载
data_dir = Path('/root/workspace/HRI_MLLM/data/motion_classification_dataset')
dataset_info_path = data_dir / 'dataset_info.json'
train_list_path = data_dir / 'train_list.json'

if not dataset_info_path.exists():
    print(f"数据集信息文件不存在: {dataset_info_path}")
    print("请先运行 prepare_dataset.py")
    sys.exit(1)

with open(train_list_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"训练集样本数: {len(train_data)}")

# 创建数据集
dataset = MotionClassificationDataset(
    train_data[:10],  # 只测试前10个样本
    dataset_info_path,
    num_frames=50,
    normalize=True
)

print(f"\n数据集创建成功")
print(f"数据集大小: {len(dataset)}")
print(f"类别数: {dataset.num_classes}")

# 测试加载一个样本
sample = dataset[0]
print(f"\n样本信息:")
print(f"  motion形状: {sample['motion'].shape}")
print(f"  label: {sample['label']}")
print(f"  motion_name: {sample['motion_name']}")
print(f"  motion_class: {sample['motion_class']}")

# 验证形状
expected_shape = (50, 491)
if sample['motion'].shape == expected_shape:
    print(f"\n✅ 数据形状正确: {sample['motion'].shape}")
else:
    print(f"\n❌ 数据形状错误: 期望 {expected_shape}, 实际 {sample['motion'].shape}")

# 测试batch加载
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))

print(f"\nBatch信息:")
print(f"  motion形状: {batch['motion'].shape}")
print(f"  label形状: {batch['label'].shape}")

expected_batch_shape = (2, 50, 491)
if batch['motion'].shape == expected_batch_shape:
    print(f"\n✅ Batch形状正确: {batch['motion'].shape}")
else:
    print(f"\n❌ Batch形状错误: 期望 {expected_batch_shape}, 实际 {batch['motion'].shape}")

print("\n数据加载测试完成！")


