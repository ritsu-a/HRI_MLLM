#!/usr/bin/env python3
"""
将single_motion_sentence_version2_kimi_labeled_tokens.jsonl划分成训练集和测试集
"""

import json
import random
import os
from pathlib import Path

def split_jsonl(input_file, train_output, test_output, test_ratio=0.2, seed=42):
    """
    将JSONL文件划分成训练集和测试集
    
    Args:
        input_file: 输入的JSONL文件路径
        train_output: 训练集输出路径
        test_output: 测试集输出路径
        test_ratio: 测试集比例（默认0.2，即20%）
        seed: 随机种子，确保可重复性
    """
    # 设置随机种子
    random.seed(seed)
    
    # 读取所有行
    print(f"Reading {input_file}...")
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                lines.append(line)
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines}")
    
    # 打乱顺序
    random.shuffle(lines)
    
    # 计算划分点
    test_size = int(total_lines * test_ratio)
    train_size = total_lines - test_size
    
    print(f"Train set: {train_size} lines ({train_size/total_lines*100:.1f}%)")
    print(f"Test set: {test_size} lines ({test_size/total_lines*100:.1f}%)")
    
    # 写入训练集
    print(f"Writing train set to {train_output}...")
    with open(train_output, 'w', encoding='utf-8') as f:
        for line in lines[:train_size]:
            f.write(line)
    
    # 写入测试集
    print(f"Writing test set to {test_output}...")
    with open(test_output, 'w', encoding='utf-8') as f:
        for line in lines[train_size:]:
            f.write(line)
    
    print(f"✅ Split completed!")
    print(f"   Train: {train_output}")
    print(f"   Test: {test_output}")

if __name__ == "__main__":
    # 文件路径
    data_dir = Path("/root/workspace/HRI_MLLM/data")
    input_file = data_dir / "single_motion_sentence_version2_kimi_labeled_tokens.jsonl"
    train_output = data_dir / "single_motion_sentence_version2_kimi_labeled_tokens_train.jsonl"
    test_output = data_dir / "single_motion_sentence_version2_kimi_labeled_tokens_test.jsonl"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        exit(1)
    
    # 划分数据集（80%训练，20%测试）
    split_jsonl(
        input_file=str(input_file),
        train_output=str(train_output),
        test_output=str(test_output),
        test_ratio=0.2,
        seed=42
    )

