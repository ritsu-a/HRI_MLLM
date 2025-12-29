#!/usr/bin/env python3
"""
训练动作分类器
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from motion_dataset import MotionClassificationDataset
from model import MotionClassifier, MotionClassifierCNN


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="训练")
    for batch in pbar:
        motions = batch['motion'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(motions)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, class_names=None):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="评估")
        for batch in pbar:
            motions = batch['motion'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(motions)
            loss = criterion(logits, labels)
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 计算每个类别的指标
    per_class_metrics = {}
    if class_names:
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        for i, class_name in class_names.items():
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support[i])
            }
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_metrics': per_class_metrics,
        'predictions': all_preds,
        'labels': all_labels
    }


def compute_class_weights(dataset_info, train_data):
    """计算类别权重（用于处理类别不平衡）"""
    class_counts = {}
    for item in train_data:
        class_name = item['motion_class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # 计算权重：总样本数 / (类别数 * 该类别的样本数)
    total_samples = len(train_data)
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_name, count in class_counts.items():
        class_weights[class_name] = total_samples / (num_classes * count)
    
    # 转换为tensor
    id_to_class = {int(k): v for k, v in dataset_info['id_to_class'].items()}
    weights = torch.zeros(len(id_to_class))
    for class_id, class_name in id_to_class.items():
        if class_name in class_weights:
            weights[class_id] = class_weights[class_name]
    
    return weights


def main():
    parser = argparse.ArgumentParser(description='训练动作分类器')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据集目录（包含train_list.json和test_list.json）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num_frames', type=int, default=50,
                       help='使用的帧数，默认50')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小，默认32')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数，默认50')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率，默认0.001')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='隐藏层维度，默认512')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM层数，默认2')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout率，默认0.3')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'cnn'],
                       help='模型类型：lstm或cnn，默认lstm')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='是否使用类别权重（处理类别不平衡）')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='是否归一化数据，默认True')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备：cuda或cpu，默认cuda')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集信息
    data_dir = Path(args.data_dir)
    dataset_info_path = data_dir / 'dataset_info.json'
    train_list_path = data_dir / 'train_list.json'
    test_list_path = data_dir / 'test_list.json'
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    with open(train_list_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(test_list_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    num_classes = dataset_info['num_classes']
    id_to_class = {int(k): v for k, v in dataset_info['id_to_class'].items()}
    
    print(f"类别数: {num_classes}")
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    # 创建数据集
    train_dataset = MotionClassificationDataset(
        train_data, dataset_info_path, num_frames=args.num_frames, normalize=args.normalize
    )
    test_dataset = MotionClassificationDataset(
        test_data, dataset_info_path, num_frames=args.num_frames, normalize=args.normalize
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # 创建模型
    if args.model_type == 'lstm':
        model = MotionClassifier(
            input_dim=491,
            num_frames=args.num_frames,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    else:
        model = MotionClassifierCNN(
            input_dim=491,
            num_frames=args.num_frames,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    if args.use_class_weights:
        class_weights = compute_class_weights(dataset_info, train_data).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("使用类别权重")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练
    best_f1 = 0
    train_history = []
    
    print("\n开始训练...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        test_metrics = evaluate(model, test_loader, criterion, device, id_to_class)
        
        # 更新学习率
        scheduler.step(test_metrics['loss'])
        
        # 记录历史
        history = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1']
        }
        train_history.append(history)
        
        print(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"测试 - Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"测试 - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_metrics': test_metrics,
                'model_config': {
                    'input_dim': 491,
                    'num_frames': args.num_frames,
                    'num_classes': num_classes,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers if args.model_type == 'lstm' else None,
                    'dropout': args.dropout,
                    'model_type': args.model_type
                }
            }, best_model_path)
            print(f"保存最佳模型 (F1: {best_f1:.4f}) 到 {best_model_path}")
    
    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(train_history, f, indent=2, ensure_ascii=False)
    print(f"\n训练历史已保存到: {history_path}")
    
    # 保存最终模型
    final_model_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
        'model_config': {
            'input_dim': 491,
            'num_frames': args.num_frames,
            'num_classes': num_classes,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers if args.model_type == 'lstm' else None,
            'dropout': args.dropout,
            'model_type': args.model_type
        }
    }, final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 保存测试集详细结果
    final_test_metrics = evaluate(model, test_loader, criterion, device, id_to_class)
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'overall_metrics': {
                'loss': final_test_metrics['loss'],
                'accuracy': final_test_metrics['accuracy'],
                'precision': final_test_metrics['precision'],
                'recall': final_test_metrics['recall'],
                'f1': final_test_metrics['f1']
            },
            'per_class_metrics': final_test_metrics['per_class_metrics']
        }, f, indent=2, ensure_ascii=False)
    print(f"测试结果已保存到: {results_path}")
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()


