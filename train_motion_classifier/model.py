#!/usr/bin/env python3
"""
动作分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionClassifier(nn.Module):
    """动作分类器模型"""
    
    def __init__(self, input_dim=491, num_frames=50, num_classes=208, hidden_dim=512, num_layers=2, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度（491）
            num_frames: 帧数（50）
            num_classes: 类别数（208）
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout率
        """
        super(MotionClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类头
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 because bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, input_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # lstm_out: (batch_size, num_frames, hidden_dim * 2)
        # 也可以使用平均池化或注意力机制
        # 这里使用最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # 分类
        out = self.fc1(last_hidden)
        out = F.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits


class MotionClassifierCNN(nn.Module):
    """基于CNN的动作分类器（备选方案）"""
    
    def __init__(self, input_dim=491, num_frames=50, num_classes=208, hidden_dim=512, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度（491）
            num_frames: 帧数（50）
            num_classes: 类别数（208）
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super(MotionClassifierCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # 1D卷积层
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_frames, input_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # 确保输入是3D的
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, frames, features), got {x.dim()}D input with shape {x.shape}")
        
        # 转换为 (batch_size, input_dim, num_frames)
        # 从 (batch_size, num_frames, input_dim) 到 (batch_size, input_dim, num_frames)
        x = x.transpose(1, 2)
        
        # 卷积
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 全局平均池化
        x = self.pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # 分类
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits

