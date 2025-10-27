# VQ-VAE最终版本说明

## 📁 核心文件（已清理，只保留最佳版本）

### 配置文件
- `HRI_mllm/model/motion_encoder/g1_vqvae_final.yaml` - 最终优化配置（固定32帧窗口）
- `HRI_mllm/model/motion_encoder/g1_vqvae_arbitrary_length.yaml` - 🆕 任意长度训练配置

### 训练
- `HRI_mllm/train/train_vqvae_final.py` - 最终训练脚本

### 测试
- `HRI_mllm/test/test_vqvae.py` - 最终测试脚本

### 工具
- `HRI_mllm/utils/motion_utils/g1ml3d_final.py` - 工具函数

### 统计量
- `data/Mixed_Statistics_Equal_Fixed/` - 修复后的统计量

---

## 🚀 快速开始

### 训练

#### 1. 固定32帧窗口训练（稳定推荐）
```bash
VQVAE_CONFIG="g1_vqvae_final.yaml" \
  torchrun --nproc_per_node=8 \
  HRI_mllm/train/train_vqvae_final.py
```

#### 2. 🆕 任意长度训练（实验性）
```bash
VQVAE_CONFIG="g1_vqvae_arbitrary_length.yaml" \
  torchrun --nproc_per_node=8 \
  HRI_mllm/train/train_vqvae_final.py
```

**训练模式对比**：
- **固定32帧窗口**：稳定可靠，推荐用于生产环境
- **任意长度训练**：支持任意长度序列，但内存消耗更大，实验性功能

### 测试

#### 1. 固定32帧窗口测试
```bash
python HRI_mllm/test/test_vqvae.py \
  --config g1_vqvae_final.yaml \
  --checkpoint output/vqvae_final/checkpoints/vqvae_final.pt
```

#### 2. 🆕 任意长度测试
```bash
python HRI_mllm/test/test_vqvae.py \
  --config g1_vqvae_arbitrary_length.yaml \
  --checkpoint output/vqvae_arbitrary_length/checkpoints/vqvae_final.pt
```

**测试模式对比**：
- **固定32帧窗口**：按窗口切分测试，与训练一致
- **任意长度测试**：直接处理完整序列，无需窗口切分

---

## 📚 详细文档

### 核心文档（必读）
- **`STABLE_CONFIG_SUMMARY.md`** ⭐ - 当前稳定配置总结
- `TRAINING_DIAGNOSIS_REPORT.md` - 训练问题诊断（Loss不收敛、Beta太高）
- `TEMPORAL_SMOOTHNESS_SOLUTION.md` - 时序平滑解决方案（防止跳变）

### 技术探索（参考）
- `VARIABLE_LENGTH_TRAINING.md` - 变长窗口训练原理
- `VARIABLE_LENGTH_LOSS_DEBUG.md` - 变长训练bug分析（Mask loss放大491倍）
- `ROLLBACK_TO_FIXED_WINDOW.md` - 回退到固定窗口的原因

### 其他
- `VQVAE_FINAL_README.md` - 完整使用指南
- `COMPLETE_BUG_FIX_SUMMARY.md` - Bug修复总结
- `TEST_VQVAE_USAGE.md` - 测试脚本使用指南

## ⚠️ 常见问题

### Q: 训练Loss不收敛，剧烈波动？
**A**: 检查`beta`值！
- ❌ `beta: 0.5` - 太高，会导致quant_loss爆炸
- ✅ `beta: 0.25` - 推荐值（已修复）
- 详见：`TRAINING_DIAGNOSIS_REPORT.md`

### Q: 测试Loss比训练Loss高很多？
**A**: 序列长度不匹配！
- 训练用32帧窗口，但测试用完整序列
- 解决方案：
  1. 测试时按窗口切分（已修复）
  2. 使用变长窗口训练（推荐）
- 详见：`VARIABLE_LENGTH_TRAINING.md`

### Q: 生成的动作有跳变/抽搐？
**A**: 两个原因：VQ量化 + 窗口边界
- ❌ 原因1：VQ离散量化导致时序不连续
- ❌ 原因2：32帧窗口切分，边界处可能不连续
- ✅ 解决方案：增加时序平滑loss
  ```yaml
  velocity_loss_weight: 0.5  # 一阶平滑
  use_acceleration_loss: true  # 二阶平滑
  acceleration_loss_weight: 0.2
  ```
- 效果：跳变减少60-80%（完全消除很难，这是VQ的固有特性）
- 详见：`TEMPORAL_SMOOTHNESS_SOLUTION.md`
