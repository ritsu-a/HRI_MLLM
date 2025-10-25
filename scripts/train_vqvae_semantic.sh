#!/bin/bash
# VQ-VAE语义增强训练启动脚本

# 设置项目根目录
PROJECT_ROOT="/root/workspace/HRI_MLLM"
cd $PROJECT_ROOT

# 配置选择
CONFIG_TYPE=${1:-"enhanced"}  # 默认使用增强配置
NUM_GPUS=${2:-1}              # 默认单GPU

# 根据配置类型选择配置文件
case $CONFIG_TYPE in
    "quick")
        CONFIG_PATH="HRI_mllm/model/motion_encoder/configs/vqvae_semantic_quick_test.yaml"
        echo "🚀 Using QUICK TEST config (200 epochs, ~1 hour)"
        ;;
    "conservative")
        CONFIG_PATH="HRI_mllm/model/motion_encoder/configs/vqvae_semantic_conservative.yaml"
        echo "🚀 Using CONSERVATIVE config (2000 epochs, balanced)"
        ;;
    "enhanced")
        CONFIG_PATH="HRI_mllm/model/motion_encoder/g1_vqvae_semantic_enhanced.yaml"
        echo "🚀 Using ENHANCED config (3000 epochs, recommended)"
        ;;
    "aggressive")
        CONFIG_PATH="HRI_mllm/model/motion_encoder/configs/vqvae_semantic_aggressive.yaml"
        echo "🚀 Using AGGRESSIVE config (5000 epochs, SeG focused)"
        ;;
    "original")
        CONFIG_PATH="HRI_mllm/model/motion_encoder/g1_vqvae_full_qpos.yaml"
        echo "🚀 Using ORIGINAL config (modified with semantic_weight)"
        ;;
    *)
        echo "❌ Unknown config type: $CONFIG_TYPE"
        echo "Available options: quick, conservative, enhanced, aggressive, original"
        exit 1
        ;;
esac

# 选择训练脚本
if [ "$CONFIG_TYPE" = "original" ]; then
    SCRIPT="HRI_mllm/train/train_vae.py"
    echo "📝 Using standard training script"
else
    SCRIPT="HRI_mllm/train/train_vae_semantic_enhanced.py"
    echo "📝 Using enhanced training script with detailed monitoring"
fi

echo "📁 Config: $CONFIG_PATH"
echo "🖥️  GPUs: $NUM_GPUS"
echo ""

# 启动训练
if [ $NUM_GPUS -gt 1 ]; then
    echo "🚀 Starting multi-GPU training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS $SCRIPT
else
    echo "🚀 Starting single-GPU training..."
    python $SCRIPT
fi

echo ""
echo "✅ Training completed!"

