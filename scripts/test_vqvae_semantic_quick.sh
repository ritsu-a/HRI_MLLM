#!/bin/bash
# VQ-VAE语义增强模型快速测试脚本

PROJECT_ROOT="/root/workspace/HRI_MLLM"
cd $PROJECT_ROOT

# 参数
CHECKPOINT=${1:-"output/vqvae_semantic_enhanced/checkpoints/vqvae_final.pt"}
CONFIG=${2:-"g1_vqvae_semantic_enhanced.yaml"}
DEVICE=${3:-"cuda"}

echo "🔍 Testing VQ-VAE Semantic Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 Checkpoint: $CHECKPOINT"
echo "⚙️  Config:     $CONFIG"
echo "🖥️  Device:     $DEVICE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "可用的checkpoint路径示例:"
    echo "  - output/vqvae_semantic_enhanced/checkpoints/vqvae_final.pt"
    echo "  - output/vqvae_semantic_training/checkpoints/vqvae_semantic_final.pt"
    echo "  - output/vqvae_semantic_enhanced/checkpoints/vqvae_epoch_*.pt"
    echo ""
    echo "用法: $0 [checkpoint_path] [config_name] [device]"
    echo "示例: $0 output/vqvae_semantic_enhanced/checkpoints/vqvae_epoch_1000.pt"
    exit 1
fi

# 运行测试
python HRI_mllm/test/test_vae.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE"

echo ""
echo "✅ Testing completed!"
echo ""
echo "生成的文件:"
echo "  - vqvae_output_source.mp4    (原始动作)"
echo "  - vqvae_output_decoded.mp4   (重建动作)"
echo "  - vqvae_output_compare.mp4   (对比视频)"

