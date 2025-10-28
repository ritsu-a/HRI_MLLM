#!/bin/bash

# 对比Teacher Forcing和Free Generation模式的脚本
# 同时显示两种模式的loss和生成效果

echo "🔄 开始对比Teacher Forcing和Free Generation模式..."

# 激活qwen环境
echo "🔧 激活qwen环境..."
source /root/anaconda3/bin/activate qwen

# 设置参数
JSONL_PATH="/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl"
GPT2_MODEL_PATH="/root/workspace/HRI_MLLM/output/motion_adaptor_v2/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_300.pt"
OUTPUT_DIR="./mode_comparison_test"
NUM_SAMPLES=3  # 对比3个样本
MAX_NEW_TOKENS=4096

echo "📁 输出目录: $OUTPUT_DIR"
echo "🔢 对比样本数: $NUM_SAMPLES"
echo "🎯 最大新生成token数: $MAX_NEW_TOKENS"
echo "🔄 模式: 对比Teacher Forcing vs Free Generation"

# 运行对比测试
python visualize_training_reconstruction.py \
    --jsonl_path "$JSONL_PATH" \
    --gpt2_model_path "$GPT2_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --compare_modes

echo "✅ 对比测试完成！"
echo "📂 结果保存在: $OUTPUT_DIR"
echo "📊 检查Teacher Forcing loss是否接近0.01"
echo "🎥 对比视频显示两种模式的生成效果差异"
