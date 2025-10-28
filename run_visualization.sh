#!/bin/bash

# 可视化训练json里动作序列重建效果的运行脚本

echo "🎬 开始可视化训练json里动作序列重建效果..."

# 激活qwen环境
echo "🔧 激活qwen环境..."
source /root/anaconda3/bin/activate qwen

# 设置参数
JSONL_PATH="/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl"
GPT2_MODEL_PATH="/root/workspace/HRI_MLLM/output/motion_adaptor_v3/kimi_audio_motion_gpt2_brainco_30_100/transformers_format"
OUTPUT_DIR="./training_reconstruction_visualization"
NUM_SAMPLES=3  # 处理3个样本
MAX_NEW_TOKENS=4096
TEMPERATURE=0.8
TOP_K=50
REPETITION_PENALTY=1.1

echo "📁 输出目录: $OUTPUT_DIR"
echo "🔢 处理样本数: $NUM_SAMPLES"
echo "🎯 最大新生成token数: $MAX_NEW_TOKENS"
echo "🌡️  温度参数: $TEMPERATURE"
echo "🔝 Top-K采样: $TOP_K"
echo "🔄 重复惩罚: $REPETITION_PENALTY"

# 运行可视化脚本
python visualize_training_reconstruction.py \
    --jsonl_path "$JSONL_PATH" \
    --gpt2_model_path "$GPT2_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --repetition_penalty $REPETITION_PENALTY

echo "✅ 可视化完成！"
echo "📂 结果保存在: $OUTPUT_DIR"
echo "🎥 每个样本包含:"
echo "   - generated_motion.mp4 (生成的动作)"
echo "   - ground_truth_motion.mp4 (真实动作)"
echo "   - comparison.mp4 (对比视频)"
