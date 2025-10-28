#!/bin/bash

# Teacher Forcing模式测试脚本
# 用于验证模型loss是否与训练时一致（~0.01）

echo "🎓 开始Teacher Forcing模式测试..."

# 激活qwen环境
echo "🔧 激活qwen环境..."
source /root/anaconda3/bin/activate qwen

# 设置参数
JSONL_PATH="/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl"
GPT2_MODEL_PATH="/root/workspace/HRI_MLLM/output/motion_adaptor_v2/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_300.pt"
OUTPUT_DIR="./teacher_forcing_test"
NUM_SAMPLES=5  # 测试5个样本
MAX_NEW_TOKENS=4096

echo "📁 输出目录: $OUTPUT_DIR"
echo "🔢 测试样本数: $NUM_SAMPLES"
echo "🎯 最大新生成token数: $MAX_NEW_TOKENS"
echo "🎓 模式: Teacher Forcing (计算loss)"

# 运行teacher forcing测试
python visualize_training_reconstruction.py \
    --jsonl_path "$JSONL_PATH" \
    --gpt2_model_path "$GPT2_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --max_new_tokens $MAX_NEW_TOKENS \
    --teacher_forcing

echo "✅ Teacher Forcing测试完成！"
echo "📂 结果保存在: $OUTPUT_DIR"
echo "📊 检查loss值是否接近训练时的0.01水平"
