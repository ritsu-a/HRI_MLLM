#!/bin/bash
# 统一Kimi-Motion模型训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

# 训练参数
KIMI_MODEL_PATH="moonshotai/Kimi-Audio-7B"
TRAIN_JSON_PATHS=(
    "/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl"
    "/root/workspace/HRI_MLLM/data/internet_data_v1_kimi_tokens.jsonl"
)
SAVE_DIR="output/unified_kimi_motion_model"
BATCH_SIZE=2
NUM_EPOCHS=5
LEARNING_RATE=1e-4

# 检查数据文件是否存在
echo "🔍 Checking data files..."
for json_path in "${TRAIN_JSON_PATHS[@]}"; do
    if [ ! -f "$json_path" ]; then
        echo "❌ Training data file not found: $json_path"
        echo "💡 Please ensure the JSON data files exist"
        exit 1
    fi
done

echo "✅ Data files check completed"

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 构建训练命令
TRAIN_CMD="python HRI_mllm/train/train_unified_kimi_motion.py"
TRAIN_CMD="$TRAIN_CMD --kimi_model_path $KIMI_MODEL_PATH"
TRAIN_CMD="$TRAIN_CMD --train_json_paths ${TRAIN_JSON_PATHS[*]}"

TRAIN_CMD="$TRAIN_CMD --save_dir $SAVE_DIR"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --num_epochs $NUM_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --max_audio_length 1024"
TRAIN_CMD="$TRAIN_CMD --max_motion_length 1024"
TRAIN_CMD="$TRAIN_CMD --interleave_ratio 1 1"
TRAIN_CMD="$TRAIN_CMD --save_every 50"
TRAIN_CMD="$TRAIN_CMD --use_wandb"
TRAIN_CMD="$TRAIN_CMD --wandb_project unified-kimi-motion"
TRAIN_CMD="$TRAIN_CMD --freeze_kimi"
TRAIN_CMD="$TRAIN_CMD --freeze_adaptor"
TRAIN_CMD="$TRAIN_CMD --train_mixer_only"

# Loss权重配置（motion token更重要）
TRAIN_CMD="$TRAIN_CMD --motion_loss_weight 1.0"
TRAIN_CMD="$TRAIN_CMD --audio_loss_weight 0.1"

# LoRA配置（用于微调Kimi模型最后2层，保持原始性能）
TRAIN_CMD="$TRAIN_CMD --lora_r 16"
TRAIN_CMD="$TRAIN_CMD --lora_alpha 32"
TRAIN_CMD="$TRAIN_CMD --lora_dropout 0.1"

TRAIN_CMD="$TRAIN_CMD --debug"

# 打印训练配置
echo "🚀 Starting Unified Kimi-Motion Model Training"
echo "📋 Training Configuration:"
echo "   - Kimi Model: $KIMI_MODEL_PATH"
echo "   - Train Data: ${TRAIN_JSON_PATHS[*]}"
echo "   - Save Dir: $SAVE_DIR"
echo "   - Batch Size: $BATCH_SIZE"
echo "   - Epochs: $NUM_EPOCHS"
echo "   - Learning Rate: $LEARNING_RATE"
echo "   - Max Audio Length: 1024"
echo "   - Max Motion Length: 1024"
echo "   - Interleave Ratio: 1:1"
echo "   - Freeze Kimi: True (using LoRA for last 2 layers)"
echo "   - Freeze Adaptor: True"
echo "   - Train Mixer Only: True"
echo "   - Loss Weights: motion=1.0, audio=0.1"
echo "   - LoRA Config: r=16, alpha=32, dropout=0.1"
echo ""

# 执行训练
echo "🔄 Executing training command..."
echo "$TRAIN_CMD"
echo ""

eval $TRAIN_CMD

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Model saved to: $SAVE_DIR"
    echo ""
    echo "📊 Training outputs:"
    ls -la "$SAVE_DIR"
else
    echo ""
    echo "❌ Training failed!"
    echo "💡 Check the error messages above"
    exit 1
fi
