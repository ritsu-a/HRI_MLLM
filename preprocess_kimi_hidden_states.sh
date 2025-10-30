#!/bin/bash
# 预处理Kimi模型的hidden states脚本

# 设置环境变量
export MUJOCO_GL=egl

# 参数配置
INPUT_JSONL="/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl"
OUTPUT_DIR="data/preprocessed_hidden_states/BEAT_v2_kimi"
KIMI_MODEL_PATH="moonshotai/Kimi-Audio-7B"
MAX_AUDIO_LENGTH=1024
BATCH_SIZE=1
DEVICE="cuda"

# 检查输入文件
if [ ! -f "$INPUT_JSONL" ]; then
    echo "❌ Input JSONL file not found: $INPUT_JSONL"
    exit 1
fi

echo "🚀 Starting Kimi hidden states preprocessing"
echo "📋 Configuration:"
echo "   - Input: $INPUT_JSONL"
echo "   - Output: $OUTPUT_DIR"
echo "   - Kimi Model: $KIMI_MODEL_PATH"
echo "   - Max Audio Length: $MAX_AUDIO_LENGTH"
echo "   - Batch Size: $BATCH_SIZE"
echo "   - Device: $DEVICE"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行预处理脚本
python HRI_mllm/scripts/preprocess_kimi_hidden_states.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --kimi_model_path "$KIMI_MODEL_PATH" \
    --max_audio_length $MAX_AUDIO_LENGTH \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Preprocessing completed successfully!"
    echo "📁 Preprocessed data saved to: $OUTPUT_DIR"
    echo ""
    echo "📊 Output structure:"
    echo "   - hidden_states/: Saved hidden states tensors"
    echo "   - metadata/: Sample metadata JSON files"
    echo "   - index.json: Index file with all sample information"
    echo ""
    echo "💡 To use preprocessed hidden states in training, add:"
    echo "   --preprocessed_hidden_states_dir $OUTPUT_DIR"
else
    echo ""
    echo "❌ Preprocessing failed!"
    echo "💡 Check the error messages above"
    exit 1
fi

