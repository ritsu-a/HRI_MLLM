#!/bin/bash
# 示例脚本：准备mixer训练数据

# 设置Kimi模型路径
KIMI_MODEL_PATH="/DATA/disk1/Kimi-Audio-7B-Instruct"

# 设置输出目录
OUTPUT_DIR="output/mixer_training_data"

# Beat数据集JSON文件路径列表
JSON_PATHS=(
    "data/beat_english_v0.2.1_tokens.jsonl"
    # 可以添加更多JSON文件路径
    # "data/other_dataset_tokens.jsonl"
)

# 运行数据准备脚本
python HRI_mllm/train/prepare_mixer_training_data.py \
    --json_paths "${JSON_PATHS[@]}" \
    --kimi_model_path "$KIMI_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda" \
    --max_samples 1000 \
    --debug

# 如果要处理全部数据，去掉--max_samples参数
# python HRI_mllm/train/prepare_mixer_training_data.py \
#     --json_paths "${JSON_PATHS[@]}" \
#     --kimi_model_path "$KIMI_MODEL_PATH" \
#     --output_dir "$OUTPUT_DIR" \
#     --device "cuda"

echo "✅ Data preparation completed!"
echo "Output directory: $OUTPUT_DIR"
echo "You can now use --preprocessed_hidden_states_dir=$OUTPUT_DIR when training mixer"

