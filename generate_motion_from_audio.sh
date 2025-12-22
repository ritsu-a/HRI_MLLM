#!/bin/bash

# 从audio tokens生成motion tokens的推理脚本
# 对应训练脚本: train_beat_8gpu.sh

# 配置参数
CHECKPOINT_PATH="output_disk0/motion_adaptor_v19_synthetic_data_en/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_300.pt"
AUDIO_TOKENS_FILE="input_audio_tokens.json"
OUTPUT_FILE="output_motion_tokens.json"

# 生成参数
MAX_NEW_TOKENS=512
TEMPERATURE=0.8
TOP_K=50
REPETITION_PENALTY=1.1

python HRI_mllm/inference/generate_motion_from_audio.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --audio_tokens_file ${AUDIO_TOKENS_FILE} \
    --output_file ${OUTPUT_FILE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_k ${TOP_K} \
    --repetition_penalty ${REPETITION_PENALTY}



