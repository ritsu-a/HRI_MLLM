#!/bin/bash

# 测试脚本 - 对应训练版本 v14
# 使用带标签的single_motion数据集进行测试
# 注意：需要根据实际训练进度更新checkpoint路径（epoch_XXX.pt）

python HRI_mllm/test/test_adaptor_jsonl_comparison.py \
    --jsonl_path /root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens_with_labels_train_other.jsonl \
    --motion_adaptor_path /root/workspace/HRI_MLLM/output_disk0/motion_adaptor_v14/kimi_audio_motion_gpt2_brainco_finetune_beat_single_motion/checkpoints/epoch_4000.pt \
    --vqvae_config g1_vqvae_arbitrary_length_balanced.yaml \
    --vqvae_checkpoint output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt \
    --output_dir ./adaptor_comparison_results \
    --num_samples 50 \
    --temperature 1.8 \
    --top_k 10 \
    --repetition_penalty 1.5 \
    --enable_model_debug