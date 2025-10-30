#!/bin/bash

# 8卡训练SG_2_or_3_long_sentence_1030_en_kimi数据集，从checkpoint继续训练到1000 epochs

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  HRI_mllm/train/train_motion_adaptor.py \
  --resume_from output/motion_adaptor_10_v4/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_500.pt \
  --datasets BEAT SG_2_or_3_long_sentence_1030_en_kimi_tokens \
  --epochs 1000

