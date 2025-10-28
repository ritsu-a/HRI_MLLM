#!/bin/bash

# 8卡训练BEAT数据集，从checkpoint继续训练到1000 epochs

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  HRI_mllm/train/train_motion_adaptor.py \
  --resume_from output/motion_adaptor_v2/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_300.pt \
  --datasets BEAT \
  --epochs 1000

