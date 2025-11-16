#!/bin/bash

# 8卡训练BEAT和single_motion数据集（合并训练）
# 
# 训练策略：
# - BEAT和single_motion合并训练
# - BEAT裁剪到最长256
# - single_motion每个epoch随机选取1/3的数据
# - 统一batch_size=64, max_seq_length=256
#
# 从 epoch 1350 继续训练

torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  HRI_mllm/train/train_motion_adaptor_finetune.py \
  --resume_from /root/workspace/HRI_MLLM/output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1350.pt \
  --epochs 3000 \
  --batch_size 64 \
  --max_seq_length 4096 \
  --single_motion_max_seq_length 4096 \
  --use_weighted_datasets \
  --dataset_weights 1.0 2.0

