#!/bin/bash

# 8卡训练BEAT和single_motion数据集，支持动态batching和均匀采样
# 
# 功能说明：
# 1. --use_dynamic_batching: 按序列长度分组，提高GPU利用率
# 2. --use_balanced_sampling: 对single_motion的8类动作进行均匀采样
# 3. --dataset_weights: BEAT和single_motion的权重（1.0 1.0表示等权重）
#
# 注意：请根据实际情况修改--resume_from路径

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  HRI_mllm/train/train_motion_adaptor_finetune.py \
  --resume_from output/motion_adaptor_v10/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_300.pt \
  --epochs 3000 \
  --use_dynamic_batching \
  --use_balanced_sampling \
  --batch_size 64 \
  --max_seq_length 4096 \
  --dataset_weights 1.0 0.0

