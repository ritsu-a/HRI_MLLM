#!/bin/bash

# 8卡训练BEAT和single_motion数据集（合并训练）
# 
# 训练策略：
# - BEAT和single_motion合并训练
# - BEAT裁剪到最长256
# - single_motion每个epoch随机选取1/3的数据
# - 统一batch_size=64, max_seq_length=4096
#
# 从 epoch 2550 继续训练
# 使用带标签的single_motion数据集（支持9分类label任务）

torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  HRI_mllm/train/train_motion_adaptor_finetune.py \
  --resume_from /root/workspace/HRI_MLLM/output_disk0/motion_adaptor_v12/kimi_audio_motion_gpt2_brainco_finetune_beat_single_motion/checkpoints/epoch_2850.pt \
  --epochs 4000 \
  --batch_size 64 \
  --max_seq_length 4096 \
  --single_motion_max_seq_length 4096 \
  --single_motion_jsonl /root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens_with_labels_train_other.jsonl \
  --version v12 \
  --use_weighted_datasets \
  --dataset_weights 1.0 2.0 \
  --debug_label_loss

