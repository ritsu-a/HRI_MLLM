#!/bin/bash

# 8卡训练BEAT和single_motion数据集（合并训练）
# 
# 训练策略：
# - BEAT和single_motion合并训练
# - BEAT如果序列过长，随机裁剪到max_seq_length=512
# - single_motion每个epoch随机选取1/3的数据
# - 统一batch_size=64, max_seq_length=512
# - 模型n_positions=512
#
# 从 epoch 3000 继续训练
# 使用带标签的single_motion数据集（支持9分类label任务）

torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  HRI_mllm/train/train_motion_adaptor_finetune.py \
  --resume_from /root/workspace/HRI_MLLM/output_disk0/motion_adaptor_v13/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_3000.pt \
  --epochs 4000 \
  --batch_size 64 \
  --max_seq_length 512 \
  --single_motion_max_seq_length 512 \
  --single_motion_jsonl /root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens_with_labels_train_other.jsonl \
  --version v14 \
  --use_weighted_datasets \
  --dataset_weights 1.0 1.0 \
  --debug_label_loss

