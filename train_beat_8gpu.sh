#!/bin/bash

# 8卡训练BEAT和single_motion_sentence_version2_kimi_tokens数据集，保存在output/motion_adaptor_v7目录下

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  HRI_mllm/train/train_motion_adaptor.py \
  --jsonl_files /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/synthetic_data_en_tokens_train.jsonl \
  --epochs 3000 \
  --version v19_synthetic_data_en

  # --jsonl_files /root/workspace/HRI_MLLM/data/single_motion_for_tokenizer_1108_tokens_with_labels_train.jsonl \

