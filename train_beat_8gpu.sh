#!/bin/bash

# 8卡训练BEAT和single_motion_sentence_version2_kimi_tokens数据集，保存在output/motion_adaptor_v7目录下

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  HRI_mllm/train/train_motion_adaptor.py \
  --datasets BEAT \
  --epochs 100

