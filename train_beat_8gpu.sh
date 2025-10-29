#!/bin/bash

# 8卡训练BEAT数据集，从checkpoint继续训练到1000 epochs

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  HRI_mllm/train/train_motion_adaptor.py \
  --datasets BEAT \
  --epochs 500

