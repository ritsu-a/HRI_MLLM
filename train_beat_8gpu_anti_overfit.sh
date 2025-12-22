#!/bin/bash

  # 8卡训练BEAT和single_motion_sentence_version2_kimi_tokens数据集，使用防过拟合优化
# 优化措施：
# 1. 添加验证集支持（每50个epoch验证一次）
# 2. 早停机制（patience=200）
# 3. 增强正则化（dropout=0.15, weight_decay=0.01, label_smoothing=0.1）
# 4. 优化学习率调度（CosineAnnealingLR with warmup）
# 5. 验证频率控制（每50个epoch验证一次，减少验证开销）

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  HRI_mllm/train/train_motion_adaptor_anti_overfit.py \
  --jsonl_files /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/synthetic_data_en_tokens_train.jsonl \
                /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_part2_joint_vecs/synthetic_data_en_part2_tokens_train.jsonl \
  --val_jsonl_files /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/synthetic_data_en_tokens_test.jsonl \
  --epochs 3000 \
  --version v20_synthetic_data_en_fix_history_64 \
  --weight_decay 0.01 \
  --label_smoothing 0.1 \
  --dropout 0.15 \
  --early_stop_patience 200 \
  --early_stop_min_delta 0.001 \
  --no_validation

  # 参数说明：
  # --epochs: 最大训练轮数
  # --weight_decay: L2正则化系数，防止过拟合
  # --label_smoothing: 标签平滑，提高泛化能力
  # --dropout: Dropout率（resid_pdrop, embd_pdrop, attn_pdrop）
  # --no_validation: 完全禁用验证功能（加快训练速度）

