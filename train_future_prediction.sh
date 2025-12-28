#!/bin/bash

# 训练脚本：音频-动作未来预测任务
# 任务：输入 = 3*padding + 25*past_motion + 50*future_audio + 50*future_motion_padding = 128
#      输出 = 50*future_motion（作为监督）

torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  HRI_mllm/train/train_motion_future_prediction.py \
  --jsonl_files /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/synthetic_data_en_tokens_train.jsonl \
  --val_jsonl_files /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/synthetic_data_en_tokens_test.jsonl \
  --epochs 1000 \
  --version v23_future_prediction \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --dropout 0.1 \
  --val_freq 1000 \
  --save_freq 5 \
  --padding_frames 3 \
  --past_motion_frames 25 \
  --future_audio_frames 50 \
  --future_motion_frames 50 

# 参数说明：
# --jsonl_files: 训练数据JSONL文件路径（可多个）
# --val_jsonl_files: 验证数据JSONL文件路径（可多个）
# --epochs: 训练轮数
# --version: 模型版本标识
# --batch_size: 每个GPU的batch size
# --learning_rate: 学习率
# --weight_decay: L2正则化系数
# --dropout: Dropout率
# --val_freq: 验证频率（每N个epoch验证一次）
# --save_freq: 保存checkpoint频率（每N个epoch保存一次）
# --padding_frames: 初始padding帧数（默认3）
# --past_motion_frames: 历史motion帧数（默认25）
# --future_audio_frames: 未来audio帧数（默认50）
# --future_motion_frames: 未来motion帧数（默认50，作为输出监督，输入序列中对应位置是padding）
# --resume_from: 从指定checkpoint文件恢复训练
# --max_samples: 最大样本数量（用于调试，None表示加载全部，当前设置为100）

