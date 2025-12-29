#!/bin/bash
# 动作分类器训练脚本

# 设置路径
NPY_DIR="/root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/extracted_motions/npy"
DATA_DIR="/root/workspace/HRI_MLLM/data/motion_classification_dataset"
MODEL_DIR="/root/workspace/HRI_MLLM/models/motion_classifier"

# 步骤1: 准备数据集
echo "步骤1: 准备数据集..."
python prepare_dataset.py \
    --npy_dir $NPY_DIR \
    --output_dir $DATA_DIR \
    --test_ratio 0.2 \
    --random_seed 42

# 步骤2: 训练模型
echo ""
echo "步骤2: 训练模型..."
python train.py \
    --data_dir $DATA_DIR \
    --output_dir $MODEL_DIR \
    --num_frames 50 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.3 \
    --model_type lstm \
    --use_class_weights \
    --normalize \
    --device cuda

echo "训练完成！"


