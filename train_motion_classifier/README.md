# 动作分类器训练

这个目录包含训练动作分类器的代码，用于将动作向量（50×491维）分类到208个动作类别之一。

## 文件说明

- `prepare_dataset.py`: 准备数据集，划分训练集和测试集
- `motion_dataset.py`: PyTorch Dataset类，用于加载数据
- `model.py`: 模型定义（LSTM和CNN两种架构）
- `train.py`: 训练脚本

## 使用步骤

### 1. 准备数据集

首先运行 `prepare_dataset.py` 来划分训练集和测试集：

```bash
python prepare_dataset.py \
    --npy_dir /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/extracted_motions/npy \
    --output_dir /root/workspace/HRI_MLLM/data/motion_classification_dataset \
    --test_ratio 0.2 \
    --random_seed 42
```

这会在 `output_dir` 中创建：
- `dataset_info.json`: 数据集信息（类别映射等）
- `train_list.json`: 训练集文件列表
- `test_list.json`: 测试集文件列表

### 2. 训练模型

运行 `train.py` 来训练模型：

```bash
python train.py \
    --data_dir /root/workspace/HRI_MLLM/data/motion_classification_dataset \
    --output_dir /root/workspace/HRI_MLLM/models/motion_classifier \
    --num_frames 50 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.3 \
    --model_type cnn \
    --use_class_weights \
    --normalize
```

### 参数说明

#### prepare_dataset.py
- `--npy_dir`: NPY文件目录
- `--output_dir`: 输出目录
- `--test_ratio`: 测试集比例（默认0.2）
- `--random_seed`: 随机种子（默认42）

#### train.py
- `--data_dir`: 数据集目录（包含train_list.json和test_list.json）
- `--output_dir`: 模型输出目录
- `--num_frames`: 使用的帧数（默认50，原始数据是100帧，会自动下采样）
- `--batch_size`: 批次大小（默认32）
- `--epochs`: 训练轮数（默认50）
- `--lr`: 学习率（默认0.001）
- `--hidden_dim`: 隐藏层维度（默认512）
- `--num_layers`: LSTM层数（默认2，仅对LSTM模型有效）
- `--dropout`: Dropout率（默认0.3）
- `--model_type`: 模型类型，`lstm` 或 `cnn`（默认lstm）
- `--use_class_weights`: 使用类别权重处理类别不平衡
- `--normalize`: 归一化数据（默认开启）
- `--device`: 设备，`cuda` 或 `cpu`（默认cuda）

### 3. 输出文件

训练完成后，在 `output_dir` 中会生成：
- `best_model.pth`: 最佳模型（基于F1分数）
- `final_model.pth`: 最终模型
- `training_history.json`: 训练历史记录
- `test_results.json`: 测试集详细结果（包括每个类别的指标）

## 模型架构

### LSTM模型
- 双向LSTM编码器
- 使用最后一个时间步的输出
- 全连接层分类

### CNN模型
- 1D卷积层
- 全局平均池化
- 全连接层分类

## 数据格式

- 输入：`(batch_size, num_frames, 491)` 的动作向量
- 输出：`(batch_size, num_classes)` 的类别logits

## 评估指标

- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1 Score
- 每个类别的详细指标

## 测试模型

使用 `test_model.py` 来测试训练好的模型在单个文件上的分类结果：

```bash
python test_model.py \
    --model_path /root/workspace/HRI_MLLM/models/motion_classifier/best_model.pth \
    --data_dir /root/workspace/HRI_MLLM/data/motion_classification_dataset \
    --npy_file /root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs/extracted_motions/npy/1_motion_3_FOREFINGER_KICK_4.npy \
    --top_k 5
```

### 参数说明

- `--model_path`: 模型文件路径（best_model.pth 或 final_model.pth）
- `--data_dir`: 数据集目录（包含 dataset_info.json）
- `--npy_file`: 要测试的NPY文件路径
- `--device`: 设备，`cuda` 或 `cpu`（默认cuda）
- `--top_k`: 显示前k个预测结果（默认5）

### 输出示例

脚本会显示：
- 最高预测类别和概率
- Top-k预测结果
- 如果文件名包含类别信息，会显示真实类别并判断预测是否正确

