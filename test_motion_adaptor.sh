#!/bin/bash

# 测试Motion Adaptor模型在测试集上的性能（Teacher-forcing模式）
# 批量测试多个checkpoint：从epoch 50到500，每50个epoch一次
# 对应训练脚本: train_beat_8gpu.sh

# 配置参数
BASE_CHECKPOINT_DIR="output_disk0/motion_adaptor_v19_synthetic_data_en/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints"
BASE_DATA_DIR="/root/workspace/HRI_MLLM/data/synthetic_data/SG_2_or_3_long_sentence_1030_en_joint_vecs"
TEST_JSONL_TRAIN="${BASE_DATA_DIR}/synthetic_data_en_tokens_train.jsonl"
TEST_JSONL_TEST="${BASE_DATA_DIR}/synthetic_data_en_tokens_test.jsonl"
OUTPUT_DIR="./test_results_motion_adaptor"
NUM_SAMPLES=100
BATCH_SIZE=8

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 存储所有checkpoint的结果
ALL_RESULTS_FILE="${OUTPUT_DIR}/all_checkpoints_results.json"
ALL_RESULTS="{}"

# 循环测试多个checkpoint：50, 100, 150, ..., 500
for epoch in 50 100 150 200 250 300 350 400 450 500; do
    CHECKPOINT_PATH="${BASE_CHECKPOINT_DIR}/epoch_${epoch}.pt"
    
    # 检查checkpoint是否存在
    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "⚠️  Checkpoint不存在，跳过: ${CHECKPOINT_PATH}"
        continue
    fi
    
    echo ""
    echo "============================================================"
    echo "测试 Checkpoint: epoch_${epoch}.pt"
    echo "============================================================"
    echo ""
    
    # 为每个checkpoint创建单独的输出目录
    CHECKPOINT_OUTPUT_DIR="${OUTPUT_DIR}/epoch_${epoch}"
    
    # 运行测试
    python HRI_mllm/test/test_motion_adaptor.py \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --test_jsonl ${TEST_JSONL_TRAIN} ${TEST_JSONL_TEST} \
        --output_dir ${CHECKPOINT_OUTPUT_DIR} \
        --num_samples ${NUM_SAMPLES} \
        --batch_size ${BATCH_SIZE}
    
    # 读取该checkpoint的结果并合并到总结果中
    if [ -f "${CHECKPOINT_OUTPUT_DIR}/test_results.json" ]; then
        # 使用Python合并JSON结果
        python3 << EOF
import json
import sys

# 读取总结果
try:
    with open('${ALL_RESULTS_FILE}', 'r') as f:
        all_results = json.load(f)
except:
    all_results = {}

# 读取当前checkpoint的结果
with open('${CHECKPOINT_OUTPUT_DIR}/test_results.json', 'r') as f:
    checkpoint_result = json.load(f)

# 添加当前checkpoint的结果
all_results[f'epoch_${epoch}'] = checkpoint_result

# 保存总结果
with open('${ALL_RESULTS_FILE}', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"✅ 已合并 epoch_${epoch} 的结果")
EOF
    fi
    
    echo ""
    echo "完成测试: epoch_${epoch}.pt"
    echo ""
done

# 生成汇总报告
echo ""
echo "============================================================"
echo "所有Checkpoint测试汇总"
echo "============================================================"
echo ""

python3 << EOF
import json
import os

results_file = '${ALL_RESULTS_FILE}'
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Test Loss':<15} {'Test Acc':<15}")
    print("-" * 70)
    
    for epoch in sorted(all_results.keys(), key=lambda x: int(x.split('_')[1])):
        result = all_results[epoch]
        checkpoint_path = result.get('checkpoint_path', '')
        test_datasets = result.get('test_datasets', {})
        
        train_result = test_datasets.get('synthetic_data_en_tokens_train', {}).get('teacher_forcing', {})
        test_result = test_datasets.get('synthetic_data_en_tokens_test', {}).get('teacher_forcing', {})
        
        train_loss = train_result.get('loss', 0.0)
        train_acc = train_result.get('accuracy', 0.0)
        test_loss = test_result.get('loss', 0.0)
        test_acc = test_result.get('accuracy', 0.0)
        
        print(f"{epoch:<10} {train_loss:<15.4f} {train_acc:<15.4f} {test_loss:<15.4f} {test_acc:<15.4f}")
    
    print(f"\n✅ 所有结果已保存到: {results_file}")
else:
    print("⚠️  未找到结果文件")
EOF

echo ""
echo "✅ 批量测试完成！"

