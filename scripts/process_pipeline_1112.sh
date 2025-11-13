#!/bin/bash

# 完整数据处理流程：从远程拷贝 -> 处理joint_vecs -> 生成tokens和jsonl
# 适用于1112批次数据

BASE_DIR="/root/workspace/HRI_MLLM"

echo "========================================"
echo "数据处理完整流程 - 1112批次"
echo "========================================"
echo ""
echo "本流程包含三个步骤："
echo "  1. 从远程服务器拷贝数据"
echo "  2. 处理数据为joint_vecs格式"
echo "  3. 生成tokens和jsonl文件"
echo ""
echo "========================================"
echo ""

# 询问用户从哪一步开始
echo "请选择开始步骤："
echo "  1 - 从步骤1开始（完整流程）"
echo "  2 - 从步骤2开始（跳过拷贝，已有原始数据）"
echo "  3 - 从步骤3开始（跳过拷贝和处理，已有joint_vecs）"
echo ""
read -p "请输入选项 [1-3]: " start_step

case $start_step in
    1)
        echo ""
        echo "========================================"
        echo "步骤 1/3: 从远程服务器拷贝数据"
        echo "========================================"
        bash ${BASE_DIR}/copy.sh
        if [ $? -ne 0 ]; then
            echo "❌ 步骤1失败，终止流程"
            exit 1
        fi
        echo "✅ 步骤1完成"
        echo ""
        
        echo "========================================"
        echo "步骤 2/3: 处理数据为joint_vecs"
        echo "========================================"
        bash ${BASE_DIR}/scripts/process_all_motions.sh
        if [ $? -ne 0 ]; then
            echo "❌ 步骤2失败，终止流程"
            exit 1
        fi
        echo "✅ 步骤2完成"
        echo ""
        
        echo "========================================"
        echo "步骤 3/3: 生成tokens和jsonl"
        echo "========================================"
        bash ${BASE_DIR}/scripts/generate_all_tokens_1112.sh
        if [ $? -ne 0 ]; then
            echo "❌ 步骤3失败"
            exit 1
        fi
        echo "✅ 步骤3完成"
        ;;
        
    2)
        echo ""
        echo "⏩ 跳过步骤1（拷贝数据）"
        echo ""
        
        echo "========================================"
        echo "步骤 2/3: 处理数据为joint_vecs"
        echo "========================================"
        bash ${BASE_DIR}/scripts/process_all_motions.sh
        if [ $? -ne 0 ]; then
            echo "❌ 步骤2失败，终止流程"
            exit 1
        fi
        echo "✅ 步骤2完成"
        echo ""
        
        echo "========================================"
        echo "步骤 3/3: 生成tokens和jsonl"
        echo "========================================"
        bash ${BASE_DIR}/scripts/generate_all_tokens_1112.sh
        if [ $? -ne 0 ]; then
            echo "❌ 步骤3失败"
            exit 1
        fi
        echo "✅ 步骤3完成"
        ;;
        
    3)
        echo ""
        echo "⏩ 跳过步骤1（拷贝数据）"
        echo "⏩ 跳过步骤2（处理joint_vecs）"
        echo ""
        
        echo "========================================"
        echo "步骤 3/3: 生成tokens和jsonl"
        echo "========================================"
        bash ${BASE_DIR}/scripts/generate_all_tokens_1112.sh
        if [ $? -ne 0 ]; then
            echo "❌ 步骤3失败"
            exit 1
        fi
        echo "✅ 步骤3完成"
        ;;
        
    *)
        echo "❌ 无效的选项"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "🎉 所有流程执行完成！"
echo "========================================"
echo ""
echo "生成的文件位置："
echo "  - joint_vecs目录: ${BASE_DIR}/data/*_1112_joint_vecs_0_999/"
echo "  - tokens目录: ${BASE_DIR}/data/*_1112_joint_vecs_0_999/tokens/"
echo "  - jsonl文件: ${BASE_DIR}/data/*.jsonl"
echo ""


