#!/bin/bash

# 批量处理所有动作数据文件夹
# 每个文件夹处理前1000个文件（编号0-999）

# 定义基础路径
BASE_DIR="/root/workspace/HRI_MLLM"
DATA_DIR="${BASE_DIR}/data"
PROCESS_SCRIPT="${BASE_DIR}/HRI_mllm/datasets/process_hand_ring_0_1000.py"

# 定义所有需要处理的动作文件夹
MOTIONS=(
    "HAND_V_SIGN_1112"
    "FOREFINGER_RAISE_ONE-2_1112"
    "THUMB_UP_1112"
    "THUMB_FOREFINGER_AND_LITTLE_FINGER_RAISE_1112"
    "FIST-BEAT_1112"
    "HAND_CALL_1112"
    "HAND_RING_1112"
    "PALM_HALT_1112"
)

# 记录开始时间
START_TIME=$(date +%s)

echo "========================================"
echo "开始批量处理动作数据"
echo "处理范围: 0-999 (前1000个文件)"
echo "总共 ${#MOTIONS[@]} 个文件夹"
echo "========================================"
echo ""

# 循环处理每个文件夹
for i in "${!MOTIONS[@]}"; do
    motion="${MOTIONS[$i]}"
    input_dir="${DATA_DIR}/${motion}"
    
    echo "----------------------------------------"
    echo "[$(($i+1))/${#MOTIONS[@]}] 正在处理: ${motion}"
    echo "输入目录: ${input_dir}"
    echo "----------------------------------------"
    
    # 检查输入目录是否存在
    if [ ! -d "${input_dir}" ]; then
        echo "⚠️  警告: 目录不存在，跳过: ${input_dir}"
        echo ""
        continue
    fi
    
    # 调用Python处理脚本
    python ${PROCESS_SCRIPT} \
        --input_dir "${input_dir}" \
        --start_idx 0 \
        --end_idx 999
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ ${motion} 处理完成"
    else
        echo "❌ ${motion} 处理失败"
    fi
    echo ""
done

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "========================================"
echo "所有文件夹处理完成！"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "========================================"
