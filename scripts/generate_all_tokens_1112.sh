#!/bin/bash

# 批量生成所有动作数据的tokens和jsonl文件
# 处理所有1112后缀的数据

# 定义基础路径
BASE_DIR="/root/workspace/HRI_MLLM"
DATA_DIR="${BASE_DIR}/data"
SCRIPT_PATH="${BASE_DIR}/HRI_mllm/datasets/BEAT_preprocess/generate_hand_ring_tokens.py"

# VQ-VAE配置
VQVAE_CONFIG="g1_vqvae_arbitrary_length_balanced.yaml"
MODEL_NAME="moonshotai/Kimi-Audio-7B"
USER_PROMPT="Please response the following spoken content with speech and a corresponding full-body motion sequence."

# 定义所有需要处理的动作
declare -A MOTIONS=(
    ["HAND_V_SIGN_1112"]="HAND_V_SIGN_1112_tokens.jsonl"
    ["FOREFINGER_RAISE_ONE-2_1112"]="FOREFINGER_RAISE_ONE-2_1112_tokens.jsonl"
    ["THUMB_UP_1112"]="THUMB_UP_1112_tokens.jsonl"
    ["THUMB_FOREFINGER_AND_LITTLE_FINGER_RAISE_1112"]="THUMB_FOREFINGER_AND_LITTLE_FINGER_RAISE_1112_tokens.jsonl"
    ["FIST-BEAT_1112"]="FIST-BEAT_1112_tokens.jsonl"
    ["HAND_CALL_1112"]="HAND_CALL_1112_tokens.jsonl"
    ["HAND_RING_1112"]="HAND_RING_1112_tokens.jsonl"
    ["PALM_HALT_1112"]="PALM_HALT_1112_tokens.jsonl"
)

# 记录开始时间
START_TIME=$(date +%s)

echo "========================================"
echo "开始批量生成tokens和jsonl文件"
echo "处理范围: 0-999 (前1000个文件)"
echo "总共 ${#MOTIONS[@]} 个动作"
echo "========================================"
echo ""

# 计数器
total=${#MOTIONS[@]}
current=0

# 循环处理每个动作
for motion_name in "${!MOTIONS[@]}"; do
    current=$((current + 1))
    jsonl_filename="${MOTIONS[$motion_name]}"
    
    # 构建数据目录路径 (joint_vecs目录)
    data_dir="${DATA_DIR}/${motion_name}_joint_vecs_0_999"
    
    echo "----------------------------------------"
    echo "[${current}/${total}] 正在处理: ${motion_name}"
    echo "数据目录: ${data_dir}"
    echo "输出文件: ${jsonl_filename}"
    echo "----------------------------------------"
    
    # 检查数据目录是否存在
    if [ ! -d "${data_dir}" ]; then
        echo "⚠️  警告: 目录不存在，跳过: ${data_dir}"
        echo ""
        continue
    fi
    
    # 检查npy、wav目录
    if [ ! -d "${data_dir}/npy" ] || [ ! -d "${data_dir}/wav" ]; then
        echo "⚠️  警告: npy或wav目录不存在，跳过: ${motion_name}"
        echo ""
        continue
    fi
    
    # 运行Python脚本生成tokens和jsonl
    python ${SCRIPT_PATH} \
        --vqvae_config "${VQVAE_CONFIG}" \
        --model_name_or_path "${MODEL_NAME}" \
        --data_dir "${data_dir}" \
        --create_jsonl \
        --jsonl_output_name "${jsonl_filename}" \
        --user_prompt "${USER_PROMPT}" \
        --device cuda
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ ${motion_name} 处理完成"
    else
        echo "❌ ${motion_name} 处理失败"
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
echo "所有动作处理完成！"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "生成的jsonl文件位于: ${DATA_DIR}/../data/"
echo "========================================"

# 列出生成的所有jsonl文件
echo ""
echo "生成的jsonl文件列表:"
for jsonl_file in "${MOTIONS[@]}"; do
    jsonl_path="${DATA_DIR}/../data/${jsonl_file}"
    if [ -f "${jsonl_path}" ]; then
        file_size=$(du -h "${jsonl_path}" | cut -f1)
        echo "  ✅ ${jsonl_file} (${file_size})"
    else
        echo "  ❌ ${jsonl_file} (未找到)"
    fi
done


