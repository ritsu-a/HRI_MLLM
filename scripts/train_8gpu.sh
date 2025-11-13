#!/bin/bash

# 8卡分布式训练启动脚本
# 使用方法：bash scripts/train_8gpu.sh

echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo "8卡分布式训练：Kimi LoRA + Adaptor Full (BEAT + 8种手势)"
echo "数据集："
echo "  - BEAT_v2: 1945样本 (随机裁剪)"
echo "  - 8种手势数据 (1112版本): ~7710样本 (不裁剪)"
echo "    * FIST-BEAT: 994"
echo "    * FOREFINGER_RAISE: 982"
echo "    * HAND_CALL: 806"
echo "    * HAND_RING: 980"
echo "    * HAND_V_SIGN: 988"
echo "    * PALM_HALT: 984"
echo "    * THUMB_3FINGER_RAISE: 994"
echo "    * THUMB_UP: 982"
echo "总计：~9655样本"
echo "方法：Gumbel-Softmax + DDP + 选择性裁剪"
echo "序列长度：max_len=200 (BEAT裁剪, 手势保持原长)"
echo "保存频率：每10个epoch"
echo "Loss权重：audio_loss:motion_loss = 1:1"
echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo ""

# 切换到项目目录
cd /root/workspace/HRI_MLLM

# 检查GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "📊 检测到 $NUM_GPUS 个GPU"

if [ "$NUM_GPUS" -lt 8 ]; then
    echo "⚠️  警告：检测到的GPU少于8个，将使用所有可用GPU"
    USE_GPUS=$NUM_GPUS
else
    USE_GPUS=8
fi

# 显示GPU信息
echo ""
echo "🖥️  GPU信息："
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# 检查peft
echo ""
echo "📦 检查依赖..."
if ! python -c "import peft" 2>/dev/null; then
    echo "⚠️  peft未安装，正在安装..."
    pip install peft
else
    echo "✅ peft已安装"
fi

# 创建输出目录
OUTPUT_DIR="/root/workspace/HRI_MLLM/output_disk0/ddp_kimi_lora_adaptor_beat_gestures"
mkdir -p "$OUTPUT_DIR"
echo "✅ 输出目录: $OUTPUT_DIR"

# 创建日志目录
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo ""
echo "📝 日志将保存到: $LOG_FILE"
echo ""

# 训练配置
echo "⚙️  训练配置："
echo "   GPU数量: $USE_GPUS"
echo "   每GPU batch size: 2"
echo "   有效batch size: $((2 * USE_GPUS))"
echo "   Epochs: 30"
echo "   预计时间: ~1.5小时（8卡）"
echo ""

# 询问确认
read -p "是否开始训练？(y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 已取消训练"
    exit 0
fi

echo ""
echo "🚀 启动8卡分布式训练..."
echo "   按 Ctrl+C 可以停止训练"
echo "   监控命令: tail -f $LOG_FILE"
echo ""

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=$USE_GPUS \
    --master_port=29500 \
    HRI_mllm/finetune/train_jsonl_ddp.py \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 训练完成！"
    echo "   日志文件: $LOG_FILE"
    echo "   输出目录: $OUTPUT_DIR"
    echo "   最佳模型: $OUTPUT_DIR/best_model.pt"
else
    echo "❌ 训练出错（退出码: $EXIT_CODE）"
    echo "   查看日志: cat $LOG_FILE"
fi

exit $EXIT_CODE

