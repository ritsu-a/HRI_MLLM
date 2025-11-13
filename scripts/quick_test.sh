#!/bin/bash

# 快速批量测试脚本
# 默认测试10个样本
# 使用方法：bash scripts/quick_test.sh [checkpoint_path] [num_samples]

CHECKPOINT=${1:-"/root/workspace/HRI_MLLM/output_disk0/ddp_kimi_lora_adaptor_beat_gestures/epoch_300.pt"}
NUM_SAMPLES=${2:-10}

echo "🧪 快速批量测试"
echo "="*60
echo "Checkpoint: $CHECKPOINT"
echo "测试样本数: $NUM_SAMPLES"
echo ""

# 检查checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint不存在"
    echo "可用的checkpoint:"
    ls -lh /root/workspace/HRI_MLLM/output_disk0/ddp_kimi_lora_adaptor/*.pt 2>/dev/null | tail -10
    exit 1
fi

# 显示checkpoint信息
echo "📊 Checkpoint信息:"
python -c "
import torch
ckpt = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
print(f'   Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'   Total Loss: {ckpt.get(\"loss\", \"N/A\"):.4f}')
print(f'   Audio Loss: {ckpt.get(\"audio_loss\", \"N/A\"):.4f}')
print(f'   Motion Loss: {ckpt.get(\"motion_loss\", \"N/A\"):.4f}')
"

echo ""
echo "🚀 开始批量测试 $NUM_SAMPLES 个样本..."
echo ""

cd /root/workspace/HRI_MLLM

# 清理旧的test_output
if [ -d "test_output" ]; then
    echo "🧹 清理旧的测试输出..."
    rm -rf test_output/*
fi

# 批量测试
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in $(seq 0 $((NUM_SAMPLES-1))); do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$((i+1))/$NUM_SAMPLES] 测试样本 $i"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python HRI_mllm/test/test_trained_model_from_jsonl.py \
        --checkpoint "$CHECKPOINT" \
        --sample_idx "$i" \
        --output_dir "test_output/sample_${i}" \
        2>&1 | grep -E "(✅|❌|📊|🎉|Error)" || true
    
    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✅ 样本 $i 完成"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "❌ 样本 $i 失败"
    fi
done

echo ""
echo "🎉 批量测试完成！"
echo "   成功: $SUCCESS_COUNT / $NUM_SAMPLES"
echo "   失败: $FAIL_COUNT / $NUM_SAMPLES"
echo ""

# 生成汇总报告
echo "📊 生成汇总报告..."
python - <<EOF
import json
from pathlib import Path
import numpy as np

reports = []
for i in range($NUM_SAMPLES):
    report_file = Path(f"test_output/sample_{i}/test_report.json")
    if report_file.exists():
        with open(report_file) as f:
            reports.append(json.load(f))

if reports:
    print(f"\n📋 测试汇总:")
    print(f"{'='*80}")
    print(f"{'样本':^6} | {'Audio Tokens':^15} | {'Motion Tokens':^15} | {'准确率':^12} | {'状态':^8}")
    print(f"{'-'*6}-|-{'-'*15}-|-{'-'*15}-|-{'-'*12}-|-{'-'*8}")
    
    accuracies = []
    for r in reports:
        idx = r['sample_idx']
        audio_gen = r['generated']['audio_tokens']
        motion_gen = r['generated']['motion_tokens']
        accuracy = r['comparison']['motion_accuracy']
        
        # 提取数值
        if accuracy != 'N/A':
            acc_val = float(accuracy.rstrip('%'))
            accuracies.append(acc_val)
        
        status = "✅" if accuracy != 'N/A' else "⚠️"
        print(f"{idx:^6} | {audio_gen:^15} | {motion_gen:^15} | {accuracy:^12} | {status:^8}")
    
    print(f"{'='*80}")
    
    if accuracies:
        print(f"\n📈 统计信息:")
        print(f"   平均准确率: {np.mean(accuracies):.2f}%")
        print(f"   最高准确率: {np.max(accuracies):.2f}%")
        print(f"   最低准确率: {np.min(accuracies):.2f}%")
        print(f"   标准差: {np.std(accuracies):.2f}%")
    
    # 保存汇总
    with open('test_output/summary_report.json', 'w') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 汇总报告已保存: test_output/summary_report.json")
    
    # 列出所有视频
    print(f"\n🎬 生成的视频文件:")
    for i in range(len(reports)):
        video_path = f"test_output/sample_{i}/visualization.mp4"
        print(f"   {video_path}")

else:
    print("❌ 未找到测试报告")
EOF

echo ""
echo "📁 输出目录: test_output/"
echo ""
echo "💡 提示："
echo "   - 查看视频: vlc test_output/sample_0/visualization.mp4"
echo "   - 查看报告: cat test_output/summary_report.json"
echo "   - 听原始问题: aplay test_output/sample_0/user_audio_question.wav"
echo "   - 听GT回复: aplay test_output/sample_0/gt_audio_response.wav"


