#!/bin/bash

# 快速批量测试脚本
# 默认测试10个样本
# 使用方法：bash scripts/quick_test.sh [checkpoint_path] [num_samples]

CHECKPOINT=${1:-"/root/workspace/HRI_MLLM/output_disk0/ddp_kimi_lora_adaptor_beat_gestures/epoch_300.pt"}
NUM_SAMPLES=${2:-10}

echo "🧪 快速批量测试"
echo "============================================================"
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

# 查找 VQ-VAE checkpoint（可选）
VQVAE_CKPT=""
if [ -f "output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt" ]; then
    VQVAE_CKPT="--vqvae_checkpoint output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt"
elif [ -f "output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt" ]; then
    VQVAE_CKPT="--vqvae_checkpoint output/vqvae_finetune_beat_segfinger/checkpoints/vqvae_finetune_final.pt"
else
    echo "⚠️  未找到 VQ-VAE checkpoint，将使用预训练模型"
fi

# 使用test_trained_model_from_jsonl.py进行批量测试
for i in $(seq 0 $((NUM_SAMPLES-1))); do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$((i+1))/$NUM_SAMPLES] 测试样本 $i"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python HRI_mllm/test/test_trained_model_from_jsonl.py \
        --checkpoint "$CHECKPOINT" \
        --jsonl_path /root/workspace/HRI_MLLM/data/THUMB_FOREFINGER_AND_LITTLE_FINGER_RAISE_1112_tokens.jsonl \
        --sample_idx "$i" \
        --output_dir "test_output/sample_${i}" \
        --temperature 1.0 \
        --top_k 50 \
        --max_motion_tokens 512 \
        --vqvae_config g1_vqvae_arbitrary_length_balanced.yaml \
        $VQVAE_CKPT
    
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
    print(f"{'='*90}")
    print(f"{'样本':^6} | {'Kimi准确率':^15} | {'GT Audio准确率':^15} | {'GT Motion':^12} | {'状态':^8}")
    print(f"{'-'*6}-|-{'-'*15}-|-{'-'*15}-|-{'-'*12}-|-{'-'*8}")
    
    kimi_accuracies = []
    gt_audio_accuracies = []
    
    for r in reports:
        idx = r['sample_idx']
        motion_gt = r['ground_truth']['motion_tokens']
        
        kimi_acc = r['kimi_audio_motion']['motion_accuracy']
        gt_audio_acc = r['gt_audio_motion']['motion_accuracy']
        
        # 提取数值
        if kimi_acc != 'N/A':
            kimi_accuracies.append(float(kimi_acc.rstrip('%')))
        if gt_audio_acc != 'N/A':
            gt_audio_accuracies.append(float(gt_audio_acc.rstrip('%')))
        
        status = "✅"
        print(f"{idx:^6} | {kimi_acc:^15} | {gt_audio_acc:^15} | {motion_gt:^12} | {status:^8}")
    
    print(f"{'='*90}")
    
    if kimi_accuracies:
        print(f"\n📈 统计信息:")
        print(f"   [方案A] Kimi audio → motion:")
        print(f"      平均准确率: {np.mean(kimi_accuracies):.2f}%")
        print(f"      最高准确率: {np.max(kimi_accuracies):.2f}%")
        print(f"      最低准确率: {np.min(kimi_accuracies):.2f}%")
        print(f"      标准差: {np.std(kimi_accuracies):.2f}%")
    
    if gt_audio_accuracies:
        print(f"   [方案B] GT audio → motion:")
        print(f"      平均准确率: {np.mean(gt_audio_accuracies):.2f}%")
        print(f"      最高准确率: {np.max(gt_audio_accuracies):.2f}%")
        print(f"      最低准确率: {np.min(gt_audio_accuracies):.2f}%")
        print(f"      标准差: {np.std(gt_audio_accuracies):.2f}%")
    
    # 保存汇总
    with open('test_output/summary_report.json', 'w') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 汇总报告已保存: test_output/summary_report.json")

else:
    print("❌ 未找到测试报告")
EOF

echo ""
echo "📁 输出目录: test_output/"
echo ""
echo "💡 提示："
echo "   - 查看汇总报告: cat test_output/summary_report.json"
echo "   - 查看单个样本报告: cat test_output/sample_0/test_report.json"
echo ""
echo "🎬 视频对比："
echo "   1. GT motion (配GT assistant audio): vlc test_output/sample_0/gt_motion.mp4"
echo "   2. Kimi audio → motion (配Kimi生成audio): vlc test_output/sample_0/kimi_audio_motion.mp4"
echo "   3. GT audio → motion (配GT assistant audio): vlc test_output/sample_0/gt_audio_motion.mp4"
echo ""
echo "🔊 音频文件："
echo "   - 用户问题: aplay test_output/sample_0/user_audio.wav"
echo "   - GT 助手回答: aplay test_output/sample_0/assistant_audio.wav"
echo "   - Kimi 生成回答: aplay test_output/sample_0/kimi_generated_audio.wav"


