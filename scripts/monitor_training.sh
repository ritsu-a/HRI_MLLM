#!/bin/bash

# 训练监控脚本
# 使用方法：bash scripts/monitor_training.sh

OUTPUT_DIR="/root/workspace/HRI_MLLM/output/ddp_kimi_lora_adaptor"
LOG_DIR="$OUTPUT_DIR/logs"

echo "🔍 训练监控面板"
echo "="*60

# 查找最新的日志文件
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo "📝 最新日志: $LATEST_LOG"
        echo ""
        
        # 显示选项
        echo "选择监控方式："
        echo "  1) 实时查看日志 (tail -f)"
        echo "  2) 查看最后50行"
        echo "  3) 查看训练总结 (每个epoch的Summary)"
        echo "  4) 查看GPU使用情况"
        echo "  5) 查看loss变化趋势"
        echo "  6) 全部显示"
        echo ""
        
        read -p "请选择 (1-6): " -n 1 -r
        echo ""
        echo ""
        
        case $REPLY in
            1)
                echo "📺 实时监控日志（按Ctrl+C退出）..."
                echo ""
                tail -f "$LATEST_LOG"
                ;;
            2)
                echo "📄 最后50行："
                echo ""
                tail -50 "$LATEST_LOG"
                ;;
            3)
                echo "📊 训练总结："
                echo ""
                grep "Summary" "$LATEST_LOG" || echo "暂无训练总结"
                ;;
            4)
                echo "🖥️  GPU使用情况："
                echo ""
                nvidia-smi
                ;;
            5)
                echo "📈 Loss变化趋势："
                echo ""
                echo "Epoch | Total Loss | Audio Loss | Motion Loss"
                echo "------|-----------|-----------|-------------"
                grep "Summary" "$LATEST_LOG" | while read -r line; do
                    if [[ $line =~ Epoch\ ([0-9]+) ]]; then
                        epoch="${BASH_REMATCH[1]}"
                        total=$(echo "$line" | grep -oP 'Total Loss: \K[0-9.]+' || echo "N/A")
                        audio=$(echo "$line" | grep -oP 'Audio Loss: \K[0-9.]+' || echo "N/A")
                        motion=$(echo "$line" | grep -oP 'Motion Loss: \K[0-9.]+' || echo "N/A")
                        echo "$epoch | $total | $audio | $motion"
                    fi
                done
                ;;
            6)
                echo "📊 完整训练状态："
                echo ""
                echo "1️⃣ 最新checkpoint:"
                ls -lth "$OUTPUT_DIR"/*.pt 2>/dev/null | head -5 || echo "   暂无checkpoint"
                echo ""
                echo "2️⃣ 训练总结:"
                grep "Summary" "$LATEST_LOG" | tail -5 || echo "   暂无总结"
                echo ""
                echo "3️⃣ GPU状态:"
                nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
                echo ""
                echo "4️⃣ 实时日志（最后20行）:"
                tail -20 "$LATEST_LOG"
                ;;
            *)
                echo "❌ 无效选择"
                ;;
        esac
    else
        echo "❌ 未找到日志文件"
        echo "   训练可能还未开始"
    fi
else
    echo "❌ 日志目录不存在: $LOG_DIR"
    echo "   训练可能还未开始"
fi


