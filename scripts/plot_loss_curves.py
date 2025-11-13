"""
绘制训练loss曲线
从日志文件中提取loss数据并可视化
"""

import re
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def parse_log_file(log_file):
    """解析日志文件，提取loss数据"""
    
    epochs = []
    total_losses = []
    audio_losses = []
    motion_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 匹配Summary行
            # [2025-11-11 01:02:30] [INFO] 📊 Epoch 1 Summary:
            if 'Epoch' in line and 'Summary' in line:
                epoch_match = re.search(r'Epoch (\d+) Summary', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
            
            # 提取Total Loss
            if 'Total Loss:' in line:
                loss_match = re.search(r'Total Loss: ([\d.]+)', line)
                if loss_match and current_epoch not in epochs:
                    epochs.append(current_epoch)
                    total_losses.append(float(loss_match.group(1)))
            
            # 提取Audio Loss
            if 'Audio Loss:' in line and current_epoch in epochs:
                loss_match = re.search(r'Audio Loss: ([\d.]+)', line)
                if loss_match and len(audio_losses) < len(epochs):
                    audio_losses.append(float(loss_match.group(1)))
            
            # 提取Motion Loss
            if 'Motion Loss:' in line and current_epoch in epochs:
                loss_match = re.search(r'Motion Loss: ([\d.]+)', line)
                if loss_match and len(motion_losses) < len(epochs):
                    motion_losses.append(float(loss_match.group(1)))
    
    return epochs, total_losses, audio_losses, motion_losses


def plot_losses(epochs, total_losses, audio_losses, motion_losses, output_file='loss_curves.png'):
    """绘制loss曲线"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 总loss
    axes[0, 0].plot(epochs, total_losses, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Audio loss
    axes[0, 1].plot(epochs, audio_losses, 'g-o', linewidth=2, markersize=4)
    axes[0, 1].set_title('Audio Loss (Kimi)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Motion loss
    axes[1, 0].plot(epochs, motion_losses, 'r-o', linewidth=2, markersize=4)
    axes[1, 0].set_title('Motion Loss (Adaptor)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 三条曲线对比
    axes[1, 1].plot(epochs, total_losses, 'b-o', label='Total Loss', linewidth=2, markersize=4)
    axes[1, 1].plot(epochs, audio_losses, 'g-o', label='Audio Loss', linewidth=2, markersize=4)
    axes[1, 1].plot(epochs, motion_losses, 'r-o', label='Motion Loss', linewidth=2, markersize=4)
    axes[1, 1].set_title('All Losses Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curves saved to: {output_file}")
    
    # 打印统计信息
    print(f"\n📊 训练统计:")
    print(f"   Total epochs: {len(epochs)}")
    print(f"   Initial total loss: {total_losses[0]:.4f}")
    print(f"   Final total loss: {total_losses[-1]:.4f}")
    print(f"   Improvement: {total_losses[0] - total_losses[-1]:.4f} ({100*(total_losses[0] - total_losses[-1])/total_losses[0]:.1f}%)")
    print(f"\n   Initial audio loss: {audio_losses[0]:.4f}")
    print(f"   Final audio loss: {audio_losses[-1]:.4f}")
    print(f"   Improvement: {audio_losses[0] - audio_losses[-1]:.4f}")
    print(f"\n   Initial motion loss: {motion_losses[0]:.4f}")
    print(f"   Final motion loss: {motion_losses[-1]:.4f}")
    print(f"   Improvement: {motion_losses[0] - motion_losses[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='绘制训练loss曲线')
    parser.add_argument('--log_file', type=str, default=None,
                      help='日志文件路径（默认使用最新的）')
    parser.add_argument('--output', type=str, default='loss_curves.png',
                      help='输出图片路径')
    
    args = parser.parse_args()
    
    # 查找最新日志
    if args.log_file is None:
        log_dir = Path('/root/workspace/HRI_MLLM/output/ddp_kimi_lora_adaptor/logs')
        if log_dir.exists():
            log_files = list(log_dir.glob('train_*.log'))
            if log_files:
                args.log_file = str(max(log_files, key=lambda p: p.stat().st_mtime))
                print(f"📝 使用最新日志: {args.log_file}")
            else:
                print("❌ 未找到日志文件")
                return
        else:
            print("❌ 日志目录不存在")
            return
    
    # 解析日志
    print(f"📊 解析日志文件...")
    epochs, total_losses, audio_losses, motion_losses = parse_log_file(args.log_file)
    
    if not epochs:
        print("❌ 未找到loss数据，训练可能还未开始或日志格式不正确")
        return
    
    print(f"✅ 找到 {len(epochs)} 个epoch的数据")
    
    # 绘图
    print(f"📈 绘制loss曲线...")
    plot_losses(epochs, total_losses, audio_losses, motion_losses, args.output)


if __name__ == "__main__":
    main()


