"""
VQ-VAE模型评估脚本
用于评估训练好的模型在BEAT和SeG数据集上的重建质量
"""
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from HRI_mllm import ROOT, DATA_ROOT
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.datasets.MixedMotionDatasetVQ import MixedMotionDatasetVQ
import torch.nn.functional as F
from tqdm import tqdm
import argparse

qpos_mask = torch.zeros(491)
qpos_mask[263-17:263] = 1
qpos_mask[491-24:491] = 1

def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)

def collate_fn(batch):
    motions = torch.stack([torch.from_numpy(item[1]).float() for item in batch])
    dataset_indices = torch.tensor([item[6] for item in batch], dtype=torch.long)
    return motions, dataset_indices

def load_evaluation_datasets():
    """加载评估数据集（测试集）"""
    # 加载mean和std
    mean_std_root = os.path.join(DATA_ROOT, "BEAT_v2_kimi")
    mean = np.load(os.path.join(mean_std_root, "Mean.npy"))
    std = np.load(os.path.join(mean_std_root, "Std.npy"))
    
    # BEAT测试集
    beat_test_dataset = MixedMotionDatasetVQ(
        data_root_list=[os.path.join(DATA_ROOT, "BEAT_v2_kimi")],
        split="test",
        mean=mean,
        std=std,
        max_motion_length=196,
        min_motion_length=64,
        win_size=64,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        dataset_weights=[1.0],
        nfeats=491,
        dataset_name="BEAT_test"
    )
    
    # SeG测试集
    seg_test_dataset = MixedMotionDatasetVQ(
        data_root_list=[os.path.join(DATA_ROOT, "SeG_kimi")],
        split="test",
        mean=mean,
        std=std,
        max_motion_length=196,
        min_motion_length=64,
        win_size=64,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        dataset_weights=[1.0],
        nfeats=491,
        dataset_name="SeG_test"
    )
    
    beat_loader = DataLoader(
        beat_test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    seg_loader = DataLoader(
        seg_test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return beat_loader, seg_loader

def evaluate_model(model, dataloader, device, dataset_name=""):
    """
    评估模型在给定数据集上的性能
    Returns:
        metrics: 包含各种评估指标的字典
    """
    model.eval()
    
    total_recon_loss = 0.0
    total_quant_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    
    # 用于统计codebook使用情况
    all_body_codes = []
    all_hand_codes = []
    
    with torch.no_grad():
        for motions, _ in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            motions = motions.to(device)
            
            # 前向传播
            x_out, quant_loss, perplexity = model(motions)
            
            # 计算重建损失
            mask = qpos_mask.to(device)
            recon_loss = F.mse_loss(x_out * mask, motions * mask, reduction='mean')
            
            total_recon_loss += recon_loss.item()
            total_quant_loss += quant_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # 收集codes用于统计
            body_x = motions[:, :, 263-17:263]
            hand_x = motions[:, :, 491-24:491]
            
            (body_code, hand_code), _ = model.encode(motions)
            all_body_codes.append(body_code.cpu())
            all_hand_codes.append(hand_code.cpu())
    
    # 计算平均指标
    metrics = {
        'recon_loss': total_recon_loss / num_batches,
        'quant_loss': total_quant_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
    }
    
    # 计算codebook使用率
    all_body_codes = torch.cat(all_body_codes, dim=0)
    all_hand_codes = torch.cat(all_hand_codes, dim=0)
    
    body_used_codes = len(torch.unique(all_body_codes))
    hand_used_codes = len(torch.unique(all_hand_codes))
    
    # 获取codebook总大小
    if hasattr(model, 'module'):
        code_num = model.module.body_vae.quantizer.nb_code
    else:
        code_num = model.body_vae.quantizer.nb_code
    
    metrics['body_code_usage'] = body_used_codes / code_num
    metrics['hand_code_usage'] = hand_used_codes / code_num
    metrics['avg_code_usage'] = (body_used_codes + hand_used_codes) / (2 * code_num)
    
    return metrics

def print_comparison(beat_metrics, seg_metrics):
    """打印两个数据集的对比结果"""
    print("\n" + "="*70)
    print("                    EVALUATION RESULTS COMPARISON")
    print("="*70)
    print(f"{'Metric':<30} {'BEAT':<20} {'SeG':<20}")
    print("-"*70)
    
    # 重建损失
    print(f"{'Reconstruction Loss':<30} {beat_metrics['recon_loss']:<20.6f} {seg_metrics['recon_loss']:<20.6f}")
    ratio = seg_metrics['recon_loss'] / (beat_metrics['recon_loss'] + 1e-8)
    print(f"{'  SeG/BEAT Ratio':<30} {ratio:<20.4f}")
    
    # 量化损失
    print(f"{'Quantization Loss':<30} {beat_metrics['quant_loss']:<20.6f} {seg_metrics['quant_loss']:<20.6f}")
    
    # Perplexity
    print(f"{'Perplexity':<30} {beat_metrics['perplexity']:<20.2f} {seg_metrics['perplexity']:<20.2f}")
    
    # Codebook使用率
    print(f"{'Codebook Usage (Body)':<30} {beat_metrics['body_code_usage']:<20.2%} {seg_metrics['body_code_usage']:<20.2%}")
    print(f"{'Codebook Usage (Hand)':<30} {beat_metrics['hand_code_usage']:<20.2%} {seg_metrics['hand_code_usage']:<20.2%}")
    print(f"{'Codebook Usage (Avg)':<30} {beat_metrics['avg_code_usage']:<20.2%} {seg_metrics['avg_code_usage']:<20.2%}")
    
    print("="*70)
    
    # 质量评估
    print("\n" + "="*70)
    print("                    QUALITY ASSESSMENT")
    print("="*70)
    
    if seg_metrics['recon_loss'] < 0.01:
        seg_quality = "EXCELLENT"
    elif seg_metrics['recon_loss'] < 0.03:
        seg_quality = "GOOD"
    elif seg_metrics['recon_loss'] < 0.05:
        seg_quality = "ACCEPTABLE"
    else:
        seg_quality = "POOR"
    
    if beat_metrics['recon_loss'] < 0.03:
        beat_quality = "EXCELLENT"
    elif beat_metrics['recon_loss'] < 0.05:
        beat_quality = "GOOD"
    elif beat_metrics['recon_loss'] < 0.08:
        beat_quality = "ACCEPTABLE"
    else:
        beat_quality = "POOR"
    
    print(f"SeG Reconstruction Quality:  {seg_quality}")
    print(f"BEAT Reconstruction Quality: {beat_quality}")
    
    if ratio < 0.5:
        print(f"\n✓ SeG reconstruction is significantly better than BEAT (ratio={ratio:.3f})")
    elif ratio < 0.8:
        print(f"\n⚠ SeG reconstruction is moderately better than BEAT (ratio={ratio:.3f})")
    else:
        print(f"\n✗ SeG reconstruction is NOT significantly better than BEAT (ratio={ratio:.3f})")
    
    if beat_metrics['avg_code_usage'] > 0.3 and seg_metrics['avg_code_usage'] > 0.3:
        print("✓ Good codebook utilization (>30%)")
    else:
        print("⚠ Low codebook utilization - consider adjusting beta or codebook size")
    
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate VQ-VAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='HRI_mllm/model/motion_encoder/g1_vqvae_full_qpos.yaml',
                       help='Path to model config')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置
    config = open_yaml(args.config)
    print(f"\nLoaded config from {args.config}")
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}")
    model = VQVaeBodyHand(**config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Model loaded successfully")
    
    # 加载数据集
    print("\nLoading evaluation datasets...")
    beat_loader, seg_loader = load_evaluation_datasets()
    print(f"BEAT test set: {len(beat_loader.dataset)} samples")
    print(f"SeG test set: {len(seg_loader.dataset)} samples")
    
    # 评估模型
    print("\n" + "="*70)
    print("Starting evaluation...")
    print("="*70 + "\n")
    
    beat_metrics = evaluate_model(model, beat_loader, device, "BEAT")
    seg_metrics = evaluate_model(model, seg_loader, device, "SeG")
    
    # 打印对比结果
    print_comparison(beat_metrics, seg_metrics)
    
    # 保存结果
    results = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'beat_metrics': beat_metrics,
        'seg_metrics': seg_metrics,
    }
    
    output_path = args.checkpoint.replace('.pt', '_evaluation.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(results, f)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

