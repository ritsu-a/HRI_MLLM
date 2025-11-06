#!/usr/bin/env python3
"""
统一Kimi-Motion模型测试脚本
测试训练好的模型生成motion tokens
"""

import os
import argparse
import torch
import numpy as np
import json
from pathlib import Path
import soundfile as sf

# 设置MuJoCo渲染
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT
from HRI_mllm.model.unified_kimi_motion_model import UnifiedKimiMotionModel
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats, feats2datapkl
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion
from kimia_infer.api.kimia import KimiAudio
from kimia_infer.api.prompt_manager import KimiAPromptManager
from transformers import AutoConfig
from huggingface_hub import snapshot_download
import yaml
import pickle


def load_test_model(model_path: str, device: str = "cuda"):
    """加载测试模型"""
    print(f"🔄 Loading unified model from: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从检查点中恢复配置
    kimi_model_path = checkpoint.get('kimi_model_path', "moonshotai/Kimi-Audio-7B")
    freeze_kimi = checkpoint.get('freeze_kimi', True)
    freeze_adaptor = checkpoint.get('freeze_adaptor', True)
    train_mixer_only = checkpoint.get('train_mixer_only', True)
    
    # 创建GPT2配置
    gpt2_config = torch.load("output/motion_adaptor_10_v4/kimi_audio_motion_gpt2_brainco_30_100/checkpoints/epoch_500.pt")['model_state']
    # 这里需要从adaptor的state_dict中推断配置
    # 简化版本：使用默认配置
    from transformers import GPT2Config
    gpt2_config = GPT2Config(
        vocab_size=1034,
        n_positions=4096,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    # 从checkpoint中获取mixer配置（如果有的话）
    mixer_intermediate_size = checkpoint.get('mixer_intermediate_size', None)
    mixer_num_layers = checkpoint.get('mixer_num_layers', 2)
    
    # 创建模型实例
    model = UnifiedKimiMotionModel(
        kimi_model_path=kimi_model_path,
        gpt2_config=gpt2_config,
        freeze_kimi=freeze_kimi,
        freeze_adaptor=freeze_adaptor,
        train_mixer_only=train_mixer_only,
        mixer_intermediate_size=mixer_intermediate_size,
        mixer_num_layers=mixer_num_layers
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   - Step: {checkpoint.get('step', 'unknown')}")
    print(f"   - Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model


def load_motion_vae(vqvae_config: str = "g1_vqvae_arbitrary_length_balanced.yaml", 
                   vqvae_checkpoint: str = None):
    """加载motion VQ-VAE模型"""
    print(f"🔄 Loading motion VQ-VAE...")
    
    # 加载配置文件
    config_path = os.path.join(ROOT, "model", "motion_encoder", vqvae_config)
    with open(config_path, 'r', encoding="utf-8") as file:
        motion_config = yaml.safe_load(file)
    
    # 确定checkpoint路径
    if vqvae_checkpoint:
        checkpoint_path = vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")
    
    # 加载归一化统计量
    test_mean, test_std = load_normalization_stats(motion_config)
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
    std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
    
    # 加载模型
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to("cuda")
    
    print(f"✅ Motion VQ-VAE loaded successfully!")
    return motion_vae, mean_t, std_t


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """解码motion tokens为motion features"""
    # 分离body和hand tokens
    if len(motion_tokens) % 2 != 0:
        print(f"Warning: motion_tokens length ({len(motion_tokens)}) is odd, dropping last token")
        motion_tokens = motion_tokens[:-1]
    
    if len(motion_tokens) == 0:
        print("Error: No motion tokens after processing")
        return None
    
    body_tokens = motion_tokens[0::2]  # even indices
    hand_tokens = motion_tokens[1::2]  # odd indices
    
    # 转换为tensors
    body_tokens = torch.tensor(body_tokens).unsqueeze(0).to("cuda")
    hand_tokens = torch.tensor(hand_tokens).unsqueeze(0).to("cuda") - 512
    
    print(f"Body tokens shape: {body_tokens.shape}, range: [{body_tokens.min()}, {body_tokens.max()}]")
    print(f"Hand tokens shape: {hand_tokens.shape}, range: [{hand_tokens.min()}, {hand_tokens.max()}]")
    
    # 解码
    decoded = motion_vae.decode((body_tokens, hand_tokens))
    
    print(f"Decoded shape before truncation: {decoded.shape}")
    
    # 截断到期望长度
    if expected_frames is not None and decoded.shape[1] > expected_frames:
        decoded = decoded[:, :expected_frames, :]
        print(f"Truncated to expected length: {decoded.shape}")
    
    # 转换为数据格式
    data_dict = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())
    
    return data_dict


def test_model_generation(model, audio_path: str, motion_vae, mean_t, std_t, 
                         max_new_tokens: int = 1024, temperature: float = 0.8):
    """测试模型生成motion tokens"""
    print(f"🎵 Processing audio: {audio_path}")
    
    # 加载音频并获取tokens
    cache_path = snapshot_download("moonshotai/Kimi-Audio-7B")
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    
    prompt_manager = KimiAPromptManager(
        model_path=cache_path, 
        kimia_token_offset=model_config.kimia_token_offset, 
        kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
    )
    
    audio_tokens = torch.from_numpy(
        np.array(prompt_manager._tokenize_audio(audio_path))
    ).unsqueeze(0)
    
    print(f"Audio tokens shape: {audio_tokens.shape}")
    print(f"Audio tokens range: [{audio_tokens.min()}, {audio_tokens.max()}]")
    
    # 生成motion tokens
    print("🔄 Generating motion tokens...")
    generated_motion_tokens = model.generate_motion_tokens(
        audio_input_ids=audio_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        repetition_penalty=1.1
    )
    
    print(f"Generated motion tokens length: {len(generated_motion_tokens)}")
    if generated_motion_tokens:
        print(f"Generated motion tokens range: [{min(generated_motion_tokens)}, {max(generated_motion_tokens)}]")
        print(f"Body tokens: {len(generated_motion_tokens[0::2])}")
        print(f"Hand tokens: {len(generated_motion_tokens[1::2])}")
        print(f"Body token range: [{min(generated_motion_tokens[0::2])}, {max(generated_motion_tokens[0::2])}]")
        print(f"Hand token range: [{min(generated_motion_tokens[1::2])}, {max(generated_motion_tokens[1::2])}]")
    
    # 解码motion tokens
    print("🔄 Decoding motion tokens...")
    motion_pkl = decode_motion_tokens(generated_motion_tokens, motion_vae, mean_t, std_t)
    
    if motion_pkl is None:
        print("❌ Failed to decode motion tokens")
        return None, None
    
    return motion_pkl, generated_motion_tokens


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test Unified Kimi-Motion Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的统一模型路径')
    parser.add_argument('--audio_path', type=str, 
                       default="/root/workspace/HRI_MLLM/data/beat_english_v0.2.1/1/1_wayne_0_1_1_qwen1.wav",
                       help='测试音频文件路径')
    parser.add_argument('--output_dir', type=str, default="test_output",
                       help='输出目录')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                       help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数')
    parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='VQ-VAE配置文件')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                       help='VQ-VAE检查点路径')
    
    args = parser.parse_args()
    
    print("🧪 Starting Unified Kimi-Motion Model Test")
    print(f"📋 Arguments: {vars(args)}")
    
    # 检查输入文件
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.audio_path):
        print(f"❌ Audio file not found: {args.audio_path}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_test_model(args.model_path)
    
    # 加载motion VQ-VAE
    motion_vae, mean_t, std_t = load_motion_vae(args.vqvae_config, args.vqvae_checkpoint)
    
    # 测试生成
    motion_pkl, motion_tokens = test_model_generation(
        model, args.audio_path, motion_vae, mean_t, std_t,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    if motion_pkl is None:
        print("❌ Test failed!")
        return
    
    # 保存结果
    print("💾 Saving results...")
    
    # 保存motion pkl
    pkl_path = os.path.join(args.output_dir, "generated_motion.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(motion_pkl, f)
    
    # 转换为CSV
    csv_path = os.path.join(args.output_dir, "generated_motion.csv")
    motion_csv = load_motion_pkl_as_csv_data(pkl_path)
    np.savetxt(csv_path, motion_csv, delimiter=',', fmt='%.8f')
    
    # 复制音频文件
    audio_output_path = os.path.join(args.output_dir, "audio.wav")
    import shutil
    shutil.copyfile(args.audio_path, audio_output_path)
    
    # 生成可视化视频
    video_path = os.path.join(args.output_dir, "generated_motion.mp4")
    print("🎬 Generating visualization video...")
    vis_audio_motion(
        csv_path, 
        output_path=video_path, 
        audio_path=audio_output_path, 
        robot_type="g1_brainco", 
        rate_limit=False, 
        motion_fps=25
    )
    
    # 保存motion tokens
    tokens_path = os.path.join(args.output_dir, "motion_tokens.json")
    with open(tokens_path, 'w') as f:
        json.dump(motion_tokens, f)
    
    print(f"\n🎉 Test completed successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"   - Motion pkl: {pkl_path}")
    print(f"   - Motion csv: {csv_path}")
    print(f"   - Motion tokens: {tokens_path}")
    print(f"   - Audio: {audio_output_path}")
    print(f"   - Video: {video_path}")
    
    # 打印统计信息
    print(f"\n📊 Generation Statistics:")
    print(f"   - Generated motion tokens: {len(motion_tokens)}")
    print(f"   - Body tokens: {len(motion_tokens[0::2])}")
    print(f"   - Hand tokens: {len(motion_tokens[1::2])}")
    print(f"   - Body token range: [{min(motion_tokens[0::2])}, {max(motion_tokens[0::2])}]")
    print(f"   - Hand token range: [{min(motion_tokens[1::2])}, {max(motion_tokens[1::2])}]")


if __name__ == "__main__":
    main()
