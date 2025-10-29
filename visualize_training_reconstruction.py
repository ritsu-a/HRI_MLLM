#!/usr/bin/env python3
"""
可视化训练json里动作序列重建效果的代码
使用训练完成的GPT2 adaptor模型生成motion tokens，然后通过VQ-VAE解码为动作序列
"""

import os
import argparse
import json
import pickle
import numpy as np
import torch
import shutil
import yaml
from pathlib import Path
import subprocess

# Set MuJoCo to use EGL rendering (headless)
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats, feats2datapkl
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion
from HRI_mllm import ROOT, DATA_ROOT

# GPT2 adaptor相关导入
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from types import SimpleNamespace


def open_yaml(path):
    """打开YAML配置文件"""
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def load_gpt2_model(model_path, device="cuda"):
    """加载训练完成的GPT2 adaptor模型
    
    支持两种格式：
    1. transformers格式目录（包含config.json和pytorch_model.bin）
    2. checkpoint文件（.pt格式，包含model_state）
    """
    print(f"Loading GPT2 adaptor model from: {model_path}")
    
    # 检查是否是.pt checkpoint文件
    if model_path.endswith('.pt'):
        return load_gpt2_from_checkpoint(model_path, device)
    else:
        return load_gpt2_from_transformers(model_path, device)


def load_gpt2_from_checkpoint(checkpoint_path, device="cuda"):
    """从.pt checkpoint文件加载模型"""
    print(f"🔄 Loading from checkpoint: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # 从checkpoint中获取epoch信息
    epoch = checkpoint.get('epoch', 0)
    print(f"📊 Checkpoint epoch: {epoch}")
    
    # 创建模型配置（使用训练时的默认配置）
    model_config = GPT2Config(
        vocab_size=1034,  # 512*2 + 10 (motion_vocab_size + pad_token)
        n_positions=4096,  # max_seq_length
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    # 创建模型实例
    model = MixedInputGPT2(model_config, audio_hidden_size=3584)
    
    # 加载模型权重
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print("✅ Loaded model_state from checkpoint")
    else:
        print("⚠️  model_state not found in checkpoint, using random weights")
    
    model.eval()
    model.to(device)
    print(f"✅ GPT2 adaptor model loaded successfully from checkpoint!")
    return model


def load_gpt2_from_transformers(model_path, device="cuda"):
    """从transformers格式目录加载模型"""
    print(f"🔄 Loading from transformers format: {model_path}")
    
    # 加载配置
    config_path = os.path.join(model_path, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    
    # 创建模型实例
    model = MixedInputGPT2(config, audio_hidden_size=3584)
    
    # 加载模型权重
    model_path_pytorch = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(model_path_pytorch):
        state_dict = torch.load(model_path_pytorch, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pytorch_model.bin")
    else:
        print("⚠️  pytorch_model.bin not found, using random weights")
    
    model.eval()
    model.to(device)
    print(f"✅ GPT2 adaptor model loaded successfully from transformers format!")
    return model


def generate_motion_tokens(model, audio_tokens, device="cuda", max_new_tokens=256, 
                         temperature=0.8, top_k=50, repetition_penalty=1.1, 
                         teacher_forcing=False, ground_truth_tokens=None):
    """
    使用GPT2 adaptor模型生成motion tokens
    参考test_gpt2_adaptor.py的实现方式
    
    Args:
        model: GPT2 adaptor模型
        audio_tokens: 音频token序列
        device: 设备
        max_new_tokens: 最大新生成token数量
        temperature: 温度参数
        top_k: top-k采样
        repetition_penalty: 重复惩罚
        teacher_forcing: 是否使用teacher forcing模式
        ground_truth_tokens: ground truth motion tokens（teacher forcing时使用）
    
    Returns:
        generated_motion_tokens: 生成的motion token序列
        loss: 计算的loss值（teacher forcing模式）
    """
    model.eval()
    
    print(f"Input audio tokens length: {len(audio_tokens)}")
    print(f"Audio tokens range: [{min(audio_tokens)}, {max(audio_tokens)}]")
    
    if teacher_forcing and ground_truth_tokens is not None:
        print(f"🎓 Using teacher forcing mode with {len(ground_truth_tokens)} ground truth tokens")
        return _teacher_forcing_generation(model, audio_tokens, ground_truth_tokens, device)
    else:
        print(f"🎲 Using free generation mode")
        return _free_generation(model, audio_tokens, max_new_tokens, temperature, top_k, repetition_penalty, device)


def _teacher_forcing_generation(model, audio_tokens, ground_truth_tokens, device):
    """
    正确的Teacher Forcing模式：
    1. 使用与训练时一致的方式：一次性处理完整序列
    2. 使用模型的内部loss计算
    3. 只对motion token计算loss
    """
    # 按照训练时的穿插模式构建完整序列：audio[0], motion[0], audio[1], motion[1], ...
    full_sequence = []
    labels = []
    masks = []  # 1表示motion token，0表示audio token
    
    interleave_audios, interleave_motions = 1, 1  # 从训练配置中获取
    
    for i in range(len(audio_tokens)):
        # 添加audio token
        full_sequence.append(audio_tokens[i])
        labels.append(-100)  # audio token的label为-100
        masks.append(0)  # audio token的mask为0
        
        # 每interleave_audios个audio token后，添加interleave_motions个motion token
        if (i + 1) % interleave_audios == 0:
            motion_idx = i // interleave_audios * interleave_motions
            for j in range(interleave_motions):
                if motion_idx + j < len(ground_truth_tokens):
                    # 确保motion token在模型vocab_size范围内 [0, 1025]
                    motion_token = ground_truth_tokens[motion_idx + j]
                    motion_token = min(motion_token, 1025)  # 限制在vocab_size范围内
                    full_sequence.append(motion_token)
                    labels.append(motion_token)  # motion token的label为自身
                    masks.append(1)  # motion token的mask为1
    
    # 转换为tensor
    inputs = torch.tensor(full_sequence).unsqueeze(0).to(device)
    labels_tensor = torch.tensor(labels).unsqueeze(0).to(device)
    masks_tensor = torch.tensor(masks).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(inputs)
    
    print(f"🎓 Teacher forcing input shape: {inputs.shape}")
    print(f"🎓 Teacher forcing labels shape: {labels_tensor.shape}")
    print(f"🎓 Teacher forcing masks shape: {masks_tensor.shape}")
    print(f"🎓 Motion tokens: {masks_tensor.sum().item()}/{masks_tensor.numel()}")
    
    # 使用与训练时一致的方式计算loss
    with torch.no_grad():
        # 调用模型，使用模型的内部loss计算
        output = model(input_data=inputs, attention_mask=attention_mask, labels=labels_tensor)
        
        # 使用模型计算的loss（与训练时一致）
        # 模型的loss计算会自动忽略label为-100的位置，只计算motion token的loss
        loss = output.loss.item() if output.loss is not None else 0.0
    
    print(f"🎓 Teacher forcing loss (motion tokens only): {loss:.6f}")
    
    # 提取生成的motion tokens（模型预测的）
    generated_motion_tokens = []
    motion_token_count = 0
    
    for i, token in enumerate(full_sequence):
        if masks[i] == 1:  # 如果是motion token
            # 获取模型对该位置的预测
            with torch.no_grad():
                # 使用到当前位置的序列作为输入
                partial_inputs = torch.tensor(full_sequence[:i+1]).unsqueeze(0).to(device)
                partial_attn = torch.ones_like(partial_inputs)
                partial_labels = torch.tensor(labels[:i+1]).unsqueeze(0).to(device)
                
                output = model(input_data=partial_inputs, attention_mask=partial_attn, labels=partial_labels)
                predicted_token = torch.argmax(output.logits[0, -1, :]).item()
                
                # 确保motion token在正确范围内
                if motion_token_count % 2 == 0:
                    # body token [0, 511]
                    predicted_token = int(predicted_token % 512)
                else:
                    # hand token [512, 1023]
                    predicted_token = int(512 + (predicted_token % 512))
                
                generated_motion_tokens.append(predicted_token)
                motion_token_count += 1
    
    print(f"🎓 Generated {len(generated_motion_tokens)} motion tokens")
    if generated_motion_tokens:
        print(f"🎓 Generated tokens range: [{min(generated_motion_tokens)}, {max(generated_motion_tokens)}]")
    
    return generated_motion_tokens, loss


def _free_generation(model, audio_tokens, max_new_tokens, temperature, top_k, repetition_penalty, device):
    """
    自由生成模式：audio tokens作为teacher-forcing，只生成motion tokens
    参考teacher-forcing模式的实现，但motion tokens是生成的而不是ground truth
    """
    generated_motion_tokens = []
    current_seq = []
    token_labels = []
    generated_history = []
    
    interleave_audios, interleave_motions = 1, 1  # 从训练配置中获取
    code_num = 512  # motion token vocabulary size
    motion_token_count = 0  # 总的motion token计数
    
    with torch.no_grad():
        for i, audio_token in enumerate(audio_tokens):
            # 添加audio token（teacher-forcing）
            current_seq.append(audio_token)
            token_labels.append(-100)  # audio token对应的label为-100
            
            # 每interleave_audios个audio token后，生成interleave_motions个motion token
            if (i + 1) % interleave_audios == 0:
                motion_idx = i // interleave_audios * interleave_motions
                
                for j in range(interleave_motions):
                    # 准备输入
                    inputs = torch.tensor(current_seq).unsqueeze(0).to(device)
                    attn_mask = torch.ones_like(inputs)
                    labels = torch.tensor(token_labels).unsqueeze(0).to(device)
                    
                    # 调用模型forward方法
                    output = model(input_data=inputs, attention_mask=attn_mask, labels=labels)
                    next_token_logits = output.logits[0, -1, :]
                    
                    # 应用重复惩罚
                    if repetition_penalty != 1.0 and generated_history:
                        for token_id in set(generated_history):
                            if token_id < next_token_logits.size(-1):  # 防止索引越界
                                next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                    
                    # 应用temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k采样
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits[top_k_indices] = top_k_logits
                    
                    # 采样
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # 确保motion token在正确范围内
                    # 偶数次 motion（body）∈ [0, 511]；奇数次 motion（hand）∈ [512, 1023]
                    if motion_token_count % 2 == 0:
                        # body token [0, 511]
                        next_token = int(next_token % 512)
                    else:
                        # hand token [512, 1023]
                        next_token = int(512 + (next_token % 512))
                    
                    motion_token_count += 1
                    
                    generated_motion_tokens.append(next_token)
                    generated_history.append(next_token)
                    
                    # 保持历史记录长度
                    if len(generated_history) > 100:
                        generated_history = generated_history[-100:]
                    
                    # 添加到当前序列
                    current_seq.append(next_token)
                    token_labels.append(next_token)  # motion token对应的label为自身
                    
                    # 检查是否达到最大motion token数量
                    if len(generated_motion_tokens) >= max_new_tokens:
                        break
                
                # 检查是否达到最大motion token数量
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    print(f"Generated motion tokens length: {len(generated_motion_tokens)}")
    if generated_motion_tokens:
        print(f"Generated motion tokens range: [{min(generated_motion_tokens)}, {max(generated_motion_tokens)}]")
    
    return generated_motion_tokens, None


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """
    解码motion tokens为motion features
    
    Args:
        motion_tokens: List of motion tokens [body1, hand1, body2, hand2, ...]
        motion_vae: The motion VQ-VAE model
        mean_t: Mean tensor for denormalization
        std_t: Std tensor for denormalization
        expected_frames: Expected number of motion frames (optional, used for truncation)
    
    Returns:
        data_dict: Decoded motion data in pkl format
    """
    # Separate body and hand tokens
    # motion_tokens are alternating: [body, hand, body, hand, ...]
    if len(motion_tokens) % 2 != 0:
        print(f"Warning: motion_tokens length ({len(motion_tokens)}) is odd, dropping last token")
        motion_tokens = motion_tokens[:-1]
    
    # 检查是否有足够的tokens
    if len(motion_tokens) == 0:
        print("Error: No motion tokens after processing")
        return None
    
    body_tokens = motion_tokens[0::2]  # even indices
    hand_tokens = motion_tokens[1::2]  # odd indices
    
    # Convert to tensors
    body_tokens = torch.tensor(body_tokens).unsqueeze(0).to("cuda")
    # hand tokens need to subtract 512 offset
    hand_tokens = torch.tensor(hand_tokens).unsqueeze(0).to("cuda") - 512
    
    print(f"Body tokens shape: {body_tokens.shape}, range: [{body_tokens.min()}, {body_tokens.max()}]")
    print(f"Hand tokens shape: {hand_tokens.shape}, range: [{hand_tokens.min()}, {hand_tokens.max()}]")
    
    # Decode using VAE - this returns normalized features
    decoded = motion_vae.decode((body_tokens, hand_tokens))
    
    print(f"Decoded shape before truncation: {decoded.shape}")
    
    # If expected_frames is provided and decoded is longer, truncate to expected length
    if expected_frames is not None and decoded.shape[1] > expected_frames:
        decoded = decoded[:, :expected_frames, :]
        print(f"Truncated to expected length: {decoded.shape}")
    
    # Convert to data format (feats2datapkl will handle denormalization with provided stats)
    data_dict = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())
    
    return data_dict


def concat_side_by_side(video_left: str, video_right: str, output_path: str, target_height: int = 720):
    """使用 ffmpeg 横向拼接两个视频，统一高度为 target_height，保持宽高比。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_left,
        "-i", video_right,
        "-filter_complex",
        f"[0:v]scale=-2:{target_height},setsar=1[left];[1:v]scale=-2:{target_height},setsar=1[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",  # 使用左视频音频（若存在）
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='可视化训练json里动作序列重建效果')
    parser.add_argument('--jsonl_path', type=str, default='/root/workspace/HRI_MLLM/data/BEAT_v2_kimi_tokens.jsonl',
                       help='Path to the jsonl file containing tokens')
    parser.add_argument('--gpt2_model_path', type=str, 
                       default='/root/workspace/HRI_MLLM/output/motion_adaptor_v3/kimi_audio_motion_gpt2_brainco_30_100/transformers_format',
                       help='Path to the trained GPT2 adaptor model. Supports two formats: 1) transformers format directory (with config.json and pytorch_model.bin), 2) checkpoint file (.pt format)')
    parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                       help='VQ-VAE config file name')
    parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                       help='VQ-VAE checkpoint path. If not provided, will use the one in config file')
    parser.add_argument('--output_dir', type=str, default='./training_reconstruction_visualization',
                       help='Output directory for visualization files')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to process')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Maximum number of new tokens to generate for motion tokens')
    parser.add_argument('--temperature', type=float, default=1.8,
                       help='Temperature for generation')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.8,
                       help='Repetition penalty parameter')
    parser.add_argument('--teacher_forcing', action='store_true',
                       help='Use teacher forcing mode to calculate loss')
    parser.add_argument('--compare_modes', action='store_true',
                       help='Compare both teacher forcing and free generation modes')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GPT2 adaptor model
    gpt2_model = load_gpt2_model(args.gpt2_model_path)

    # Load VQ-VAE configuration
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
    print(f"Loading VQ-VAE config from: {config_path}")
    motion_config = open_yaml(config_path)

    # Determine checkpoint path
    if args.vqvae_checkpoint:
        checkpoint_path = args.vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
        print(f"⚠️  No checkpoint specified in config, using default: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit(1)

    print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")

    # Load normalization statistics
    test_mean, test_std = load_normalization_stats(motion_config)
    mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
    std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
    print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")

    # Load the VAE model
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device="cuda")
    print(f"✅ VQ-VAE model loaded successfully!")

    # Load and process samples from jsonl
    print(f"Reading samples from: {args.jsonl_path}")
    with open(args.jsonl_path, 'r') as f:
        lines = f.readlines()

    processed_count = 0
    for i, line in enumerate(lines[:args.num_samples]):
        try:
            data = json.loads(line.strip())
            
            # Extract audio path, audio tokens, and ground truth motion tokens
            audio_path = None
            motion_tokens_gt = None
            audio_tokens = None

            for msg in data['conversation']:
                if msg.get('message_type') == 'audio' and msg.get('audio_tokens'):
                    audio_path = msg['content']
                    audio_tokens = msg['audio_tokens']
                
                if msg.get('message_type') == 'audio_motion' and msg.get('motion_tokens'):
                    motion_tokens_gt = msg['motion_tokens']

            if motion_tokens_gt is None or audio_tokens is None:
                print(f"Skipping sample {i+1}: Missing tokens")
                continue

            print(f"\n{'='*60}")
            print(f"Processing sample {i+1}/{min(len(lines), args.num_samples)}")
            print(f"Audio path: {audio_path}")
            print(f"Audio tokens length: {len(audio_tokens)}")
            print(f"Ground truth motion tokens length: {len(motion_tokens_gt)}")

            # Create sample output directory
            sample_dir = os.path.join(args.output_dir, f"sample_{i+1}")
            os.makedirs(sample_dir, exist_ok=True)

            # Generate motion tokens using GPT2 adaptor
            print("\nGenerating motion tokens with GPT2 adaptor...")
            
            if args.compare_modes:
                print("\n🔄 Comparing teacher forcing vs free generation modes...")
                
                # Teacher forcing mode
                print("\n🎓 Teacher Forcing Mode:")
                tf_tokens, tf_loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=True,
                    ground_truth_tokens=motion_tokens_gt
                )
                
                # Free generation mode
                print("\n🎲 Free Generation Mode:")
                fg_tokens, fg_loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=False
                )
                
                print(f"\n📊 Results Comparison:")
                print(f"   Teacher Forcing Loss: {tf_loss:.6f}")
                print(f"   Free Generation Loss: {fg_loss if fg_loss else 'N/A'}")
                print(f"   Expected Training Loss: ~0.01")
                
                if tf_loss is not None:
                    if abs(tf_loss - 0.01) < 0.005:
                        print(f"   ✅ Teacher forcing loss is close to training loss!")
                    else:
                        print(f"   ⚠️  Teacher forcing loss differs from training loss")
                
                # Use free generation tokens for visualization
                generated_motion_tokens = fg_tokens
                
            elif args.teacher_forcing:
                print("\n🎓 Teacher Forcing Mode:")
                generated_motion_tokens, loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=True,
                    ground_truth_tokens=motion_tokens_gt
                )
                
                print(f"\n📊 Teacher Forcing Results:")
                print(f"   Loss: {loss:.6f}")
                print(f"   Expected Training Loss: ~0.01")
                
                if loss is not None:
                    if abs(loss - 0.01) < 0.005:
                        print(f"   ✅ Loss is close to training loss!")
                    else:
                        print(f"   ⚠️  Loss differs from training loss")
                
            else:
                print("\n🎲 Free Generation Mode:")
                generated_motion_tokens, loss = generate_motion_tokens(
                    gpt2_model, 
                    audio_tokens, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    teacher_forcing=False
                )

            # Decode generated motion tokens
            print("\nDecoding generated motion tokens...")
            generated_motion_pkl = decode_motion_tokens(
                generated_motion_tokens, motion_vae, mean_t, std_t
            )

            # Save generated motion
            generated_pkl_path = os.path.join(sample_dir, "generated_motion.pkl")
            with open(generated_pkl_path, 'wb') as f:
                pickle.dump(generated_motion_pkl, f)
            print(f"Saved generated motion pkl to: {generated_pkl_path}")

            # Convert to CSV
            generated_csv = load_motion_pkl_as_csv_data(generated_pkl_path)
            generated_csv_path = os.path.join(sample_dir, "generated_motion.csv")
            np.savetxt(generated_csv_path, generated_csv, delimiter=',', fmt='%.8f')
            print(f"Saved generated motion CSV to: {generated_csv_path}")

            # Decode ground truth motion tokens
            print("\nDecoding ground truth motion tokens...")
            gt_motion_pkl = decode_motion_tokens(
                motion_tokens_gt, motion_vae, mean_t, std_t
            )

            # Save ground truth motion
            gt_pkl_path = os.path.join(sample_dir, "ground_truth_motion.pkl")
            with open(gt_pkl_path, 'wb') as f:
                pickle.dump(gt_motion_pkl, f)

            # Convert to CSV
            gt_csv_path = os.path.join(sample_dir, "ground_truth_motion.csv")
            gt_csv = load_motion_pkl_as_csv_data(gt_pkl_path)
            np.savetxt(gt_csv_path, gt_csv, delimiter=',', fmt='%.8f')
            print(f"Saved ground truth motion CSV to: {gt_csv_path}")

            # Visualize generated motion
            print("\nCreating visualization...")
            generated_video_path = os.path.join(sample_dir, "generated_motion.mp4")
            vis_audio_motion(
                generated_csv_path,
                output_path=generated_video_path,
                audio_path=audio_path,
                robot_type="g1_brainco",
                rate_limit=False
            )
            print(f"✅ Generated motion visualization saved to: {generated_video_path}")

            # Visualize ground truth motion
            gt_video_path = os.path.join(sample_dir, "ground_truth_motion.mp4")
            vis_audio_motion(
                gt_csv_path,
                output_path=gt_video_path,
                audio_path=audio_path,
                robot_type="g1_brainco",
                rate_limit=False
            )
            print(f"✅ Ground truth motion visualization saved to: {gt_video_path}")

            # Create side-by-side comparison
            comparison_video_path = os.path.join(sample_dir, "comparison.mp4")
            concat_side_by_side(gt_video_path, generated_video_path, comparison_video_path, target_height=720)
            print(f"✅ Comparison video saved to: {comparison_video_path}")

            processed_count += 1
            print(f"✅ Sample {i+1} processed successfully!")

        except Exception as e:
            print(f"❌ Error processing sample {i+1}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print(f"🎉 Processing completed!")
    print(f"Successfully processed: {processed_count}/{min(len(lines), args.num_samples)} samples")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
