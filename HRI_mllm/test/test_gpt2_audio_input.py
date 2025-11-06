### streaming demo for qwen2_5omni_motion with direct audio input
### Based on test_gpt2_text_response.py, modified to accept direct audio file input
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

# Set MuJoCo to use EGL rendering (headless)
import os
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenMotionAdaptorStreamer
from HRI_mllm import ROOT

from HRI_mllm.utils.motion_utils.g1ml3d_final import vec_to_data_pkl, feats2datapkl
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats
from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand

from kimia_infer.api.kimia import KimiAudio


from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion

import numpy as np


import yaml
import sys
import pickle

from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig
from types import SimpleNamespace
import soundfile as sf

from typing import Optional, Dict
from dotenv import load_dotenv
from tts.comm_funcs.tts import Factory as TTS_Factory #TTS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pydub import AudioSegment

import torch
torch.cuda.set_device(0)


# 直接使用音频文件作为输入
audio_input_path = "/root/workspace/HRI_MLLM/data/beat_english_v0.2.1/1/1_wayne_0_1_1.wav"
user_text_instruction = "Please repeat the following spoken content."


def tts(text, save_path):
    load_dotenv()
    say = text
    tts = TTS_Factory.get_tts(os.getenv("TTS_TYPE"))  #TTS
    # tts.play(say)
    audio_bytes = tts.to_audio(say, async_play = False)

    if audio_bytes:
       # 用 pydub 直接封装成 WAV；参数从 handler 读取，避免硬编码
        pcm = AudioSegment(
            audio_bytes,
            sample_width=tts.handler.audio.get_sample_size(tts.handler.format),
            frame_rate=tts.handler.rate,
            channels=tts.handler.channels
        )
        pcm.export(save_path, format="wav")
        print(f"✅ 已保存:{save_path}")
    else:
        print("⚠️ 生成音频失败: audio_bytes 为空")


### loading kimi
kimi_model = KimiAudio(
    model_path="/DATA/disk1/Kimi-Audio-7B-Instruct",
    load_detokenizer=True,
)

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}






def generate_motion_tokens(model, audio_tokens, device="cuda", max_new_tokens=256, 
                         temperature=0.8, top_k=50, repetition_penalty=1.1):
    """
    使用GPT2 adaptor模型生成motion tokens
    支持special token的处理，确保训练与测试结构一致
    
    Args:
        model: GPT2 adaptor模型
        audio_tokens: 音频token序列（不包含special token）
        device: 设备
        max_new_tokens: 最大新生成token数量
        temperature: 温度参数
        top_k: top-k采样
        repetition_penalty: 重复惩罚
    
    Returns:
        generated_motion_tokens: 生成的motion token序列（已过滤special token，可直接用于VQ-VAE）
    """
    model.eval()
    
    # 获取special token ID（从模型配置中）
    gesture_start_token_id = getattr(model.config, 'gesture_start_token_id', 512*2 + 2)
    audio_gesture_start_token_id = getattr(model.config, 'audio_gesture_start_token_id', 512*2 + 3)
    gesture_end_token_id = getattr(model.config, 'gesture_end_token_id', 512*2 + 4)
    audio_gesture_end_token_id = getattr(model.config, 'audio_gesture_end_token_id', 512*2 + 5)
    
    # 定义所有special token ID
    special_token_ids = {
        gesture_start_token_id,
        audio_gesture_start_token_id,
        gesture_end_token_id,
        audio_gesture_end_token_id
    }
    
    print(f"Input audio tokens length: {len(audio_tokens)}")
    print(f"Audio tokens range: [{min(audio_tokens)}, {max(audio_tokens)}]")
    print(f"Special token IDs: {special_token_ids}")
    
    generated_motion_tokens = []  # 用于VQ-VAE的纯motion tokens（已过滤special token）
    current_seq = []  # 当前序列（包含special token，用于模型推理）
    token_labels = []  # token类型标签
    generated_history = []  # 生成历史（用于重复惩罚）
    
    interleave_audios, interleave_motions = 1, 1  # 从训练配置中获取
    motion_token_count = 0  # 总的motion token计数（不包括special token）
    
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
                    
                    # 调试信息：检查输入范围
                    if i < 3:  # 只在前几次打印调试信息
                        audio_tokens_in_input = [x for x, l in zip(current_seq, token_labels) if l == -100]
                        motion_tokens_in_input = [x for x, l in zip(current_seq, token_labels) if l != -100]
                        print(f"  🔍 调试信息 (第{i+1}个audio token, 第{j+1}个motion token):")
                        print(f"     序列长度: {len(current_seq)}")
                        print(f"     Audio tokens数量: {len(audio_tokens_in_input)}, 范围: [{min(audio_tokens_in_input) if audio_tokens_in_input else 'N/A'}, {max(audio_tokens_in_input) if audio_tokens_in_input else 'N/A'}]")
                        print(f"     Motion tokens数量: {len(motion_tokens_in_input)}, 范围: [{min(motion_tokens_in_input) if motion_tokens_in_input else 'N/A'}, {max(motion_tokens_in_input) if motion_tokens_in_input else 'N/A'}]")
                    
                    # 调用模型forward方法
                    output = model(input_data=inputs, attention_mask=attn_mask, labels=labels)
                    next_token_logits = output.logits[0, -1, :]
                    
                    # 应用重复惩罚（只对非special token应用）
                    if repetition_penalty != 1.0 and generated_history:
                        for token_id in set(generated_history):
                            if token_id not in special_token_ids and token_id < next_token_logits.size(-1):
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
                    
                    # 处理special token：根据训练时的规则补充对应的token
                    tokens_to_add = []  # 要添加到序列的tokens（包括补充的token）
                    tokens_labels_to_add = []  # 对应的labels
                    
                    if next_token == gesture_start_token_id:
                        # 模型预测了gesture_start，按照训练规则补充audio_gesture_start
                        print(f"  🔵 检测到gesture_start token ({gesture_start_token_id})，补充audio_gesture_start")
                        tokens_to_add.append(gesture_start_token_id)
                        tokens_labels_to_add.append(gesture_start_token_id)  # motion类型
                        tokens_to_add.append(audio_gesture_start_token_id)
                        tokens_labels_to_add.append(-100)  # audio类型
                        # 注意：gesture_start和audio_gesture_start都不添加到generated_motion_tokens（VQ-VAE不需要）
                        
                    elif next_token == gesture_end_token_id:
                        # 模型预测了gesture_end，按照训练规则补充audio_gesture_end
                        print(f"  🔴 检测到gesture_end token ({gesture_end_token_id})，补充audio_gesture_end")
                        tokens_to_add.append(gesture_end_token_id)
                        tokens_labels_to_add.append(gesture_end_token_id)  # motion类型
                        tokens_to_add.append(audio_gesture_end_token_id)
                        tokens_labels_to_add.append(-100)  # audio类型
                        # 注意：gesture_end和audio_gesture_end都不添加到generated_motion_tokens（VQ-VAE不需要）
                        
                    elif next_token in special_token_ids:
                        # 其他special token（audio_gesture_start或audio_gesture_end）
                        # 这些应该是模型自动生成的补充token，直接添加
                        tokens_to_add.append(next_token)
                        tokens_labels_to_add.append(-100)  # audio类型
                        # 不添加到generated_motion_tokens
                        
                    else:
                        # 普通motion token
                        # 确保motion token在正确范围内
                        vocab_size = model.config.vocab_size
                        if next_token >= vocab_size:
                            print(f"⚠️  警告: motion token {next_token} >= vocab_size {vocab_size}, 调整为 {vocab_size - 1}")
                            next_token = vocab_size - 1
                        
                        # 对于motion token，根据奇偶性调整范围（可选，如果模型已经输出正确范围）
                        # 这里保持原样，因为模型应该已经学会了正确的范围
                        
                        tokens_to_add.append(next_token)
                        tokens_labels_to_add.append(next_token)  # motion token对应的label为自身
                        
                        # 添加到generated_motion_tokens（用于VQ-VAE）
                        generated_motion_tokens.append(next_token)
                        motion_token_count += 1
                        
                        # 添加到生成历史（用于重复惩罚）
                        generated_history.append(next_token)
                    
                    # 将生成的token添加到当前序列（用于后续推理）
                    for token, label in zip(tokens_to_add, tokens_labels_to_add):
                        current_seq.append(token)
                        token_labels.append(label)
                    
                    # 保持历史记录长度
                    if len(generated_history) > 100:
                        generated_history = generated_history[-100:]
                    
                    # 检查是否达到最大motion token数量（只计算实际motion token）
                    if len(generated_motion_tokens) >= max_new_tokens:
                        break
                
                # 检查是否达到最大motion token数量
                if len(generated_motion_tokens) >= max_new_tokens:
                    break
    
    print(f"\n📊 生成统计:")
    print(f"   Generated motion tokens (for VQ-VAE): {len(generated_motion_tokens)}")
    if generated_motion_tokens:
        print(f"   Motion tokens range: [{min(generated_motion_tokens)}, {max(generated_motion_tokens)}]")
    
    # 过滤掉所有special token（额外安全检查）
    filtered_tokens = [t for t in generated_motion_tokens if t not in special_token_ids]
    if len(filtered_tokens) != len(generated_motion_tokens):
        print(f"⚠️  过滤了 {len(generated_motion_tokens) - len(filtered_tokens)} 个special token")
        generated_motion_tokens = filtered_tokens
    
    return generated_motion_tokens


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """
    解码motion tokens为motion features
    注意：motion_tokens应该已经过滤掉所有special token（由generate_motion_tokens处理）
    
    Args:
        motion_tokens: List of motion tokens [body1, hand1, body2, hand2, ...]（已过滤special token）
        motion_vae: The motion VQ-VAE model
        mean_t: Mean tensor for denormalization
        std_t: Std tensor for denormalization
        expected_frames: Expected number of motion frames (optional, used for truncation)
    
    Returns:
        data_dict: Decoded motion data in pkl format
    """
    # 额外安全检查：过滤掉任何可能的special token（以防万一）
    special_token_ids = {512*2 + 2, 512*2 + 3, 512*2 + 4, 512*2 + 5}  # gesture_start, audio_gesture_start, gesture_end, audio_gesture_end
    filtered_tokens = [t for t in motion_tokens if t not in special_token_ids]
    if len(filtered_tokens) != len(motion_tokens):
        print(f"⚠️  decode_motion_tokens: 过滤了 {len(motion_tokens) - len(filtered_tokens)} 个special token")
        motion_tokens = filtered_tokens
    
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
    
    # 验证和修正token范围
    # Body tokens应该在[0, 511]范围内
    body_tokens = [max(0, min(511, int(t))) for t in body_tokens]
    
    # Hand tokens应该在[512, 1023]范围内（在减去512之前）
    # 修正超出范围的hand tokens
    corrected_hand_tokens = []
    for t in hand_tokens:
        t_int = int(t)
        if t_int < 512:
            # 如果小于512，假设模型输出的是[0, 511]范围，需要加上512
            corrected_hand_tokens.append(512 + (t_int % 512))
        elif t_int > 1023:
            # 如果大于1023，限制到[512, 1023]
            corrected_hand_tokens.append(512 + (t_int % 512))
        else:
            corrected_hand_tokens.append(t_int)
    hand_tokens = corrected_hand_tokens
    
    # Convert to tensors
    body_tokens_tensor = torch.tensor(body_tokens).unsqueeze(0).to("cuda")
    # hand tokens need to subtract 512 offset
    hand_tokens_tensor = torch.tensor(hand_tokens).unsqueeze(0).to("cuda") - 512
    
    print(f"Body tokens shape: {body_tokens_tensor.shape}, range: [{body_tokens_tensor.min()}, {body_tokens_tensor.max()}]")
    print(f"Hand tokens shape: {hand_tokens_tensor.shape}, range: [{hand_tokens_tensor.min()}, {hand_tokens_tensor.max()}]")
    print(f"Hand tokens (before offset): {hand_tokens[:5]}... (should be in [512, 1023])")
    
    # Decode using VAE - this returns normalized features
    decoded = motion_vae.decode((body_tokens_tensor, hand_tokens_tensor))
    
    print(f"Decoded shape before truncation: {decoded.shape}")
    
    # If expected_frames is provided and decoded is longer, truncate to expected length
    if expected_frames is not None and decoded.shape[1] > expected_frames:
        decoded = decoded[:, :expected_frames, :]
        print(f"Truncated to expected length: {decoded.shape}")
    
    # Convert to data format (feats2datapkl will handle denormalization with provided stats)
    data_dict = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())
    
    return data_dict


def audioToken2motionPkl(audio_codes, motion_tokens_gt):
    """
    Convert audio codes to motion codes using the updated generation approach.
    """
    print("🔄 使用free generation模式生成motion tokens...")
    
    # kimi_model.generate()返回的audio_tokens已经减去了kimia_token_offset
    # 但GPT2 adaptor的audio_tokenizer期望的是原始的token值（加上offset）
    # 因此需要将audio_tokens重新加上kimia_token_offset
    kimia_token_offset = kimi_model.kimia_token_offset
    
    # 处理tensor或numpy array
    if isinstance(audio_codes, torch.Tensor):
        audio_codes_squeezed = audio_codes.squeeze(0)
        audio_codes_with_offset = audio_codes_squeezed + kimia_token_offset
        # 转换为列表以便在generate_motion_tokens中迭代
        audio_codes_with_offset = audio_codes_with_offset.cpu().tolist()
        audio_codes_squeezed_for_print = audio_codes_squeezed.cpu()
    else:
        import numpy as np
        audio_codes_squeezed = np.array(audio_codes).squeeze(0)
        audio_codes_with_offset = (audio_codes_squeezed + kimia_token_offset).tolist()
        audio_codes_squeezed_for_print = audio_codes_squeezed
    
    print(f"🔧 修正audio tokens: 加上kimia_token_offset={kimia_token_offset}")
    print(f"   修正前范围: [{audio_codes_squeezed_for_print.min()}, {audio_codes_squeezed_for_print.max()}]")
    print(f"   修正后范围: [{min(audio_codes_with_offset)}, {max(audio_codes_with_offset)}]")
    
    # 生成motion tokens
    motion_tokens = generate_motion_tokens(
        motion_adaptor, 
        audio_codes_with_offset, 
        device="cuda",
        max_new_tokens=4096,
        temperature=1.2,
        top_k=50,
        repetition_penalty=1.8
    )
    
    # 解码motion tokens为motion data
    motion_pkl = decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t)
    
    return motion_pkl, motion_tokens


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
    
    # 添加special token ID到模型配置（与训练时保持一致）
    model_config.gesture_start_token_id = 512*2 + 2
    model_config.audio_gesture_start_token_id = 512*2 + 3
    model_config.gesture_end_token_id = 512*2 + 4
    model_config.audio_gesture_end_token_id = 512*2 + 5
    
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
    print(f"   Special token IDs: gesture_start={model_config.gesture_start_token_id}, "
          f"audio_gesture_start={model_config.audio_gesture_start_token_id}, "
          f"gesture_end={model_config.gesture_end_token_id}, "
          f"audio_gesture_end={model_config.audio_gesture_end_token_id}")
    return model


def load_gpt2_from_transformers(model_path, device="cuda"):
    """从transformers格式目录加载模型"""
    print(f"🔄 Loading from transformers format: {model_path}")
    
    # 加载配置
    config_path = os.path.join(model_path, "config.json")
    config = AutoConfig.from_pretrained(config_path)
    
    # 确保special token ID存在（如果config中没有，使用默认值）
    if not hasattr(config, 'gesture_start_token_id'):
        config.gesture_start_token_id = 512*2 + 2
        config.audio_gesture_start_token_id = 512*2 + 3
        config.gesture_end_token_id = 512*2 + 4
        config.audio_gesture_end_token_id = 512*2 + 5
        print("⚠️  Special token IDs not found in config, using default values")
    
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
    print(f"   Special token IDs: gesture_start={config.gesture_start_token_id}, "
          f"audio_gesture_start={config.audio_gesture_start_token_id}, "
          f"gesture_end={config.gesture_end_token_id}, "
          f"audio_gesture_end={config.audio_gesture_end_token_id}")
    return model

# 加载motion adaptor模型（可以根据需要修改路径）
motion_adaptor_path = "/root/workspace/HRI_MLLM/output/motion_adaptor_v5/kimi_audio_motion_gpt2_brainco_synthetic_en/checkpoints/epoch_1000.pt"
motion_adaptor = load_gpt2_model(motion_adaptor_path)

### Loading motion VQ-VAE (Semantic Enhanced)
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

# 加载配置文件
vqvae_config = 'g1_vqvae_arbitrary_length_balanced.yaml'
config_path = os.path.join(ROOT, "model", "motion_encoder", vqvae_config)
print(f"Loading VQ-VAE config from: {config_path}")
motion_config = open_yaml(config_path)

# 确定checkpoint路径
if "ckpt" in motion_config and motion_config["ckpt"]:
    checkpoint_path = motion_config["ckpt"]
else:
    checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
    print(f"⚠️  No checkpoint specified in config, using default: {checkpoint_path}")

# 检查checkpoint是否存在
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print(f"\n可用的checkpoint路径示例:")
    print(f"  - output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt")
    print(f"\n请检查配置文件中的ckpt路径")
else:
    print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")

# 加载归一化统计量（与训练时保持一致）
test_mean, test_std = load_normalization_stats(motion_config)
mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")

# 加载模型
motion_vae = VQVaeBodyHand(**motion_config)
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")
print(f"✅ VQ-VAE model loaded successfully!")



### 使用音频输入生成motion
print(f"\n🎵 使用音频文件作为输入: {audio_input_path}")
print(f"📝 用户指令: {user_text_instruction}")

# 检查音频文件是否存在
if not os.path.exists(audio_input_path):
    print(f"❌ 错误: 音频文件不存在: {audio_input_path}")
    raise FileNotFoundError(f"Audio file not found: {audio_input_path}")

# 构建messages，包含用户文本和音频
messages = [
    {
        "role": "user",
        "message_type": "text",
        "content": user_text_instruction,
    },
    {
        "role": "user",
        "message_type": "audio",
        "content": audio_input_path,
    }
]

print("\n🔄 开始生成音频和motion...")
wav, audio_tokens, text = kimi_model.generate(messages, **sampling_params, output_type="both")

## Use a local HuggingFace model to inference.





# motion_tokens = motion_vae.encode(normalize_vec(torch.from_numpy(train_data_feature).unsqueeze(0).to("cuda:0")))[0].detach().cpu()

motion_pkl, llm_motion_tokens = audioToken2motionPkl(audio_tokens, None)




with open("llm.pkl", 'wb') as f:
    pickle.dump(motion_pkl, f)
    
motion_csv =  load_motion_pkl_as_csv_data("llm.pkl")

np.savetxt("llm.csv", motion_csv, delimiter=',', fmt='%.8f')

sf.write("audio.wav", wav.detach().cpu().view(-1).numpy(), 24000)

# 生成可视化视频
print("\n🎬 开始生成可视化视频...")
vis_audio_motion(
    "llm.csv", 
    output_path="final_output_llm_audio_input.mp4", 
    audio_path="audio.wav", 
    robot_type="g1_brainco", 
    rate_limit=False, 
    motion_fps=25
)

print(f"\n🎉 处理完成!")
print(f"   - Motion pkl: llm.pkl")
print(f"   - Motion csv: llm.csv")
print(f"   - Audio: audio.wav")
print(f"   - 可视化视频: final_output_llm_audio_input.mp4")

