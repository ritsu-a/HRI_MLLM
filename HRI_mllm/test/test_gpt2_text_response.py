### streaming demo for qwen2_5omni_motion
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


question = "What do you usually do on weekends?"


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
    model_path="moonshotai/Kimi-Audio-7B-Instruct",
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
    参考visualize_training_reconstruction.py的_free_generation实现
    
    Args:
        model: GPT2 adaptor模型
        audio_tokens: 音频token序列
        device: 设备
        max_new_tokens: 最大新生成token数量
        temperature: 温度参数
        top_k: top-k采样
        repetition_penalty: 重复惩罚
    
    Returns:
        generated_motion_tokens: 生成的motion token序列
    """
    model.eval()
    
    print(f"Input audio tokens length: {len(audio_tokens)}")
    print(f"Audio tokens range: [{min(audio_tokens)}, {max(audio_tokens)}]")
    
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
    
    return generated_motion_tokens


def decode_motion_tokens(motion_tokens, motion_vae, mean_t, std_t, expected_frames=None):
    """
    解码motion tokens为motion features
    参考visualize_training_reconstruction.py的decode_motion_tokens实现
    
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


def audioToken2motionPkl(audio_codes, motion_tokens_gt):
    """
    Convert audio codes to motion codes using the updated generation approach.
    """
    print("🔄 使用free generation模式生成motion tokens...")
    
    # 生成motion tokens
    motion_tokens = generate_motion_tokens(
        motion_adaptor, 
        audio_codes.squeeze(0), 
        device="cuda",
        max_new_tokens=4096,
        temperature=1.8,
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

# 加载motion adaptor模型（可以根据需要修改路径）
motion_adaptor_path = "output/motion_adaptor_v1/kimi_audio_motion_gpt2_hidden_30_100"
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



### generate text response
tts_save_path = "/root/pengyang/codebase/HRI_MLLM/tts_output.wav"
tts(question, tts_save_path)
# audio2audio
messages = [
    {
        "role": "user",
        "message_type": "audio",
        "content": tts_save_path,
    }
]


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
    output_path="final_output_llm.mp4", 
    audio_path="audio.wav", 
    robot_type="g1_brainco", 
    rate_limit=False, 
    motion_fps=25
)

print(f"\n🎉 处理完成!")
print(f"   - Motion pkl: llm.pkl")
print(f"   - Motion csv: llm.csv")
print(f"   - Audio: audio.wav")
print(f"   - 可视化视频: final_output_llm.mp4")
