### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

import os
import argparse

from HRI_retarget.utils.motion_lib.qpose_denoiser import low_pass_filter
from huggingface_hub import snapshot_download
os.environ['MUJOCO_GL'] = 'egl'

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenMotionAdaptorStreamer
from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT

from HRI_mllm.utils.motion_utils.g1ml3d_final import vec_to_data_pkl, feats2datapkl
from HRI_mllm.utils.motion_utils.g1ml3d_final import load_normalization_stats
from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand


from kimia_infer.api.kimia import KimiAudio


from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data

from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion

import numpy as np


import yaml
import os
import sys
import pickle

from transformers import GPT2Config, GPT2LMHeadModel
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from types import SimpleNamespace
import soundfile as sf


import os
import sys
from typing import Optional, Dict
from dotenv import load_dotenv
from tts.comm_funcs.tts import Factory as TTS_Factory #TTS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pydub import AudioSegment

import torch
torch.cuda.set_device(0)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager


def audioToken2motionPkl(audio_codes, motion_tokens_gt):
    """
    Convert audio codes to motion codes.
    """
    config={
        "beat_tts_root": "/root/workspace/HRI_MLLM/data/BEAT_v2_kimi",
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512*2,  # 与训练配置一致
        "total_vocab_size": 512*2 + 2,
        "max_seq_length": 4096,
        "min_seq_length": 128,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 1000,
        "sliding_window_step": 32,
        "pad_token_id": 512*2 + 1,
        "interleave_ratio": [1, 1],
    }
    config = SimpleNamespace(**config)

    def generate_for_long_audio(audio_tokens, motion_tokens_gt, model, device, max_length=4096, 
                           top_k=50, temperature=0.3, repetition_penalty=1.2, use_greedy=False):
        """
        生成长音频对应的运动token，使用多样性增强技术
        
        Args:
            audio_tokens: 音频token序列
            motion_tokens_gt: 真实运动token序列（用于teacher forcing）
            model: 生成模型
            device: 设备
            max_length: 最大生成长度
            top_k: top-k采样参数
            temperature: 温度参数，控制随机性
            repetition_penalty: 重复惩罚参数，>1时惩罚重复token
        """
        model.eval()
        generated = []
        current_seq = []
        motion_count = 0
        max_context = max_length - 50  # 保留空间生成新token
        token_labels = []
        
        # 用于重复惩罚的token历史记录
        generated_history = []
        
        with torch.no_grad():
            for i, token in enumerate(audio_tokens):
                
                current_seq.append(token)
                token_labels.append(-100)  # audio token对应的label为-100
                
                inputs = torch.tensor([current_seq]).to(device)
                attn_mask = torch.ones_like(inputs).to(dtype=torch.long).to(device)
                labels = torch.tensor([token_labels]).to(device)

                output = model(inputs, attention_mask=attn_mask, labels=labels)
                next_token_logits = output.logits[0, -1, :]
                
                # 应用重复惩罚
                if repetition_penalty != 1.0 and generated_history:
                    for token_id in set(generated_history):
                        next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                
                # 应用temperature
                next_token_logits = next_token_logits / temperature
                
                # top-k采样
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    # 创建mask，将非top-k的logits设为负无穷
                    mask = torch.ones_like(next_token_logits) * float('-inf')
                    mask[top_k_indices] = next_token_logits[top_k_indices]
                    next_token_logits = mask
                
                # 选择采样策略
                if use_greedy:
                    # 贪心解码：选择概率最高的token
                    next_token = torch.argmax(next_token_logits).item()
                else:
                    # 随机采样
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                
                # 将生成的 token 约束到合法的 motion 码本范围：
                # 偶数次 motion（body）∈ [0, 511]；奇数次 motion（hand）∈ [512, 1023]
                # 避免后续 embedding/indexSelect 越界
                code_num = 512
                if motion_count % 2 == 0:
                    # body token [0, 511]
                    next_token = int(next_token % code_num)
                else:
                    # hand token [512, 1023]
                    next_token = int(code_num + (next_token % code_num))

                generated.append(next_token)
                generated_history.append(next_token)  # 记录生成历史
                
                # 保持历史记录长度，避免内存占用过大
                if len(generated_history) > 100:
                    generated_history = generated_history[-100:]
                
                # free-running
                current_seq.append(next_token)  
                token_labels.append(next_token)  # motion token对应的label为自身
                motion_count += 1

                if len(current_seq) >= max_length:
                    break
                if motion_count >= len(motion_tokens_gt):
                    break   
    
        # 提取生成的motion tokens
        motion_tokens = [t for t in generated]
        return motion_tokens
    
    # 尝试不同的生成策略
    print("🔄 尝试贪心解码（确定性生成）...")
    motion_tokens = generate_for_long_audio(audio_codes.squeeze(0), range(10000), motion_adaptor, device=motion_adaptor.device, use_greedy=True)
    
    # 参考test_visualize_gt_tokens.py的解码方式
    if len(motion_tokens) % 2 != 0:
        print(f"Warning: motion_tokens length ({len(motion_tokens)}) is odd, dropping last token")
        motion_tokens = motion_tokens[:-1]
    
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
    
    print(f"Decoded shape: {decoded.shape}")
    
    # Convert to data format (feats2datapkl will handle denormalization with provided stats)
    data_dict_decoded = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())

    return data_dict_decoded, motion_tokens


# 解析命令行参数
parser = argparse.ArgumentParser(description='Test GPT2 Motion Adaptor')
parser.add_argument('--vqvae_config', type=str, default='g1_vqvae_arbitrary_length_balanced.yaml',
                   help='VQ-VAE config file name')
parser.add_argument('--vqvae_checkpoint', type=str, default=None,
                   help='VQ-VAE checkpoint path. If not provided, will use the one in config file')
parser.add_argument('--audio_path', type=str, default="/root/workspace/HRI_MLLM/data/beat_english_v0.2.1/1/1_wayne_0_1_1_qwen1.wav",
                   help='Input audio file path')
parser.add_argument('--motion_adaptor_path', type=str, 
                   default="output/motion_adaptor_v3/kimi_audio_motion_gpt2_brainco_30_100/transformers_format",
                   help='Motion adaptor model path (transformers format directory)')
args = parser.parse_args()

# 加载模型 - 使用新的transformers格式
config = GPT2Config.from_pretrained(args.motion_adaptor_path)
model_path = os.path.join(args.motion_adaptor_path, "pytorch_model.bin")
motion_adaptor = MixedInputGPT2(config, audio_hidden_size=3584)
motion_adaptor.load_state_dict(torch.load(model_path, map_location="cpu"))
motion_adaptor.to("cuda")
motion_adaptor.eval()
print(f"✅ Motion adaptor loaded from: {args.motion_adaptor_path}")

### Loading motion VQ-VAE (Semantic Enhanced)
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

# 加载配置文件
config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
print(f"Loading VQ-VAE config from: {config_path}")
motion_config = open_yaml(config_path)

# 确定checkpoint路径
if args.vqvae_checkpoint:
    checkpoint_path = args.vqvae_checkpoint
elif "ckpt" in motion_config and motion_config["ckpt"]:
    checkpoint_path = motion_config["ckpt"]
else:
    checkpoint_path = "output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt"
    print(f"⚠️  No checkpoint specified in config, using default: {checkpoint_path}")

# 检查checkpoint是否存在
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print(f"\n可用的checkpoint路径示例:")
    print(f"  - output/vqvae_arbitrary_length_balanced/checkpoints/vqvae_final.pt")
    print(f"\n请使用 --vqvae_checkpoint 参数指定正确的路径")
    exit(1)

print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")

# 加载归一化统计量（与训练时保持一致）
test_mean, test_std = load_normalization_stats(motion_config)
mean_t = torch.tensor(test_mean, dtype=torch.float32).to("cuda")
std_t = torch.tensor(test_std, dtype=torch.float32).to("cuda")
print(f"✅ Loaded normalization stats: Mean {test_mean.shape}, Std {test_std.shape}")

# 加载模型
motion_vae = VQVaeBodyHand(**motion_config)
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")
print(f"✅ VQ-VAE model loaded successfully!")



filename = "2_scott_0_3_3"
# audio_token_path =  f"{DATA_ROOT}/BEAT_v2_kimi/data/{filename}_audio_tokens.pt"
# train_data_feature = np.load(f"{DATA_ROOT}/BEAT_v2_kimi/new_joint_vecs/{filename}.npy")
# audio_path = f"{DATA_ROOT}/beat_english_v0.2.1/{filename.split('_')[0]}/{filename}.wav"

# 使用命令行指定的音频路径
audio_path = args.audio_path
print(f"Using audio file: {audio_path}")


# audio_tokens = torch.load(audio_token_path).squeeze(0)
cache_path = snapshot_download("moonshotai/Kimi-Audio-7B")
model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)

prompt_manager = KimiAPromptManager(
        model_path=cache_path, kimia_token_offset=model_config.kimia_token_offset, kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
    )
audio_tokens = torch.from_numpy(np.array(prompt_manager._tokenize_audio(audio_path))).unsqueeze(0)

## Use a local HuggingFace model to inference.




# motion_tokens = motion_vae.encode(normalize_vec(torch.from_numpy(train_data_feature).unsqueeze(0).to("cuda:0")))[0]

# 生成motion并保存结果
motion_pkl, llm_motion_tokens = audioToken2motionPkl(audio_tokens, None)

# 添加调试信息
print(f"\n📊 生成统计信息:")
print(f"   - 生成的motion tokens数量: {len(llm_motion_tokens)}")
print(f"   - Body tokens: {len(llm_motion_tokens[0::2])}")
print(f"   - Hand tokens: {len(llm_motion_tokens[1::2])}")
print(f"   - Body token范围: [{min(llm_motion_tokens[0::2])}, {max(llm_motion_tokens[0::2])}]")
print(f"   - Hand token范围: [{min(llm_motion_tokens[1::2])}, {max(llm_motion_tokens[1::2])}]")


# decoded_features = motion_vae.decode(motion_tokens).detach().cpu()

# decoded_data_pkl = feats2datapkl(decoded_features)
# source_data_pkl = feats2datapkl(normalize_vec(torch.from_numpy(train_data_feature).unsqueeze(0)))


# with open("source.pkl", 'wb') as f:
#     pickle.dump(source_data_pkl, f)
# with open("decoded.pkl", 'wb') as f:
#     pickle.dump(decoded_data_pkl, f)
with open("output.pkl", 'wb') as f:
    pickle.dump(motion_pkl, f)
    

# source_csv =  load_motion_pkl_as_csv_data("source.pkl")
# decoded_csv =  load_motion_pkl_as_csv_data("decoded.pkl")
motion_csv =  load_motion_pkl_as_csv_data("output.pkl")

# motion_csv = low_pass_filter(motion_csv)


# np.savetxt("source.csv", source_csv, delimiter=',', fmt='%.8f')
# np.savetxt("decoded.csv", decoded_csv, delimiter=',', fmt='%.8f')
np.savetxt("llm.csv", motion_csv, delimiter=',', fmt='%.8f')

import shutil
shutil.copyfile(audio_path, "audio.wav")


vis_audio_motion("llm.csv", output_path="final_output_llm_greedy.mp4", audio_path="audio.wav", robot_type="g1_brainco", rate_limit=False, motion_fps=25)

# 额外测试：尝试随机采样
print("\n🔄 尝试随机采样（对比测试）...")
# 直接调用原函数，但修改use_greedy参数
# 我们需要重新定义audioToken2motionPkl函数来支持随机采样
def audioToken2motionPkl_random(audio_codes, motion_tokens_gt):
    """随机采样版本的生成函数"""
    config={
        "beat_tts_root": "/root/workspace/HRI_MLLM/data/BEAT_v2_kimi",
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512*2,  # 与训练配置一致
        "total_vocab_size": 512*2 + 2,
        "max_seq_length": 4096,
        "min_seq_length": 128,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 1000,
        "sliding_window_step": 32,
        "pad_token_id": 512*2 + 1,
        "interleave_ratio": [1, 1],
    }
    config = SimpleNamespace(**config)

    def generate_for_long_audio_random(audio_tokens, motion_tokens_gt, model, device, max_length=4096, 
                               top_k=50, temperature=0.8, repetition_penalty=1.8):
        """随机采样版本"""
        model.eval()
        generated = []
        current_seq = []
        motion_count = 0
        max_context = max_length - 50
        token_labels = []
        generated_history = []
        
        with torch.no_grad():
            for i, token in enumerate(audio_tokens):
                current_seq.append(token)
                token_labels.append(-100)
                
                inputs = torch.tensor([current_seq]).to(device)
                attn_mask = torch.ones_like(inputs).to(dtype=torch.long).to(device)
                labels = torch.tensor([token_labels]).to(device)

                output = model(inputs, attention_mask=attn_mask, labels=labels)
                next_token_logits = output.logits[0, -1, :]
                
                # 应用重复惩罚
                if repetition_penalty != 1.0 and generated_history:
                    for token_id in set(generated_history):
                        next_token_logits[token_id] = next_token_logits[token_id] / repetition_penalty
                
                # 应用temperature
                next_token_logits = next_token_logits / temperature
                
                # top-k采样
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    mask = torch.ones_like(next_token_logits) * float('-inf')
                    mask[top_k_indices] = next_token_logits[top_k_indices]
                    next_token_logits = mask
                
                # 随机采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Token约束
                code_num = 512
                if motion_count % 2 == 0:
                    next_token = int(next_token % code_num)
                else:
                    next_token = int(code_num + (next_token % code_num))

                generated.append(next_token)
                generated_history.append(next_token)
                
                if len(generated_history) > 100:
                    generated_history = generated_history[-100:]
                
                current_seq.append(next_token)  
                token_labels.append(next_token)
                motion_count += 1

                if len(current_seq) >= max_length:
                    break
                if motion_count >= len(motion_tokens_gt):
                    break   
        
        return [t for t in generated]
    
    motion_tokens = generate_for_long_audio_random(audio_codes.squeeze(0), range(10000), motion_adaptor, device=motion_adaptor.device)
    
    # 解码部分
    if len(motion_tokens) % 2 != 0:
        print(f"Warning: motion_tokens length ({len(motion_tokens)}) is odd, dropping last token")
        motion_tokens = motion_tokens[:-1]
    
    body_tokens = motion_tokens[0::2]
    hand_tokens = motion_tokens[1::2]
    
    body_tokens = torch.tensor(body_tokens).unsqueeze(0).to("cuda")
    hand_tokens = torch.tensor(hand_tokens).unsqueeze(0).to("cuda") - 512
    
    print(f"Random - Body tokens shape: {body_tokens.shape}, range: [{body_tokens.min()}, {body_tokens.max()}]")
    print(f"Random - Hand tokens shape: {hand_tokens.shape}, range: [{hand_tokens.min()}, {hand_tokens.max()}]")
    
    decoded = motion_vae.decode((body_tokens, hand_tokens))
    print(f"Random - Decoded shape: {decoded.shape}")
    
    data_dict_decoded = feats2datapkl(decoded, mean=mean_t.cpu().numpy(), std=std_t.cpu().numpy())
    return data_dict_decoded, motion_tokens

motion_pkl_random, llm_motion_tokens_random = audioToken2motionPkl_random(audio_tokens, None)
with open("output_random.pkl", 'wb') as f:
    pickle.dump(motion_pkl_random, f)
motion_csv_random = load_motion_pkl_as_csv_data("output_random.pkl")
np.savetxt("llm_random.csv", motion_csv_random, delimiter=',', fmt='%.8f')
vis_audio_motion("llm_random.csv", output_path="final_output_llm_random.mp4", audio_path="audio.wav", robot_type="g1_brainco", rate_limit=False, motion_fps=25)

print("\n✅ 生成了两个对比视频:")
print("   - final_output_llm_greedy.mp4 (贪心解码)")
print("   - final_output_llm_random.mp4 (随机采样)")
print("\n💡 建议:")
print("   1. 比较两个视频的质量差异")
print("   2. 如果贪心解码效果更好，说明模型训练良好但采样策略需要调整")
print("   3. 如果两者效果都不好，可能需要检查VQ-VAE解码或数据预处理")
    
