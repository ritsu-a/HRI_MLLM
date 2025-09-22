### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenMotionAdaptorStreamer
from HRI_mllm import ROOT, DATA_ROOT, OUTPUT_ROOT

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl, feats2datapkl, normalize_vec
from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans

from kimia_infer.api.kimia import KimiAudio


from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data

import numpy as np


import yaml
import os
import sys
import pickle

from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2

from transformers import GPT2Config, GPT2LMHeadModel
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


def audioToken2motionPkl(audio_codes, motion_tokens_gt):
    """
    Convert audio codes to motion codes.
    """
    config = {
        "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1_kimi",
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512,
        "total_vocab_size": 16384 + 512,
        "max_seq_length": 4096,
        "min_seq_length": 128,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "epochs": 100,
        "sliding_window_step": 32,
        "pad_token_id": 16384 + 512,
        "interleave_ratio": [1, 1],
    }
    config = SimpleNamespace(**config)

    def generate_for_long_audio(audio_tokens, motion_tokens_gt, model, device, max_length=4096, 
                           top_k=50, temperature=0.8, repetition_penalty=1.8):
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
                
                # 使用softmax和多项式采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
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
    
    motion_tokens = generate_for_long_audio(audio_codes.squeeze(0), motion_tokens_gt.squeeze(0), motion_adaptor, device=motion_adaptor.device)


    decoded = motion_vae.decode(torch.tensor(motion_tokens).unsqueeze(0).to("cuda"))
    data_dict_decoded = feats2datapkl(decoded.detach().cpu())

    return data_dict_decoded, motion_tokens


motion_adaptor = MixedInputGPT2.from_pretrained("output/motion_adaptor_v1/kimi_audio_motion_gpt2_hidden_30_100", device_map="auto")

### loading motion VQVAE
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data        
motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae_body.yaml"))
motion_vae = VQVae(**motion_config)
state_dict = torch.load(motion_config["ckpt"], map_location="cpu", weights_only=False)
motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")



filename = "2_scott_0_3_3"
audio_token_path =  f"{DATA_ROOT}/BEAT_v1_kimi/data/{filename}_audio_tokens.pt"
train_data_feature = np.load(f"{DATA_ROOT}/BEAT_v1_kimi/new_joint_vecs/{filename}.npy")
audio_path = f"{DATA_ROOT}/{filename.split('_')[0]}/{filename}.wav"


audio_tokens = torch.load(audio_token_path).squeeze(0)


## Use a local HuggingFace model to inference.




motion_tokens = motion_vae.encode(normalize_vec(torch.from_numpy(train_data_feature).unsqueeze(0).to("cuda:0")))[0].detach().cpu()

motion_pkl, llm_motion_tokens = audioToken2motionPkl(audio_tokens, motion_tokens)


decoded_features = motion_vae.decode(motion_tokens.to("cuda:0")).detach().cpu()

decoded_data_pkl = feats2datapkl(decoded_features)
source_data_pkl = feats2datapkl(normalize_vec(torch.from_numpy(train_data_feature).unsqueeze(0)))


with open("source.pkl", 'wb') as f:
    pickle.dump(source_data_pkl, f)
with open("decoded.pkl", 'wb') as f:
    pickle.dump(decoded_data_pkl, f)
with open("output.pkl", 'wb') as f:
    pickle.dump(motion_pkl, f)
    

source_csv =  load_motion_pkl_as_csv_data("source.pkl")
decoded_csv =  load_motion_pkl_as_csv_data("decoded.pkl")
motion_csv =  load_motion_pkl_as_csv_data("output.pkl")


np.savetxt("source.csv", source_csv, delimiter=',', fmt='%.8f')
np.savetxt("decoded.csv", decoded_csv, delimiter=',', fmt='%.8f')
np.savetxt("llm.csv", motion_csv, delimiter=',', fmt='%.8f')

import shutil
shutil.copyfile(audio_path, "audio.wav")



    
