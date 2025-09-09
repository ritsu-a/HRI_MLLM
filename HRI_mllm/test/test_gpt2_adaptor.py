### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenMotionAdaptorStreamer
from HRI_mllm import ROOT

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl, feats2datapkl, normalize_vec
from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans

from kimia_infer.api.kimia import KimiAudio


from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data

import numpy as np


import yaml
import os
import sys
import pickle

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



def audioToken2motionPkl(audio_codes):
    """
    Convert audio codes to motion codes.
    """
    config = {
        "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1_kimi",
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512,
        "total_vocab_size": 16384 + 512,
        "max_seq_length": 1024,
        "min_seq_length": 128,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "epochs": 100,
        "sliding_window_step": 32,
        "pad_token_id": 16384 + 512,
        "interleave_ratio": [1, 1],
    }
    config = SimpleNamespace(**config)

    def generate_for_long_audio(audio_tokens, model, device, max_length=1024):
        model.eval()
        generated = []
        current_seq = []
        motion_count = 0
        max_context = max_length - 50  # 保留空间生成新token
        interleave_audios, interleave_motions = config.interleave_ratio
        
        with torch.no_grad():
            for i, token in enumerate(audio_tokens):
                current_seq.append(token)
                
                # 每5个audio token尝试生成motion
                if (i + 1) % interleave_audios == 0:
                    # 当序列过长时使用滑动窗口
                    if len(current_seq) > max_context:
                        # 保留最近的完整上下文
                        keep_from = max(0, len(current_seq) - max_context)
                        # 确保从完整单元开始
                        while keep_from < len(current_seq) and (keep_from % (interleave_audios + interleave_motions) != 0):
                            keep_from += 1
                        current_seq = current_seq[keep_from:]
                    
                    
                    
                    # 预测下一个motion token
                    for _ in range(interleave_motions):
                        inputs = torch.tensor([current_seq]).to(device)
                        attn_mask = torch.ones_like(inputs).float().to(device)

                        output = model(inputs, attention_mask=attn_mask)
                        next_token_logits = output.logits[0, -1, :]
                        
                        # 限制在motion词表范围内
                        motion_logits = next_token_logits[config.audio_vocab_size:]
                        next_token = torch.argmax(motion_logits).item() + config.audio_vocab_size
                        
                        generated.append(next_token)
                        current_seq.append(next_token)  # 添加到上下文
                        motion_count += 1

                
                if len(current_seq) >= max_length:
                    break
        
        # 提取生成的motion tokens
        motion_tokens = [t - config.audio_vocab_size for t in generated]
        return motion_tokens
    
    motion_tokens = generate_for_long_audio(audio_codes.squeeze(0), motion_adaptor, device=motion_adaptor.device)


    decoded = motion_vae.decode(torch.tensor(motion_tokens).unsqueeze(0).to("cuda"))
    data_dict_decoded = feats2datapkl(decoded.detach().cpu())

    return data_dict_decoded, motion_tokens



### loading motion adaptor
motion_adaptor = GPT2LMHeadModel.from_pretrained("output/motion_adaptor_v1/kimi_audio_motion_gpt2_v2", device_map="auto")

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



filename = "1_wayne_0_1_1"
audio_token_path =  f"/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1_kimi/data/{filename}_audio_tokens.pt"
train_data_feature = np.load(f"/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1_kimi/new_joint_vecs/{filename}.npy")
audio_path = f"/root/pengyang/codebase/HRI_MLLM/data/beat_english_v0.2.1/{filename.split('_')[0]}/{filename}.wav"


audio_tokens = torch.load(audio_token_path).squeeze(0) - 152064


## Use a local HuggingFace model to inference.


motion_pkl, llm_motion_tokens = audioToken2motionPkl(audio_tokens)


motion_tokens = motion_vae.encode(normalize_vec(torch.from_numpy(train_data_feature).unsqueeze(0).to("cuda:0")))[0].detach().cpu()

import ipdb;ipdb.set_trace()

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



    
