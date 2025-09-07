### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenMotionAdaptorStreamer
from HRI_mllm import ROOT

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl, feats2datapkl
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



def audioToken2motionPkl(audio_codes):
    """
    Convert audio codes to motion codes.
    """
    config={
        "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS_kimi",
        "audio_vocab_size": 16384,
        "motion_vocab_size": 512,
        "total_vocab_size": 16384 + 512,
        "max_seq_length": 1024,    # 模型支持的最大长度
        "min_seq_length": 128,     # 最小序列长度
        "batch_size": 8,
        "learning_rate": 5e-5,
        "epochs": 100,
        "sliding_window_step": 32,  # 滑动窗口步长（单元数）
        "pad_token_id": 16384 + 512, # 新增的填充token
        "interleave_ratio": [5, 2],      # 每5个音频token插入2个动作token
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

    return data_dict_decoded




### loading kimi
model = KimiAudio(
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

### loading motion adaptor
motion_adaptor = GPT2LMHeadModel.from_pretrained("output/motion_adaptor/kimi_audio_motion_gpt2_v1", device_map="auto")

### loading motion vae 
def open_yaml(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data    
motion_config = open_yaml(os.path.join(ROOT, "model", "motion_encoder", "g1_vqvae.yaml"))
motion_vae = VQVae(**motion_config)
state_dict = torch.load(motion_config["ckpt"], map_location="cpu", weights_only=False)
motion_vae.load_state_dict(state_dict, strict=True)
motion_vae.eval()
motion_vae.to(device="cuda")


# ## Use a local HuggingFace model to inference.
# with torch.no_grad():
#     response, audio, audio_code  = inference(text="What do you usually do on weekends? Please describe in detail, including your activities, feelings, and any other relevant information. You can also include any specific events or experiences that stand out to you. The more detailed your response, the better I can understand your weekend activities and emotions.")
#     print(response[0])

#     motion_pkl = audioToken2motionPkl(audio_code)



#     with open("out.pkl", 'wb') as f:
#         pickle.dump(motion_pkl, f)

#     sf.write("out.wav", audio.detach().cpu().numpy(), 24000)

# text = "你周末一般都做些什么事情？请详细描述一下你的活动、感受以及其他相关信息。你也可以包括一些具体的事件或经历，越详细越好，这样我才能更好地了解你周末的活动和情感。"
# tts_save_path = "output/test_audios/tts_output.wav"
# tts(text, tts_save_path)


# output_dir = "output/test_audios/output"
# os.makedirs(output_dir, exist_ok=True)


audio_path =  "/root/pengyang/codebase/HRI_MLLM/data/beat_english_v0.2.1/30/30_katya_0_2_2.wav"
# audio2audio
messages = [
    {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
    {
        "role": "user",
        "message_type": "audio",
        "content": audio_path,
    }
]


wav, audio_tokens, text = model.generate(messages, **sampling_params, output_type="both")

import ipdb;ipdb.set_trace()

sf.write(
    "output.wav",
    wav.detach().cpu().view(-1).numpy(),
    24000,
)
print(">>> output text: ", text)
## Use a local HuggingFace model to inference.


motion_pkl = audioToken2motionPkl(audio_tokens)



with open("output.pkl", 'wb') as f:
    pickle.dump(motion_pkl, f)


motion_csv =  load_motion_pkl_as_csv_data("output.pkl")

np.savetxt("llm.csv", motion_csv, delimiter=',', fmt='%.8f')



    
