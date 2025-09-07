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




audio_path =  "/root/pengyang/codebase/HRI_MLLM/data/beat_english_v0.2.1/30/30_katya_0_2_2.wav"
# audio2audio
messages = [
    {"role": "user", "message_type": "text", "content": "Please repeat the following audio:"},
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







    
