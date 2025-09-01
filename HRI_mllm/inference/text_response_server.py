# server.py
from flask import Flask, request, send_file, jsonify
### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenMotionAdaptorStreamer
from HRI_mllm import ROOT

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl, feats2datapkl
from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans

from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_motion_pkl_as_csv_data
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.motion_lib.qpose_denoiser import low_pass_filter

from kimia_infer.api.kimia import KimiAudio


import yaml
import os
import sys
import pickle

from transformers import GPT2Config, GPT2LMHeadModel
from types import SimpleNamespace
import soundfile as sf



import torch
import zipfile
from io import BytesIO
import shutil

import numpy as np

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
                    
                    
                    
                    # 预测motion token
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



app = Flask(__name__)

@app.route('/generate-files', methods=['POST'])
def generate_files():
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # 调用推理函数生成WAV和CSV文件
        # 假设 inference 函数同时生成WAV和CSV
        with torch.no_grad():

            tts_save_path = "output/test_audios/tts_output.wav"
            tts(text, tts_save_path)

            messages = [
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": tts_save_path,
                }
            ]

            wav, audio_tokens, text = model.generate(messages, **sampling_params, output_type="both")

            motion_pkl = audioToken2motionPkl(audio_tokens)



            with open("output.pkl", 'wb') as f:
                pickle.dump(motion_pkl, f)

            motion_csv =  load_motion_pkl_as_csv_data("output.pkl")
            motion_csv = low_pass_filter(motion_csv, cutoff_freq=0.2, order=4)
            np.savetxt("output.csv", motion_csv, delimiter=',', fmt='%.8f')


            sf.write("output.wav", wav.detach().cpu().view(-1).numpy(), 24000)
        # response, audio, audio_code = inference(text=text)
        print(f"Received text: {text}")
        
        # 确保文件存在
        wav_path = "output.wav"
        csv_path = "output.csv"
        
        if not (os.path.exists(wav_path) and os.path.exists(csv_path)):
            return jsonify({"error": "Files not generated"}), 500
        
        # 创建内存中的ZIP文件
        memory_zip = BytesIO()
        
        with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(wav_path, os.path.basename(wav_path))
            zipf.write(csv_path, os.path.basename(csv_path))
        
        memory_zip.seek(0)
        print("Files generated successfully")
        
        # 发送ZIP文件
        return send_file(
            memory_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name='generated_files.zip'
        )
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)