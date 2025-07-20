### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenMotionAdaptorStreamer
from HRI_mllm import ROOT

from HRI_mllm.utils.motion_utils.g1ml3d import vec_to_data_pkl, feats2datapkl
from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans



import yaml
import os
import sys
import pickle

from transformers import GPT2Config, GPT2LMHeadModel
from types import SimpleNamespace
import soundfile as sf



import torch




# @title inference function
def inference(text):

    streamer.clear_text_list()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
                {"type": "text", "text": text},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    audios = process_audio_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=None, videos=None, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)

    text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    audio = output[1]
    audio_codes = output[2]

    return text, audio, audio_codes

def audioToken2motionPkl(audio_codes):
    """
    Convert audio codes to motion codes.
    """
    config={
        "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS",
        "audio_vocab_size": 8194,
        "motion_vocab_size": 512,
        "total_vocab_size": 8194 + 512,
        "max_seq_length": 1024,    # 模型支持的最大长度
        "min_seq_length": 128,     # 最小序列长度
        "batch_size": 8,
        "learning_rate": 5e-5,
        "epochs": 100,
        "sliding_window_step": 32,  # 滑动窗口步长（单元数）
        "pad_token_id": 8194 + 512, # 新增的填充token
        "interleave_ratio": 10,      # 每10个音频token插入1个动作token
    }
    config = SimpleNamespace(**config)

    def generate_for_long_audio(audio_tokens, model, device, max_length=1024):
        model.eval()
        generated = []
        current_seq = []
        motion_count = 0
        max_context = max_length - 50  # 保留空间生成新token
        
        with torch.no_grad():
            for i, token in enumerate(audio_tokens):
                current_seq.append(token)
                
                # 每10个audio token尝试生成motion
                if (i + 1) % config.interleave_ratio == 0:
                    # 当序列过长时使用滑动窗口
                    if len(current_seq) > max_context:
                        # 保留最近的完整上下文
                        keep_from = max(0, len(current_seq) - max_context)
                        # 确保从完整单元开始
                        while keep_from < len(current_seq) and (keep_from % (config.interleave_ratio+1) != 0):
                            keep_from += 1
                        current_seq = current_seq[keep_from:]
                    
                    inputs = torch.tensor([current_seq]).to(device)
                    attn_mask = torch.ones_like(inputs).float().to(device)
                    
                    # 预测下一个motion token
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
    import ipdb;ipdb.set_trace()

    decoded = motion_vae.decode(torch.tensor(motion_tokens).unsqueeze(0).to("cuda"))
    data_dict_decoded = feats2datapkl(decoded)

    return data_dict_decoded

### loading qwen
model_path = "Qwen/Qwen2.5-Omni-3B"
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer
streamer = QwenMotionAdaptorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

monkey_patch_qwen2_5omni_for_motion_tts(Qwen2_5OmniForConditionalGeneration, streamer=streamer)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

### loading motion adaptor
motion_adaptor = GPT2LMHeadModel.from_pretrained("output/motion_adaptor/audio_motion_gpt2_v1", device_map="auto")

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


## Use a local HuggingFace model to inference.
with torch.no_grad():
    response, audio, audio_code  = inference(text="What do you usually do on weekends? Please describe in detail, including your activities, feelings, and any other relevant information. You can also include any specific events or experiences that stand out to you. The more detailed your response, the better I can understand your weekend activities and emotions.")
    print(response[0])

    motion_pkl = audioToken2motionPkl(audio_code)



    with open("out.pkl", 'wb') as f:
        pickle.dump(motion_pkl, f)

    sf.write("out.wav", audio.detach().cpu().numpy(), 24000)
