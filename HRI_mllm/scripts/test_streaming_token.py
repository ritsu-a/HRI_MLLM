### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion
from transformers import TextStreamer

import torch




# @title inference function
def inference(video_path):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
                {"type": "audio", "audio": video_path},
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
    return text, audio

model_path = "Qwen/Qwen2.5-Omni-3B"
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

monkey_patch_qwen2_5omni_for_motion(Qwen2_5OmniForConditionalGeneration, streamer=streamer)

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)


import librosa
import audioread

from IPython.display import Video
from IPython.display import Audio
from IPython.display import display

video_path = "/root/pengyang/codebase/HRI_MLLM/data/misc/music.mp4"

display(Video(video_path, width=640, height=360))
display(Audio(librosa.load(audioread.ffdec.FFmpegAudioFile(video_path), sr=16000)[0], rate=16000))

## Use a local HuggingFace model to inference.
with torch.no_grad():
    response, audio  = inference(video_path)
    print(response[0])

    display(Audio(audio, rate=24000))

    import ipdb;ipdb.set_trace()