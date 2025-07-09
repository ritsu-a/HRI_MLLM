### streaming demo for qwen2_5omni_motion
### HRI_mllm/model/qwen2_5omni_motion/monkey_patch_generate.py for monkey patching the generate function to support token-level streaming

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni_motion.monkey_patch_generate import monkey_patch_qwen2_5omni_for_motion_tts
from HRI_mllm.model.qwen2_5omni.streamers import QwenTextStreamer

import torch
import argparse
import time 
from datetime import timedelta
from tqdm import tqdm
import soundfile as sf
import textgrid

from jiwer import wer



# @title inference function
def inference(audio_path):
    textgrid_path = audio_path.replace(".wav", ".TextGrid")
    tg = textgrid.TextGrid.fromFile(textgrid_path)

    streamer.tg = tg
    streamer.clear_text_list()

    

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
                {"type": "audio", "audio": audio_path, "audio_start":tg[0][0].maxTime, "audio_end":tg[0][-1].maxTime},
                {"type": "text", "text": "Transcribe the English audio into text without any punctuation marks."}
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    audios = process_audio_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=None, videos=None, return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)
    
    try:
        output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)
    except ValueError as e:
        raise


    text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    audio = output[1]
    audio_codes = output[2]





    return text, audio, audio_codes

model_path = "Qwen/Qwen2.5-Omni-3B"
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer
streamer = QwenTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

monkey_patch_qwen2_5omni_for_motion_tts(Qwen2_5OmniForConditionalGeneration, streamer=streamer)

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


def tts(audio_path, suffix=""):


    display(Audio(librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=16000)[0], rate=16000))

    ## Use a local HuggingFace model to inference.
    with torch.no_grad():
        try:
            response, audio, audio_codes = inference(audio_path)
            save_path = audio_path.replace(".wav", f"{suffix}.wav")
            sf.write(
                save_path,
                audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )

            token_save_path = audio_path.replace(".wav", f"{suffix}_tokens.pt")
            torch.save(audio_codes, token_save_path)

            return True
        except ValueError as e:
            print("Reference Text: ", " ".join([text.strip().lower() for text in list(filter(lambda x: x != "", [ti.mark for ti in streamer.tg[0]]))]))
            print(e)
            return False

            



if __name__ == "__main__":

    # filename = "/root/pengyang/codebase/HRI_MLLM/data/beat_english_v0.2.1/1/1_wayne_0_1_1.TextGrid"
    # import textgrid
    # tg = textgrid.TextGrid.fromFile(filename)
    # import ipdb;ipdb.set_trace()

    # text = tg[0].getText()


    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="/root/pengyang/codebase/HRI_MLLM/data/tmp/beat_tts_0.txt")
    parser.add_argument("--suffix", type=str, default="_qwen1")
    args = parser.parse_args()

    result_path = args.audio_path.replace('.txt', '_result.txt')
    with open(result_path, 'w') as file:
        file.write('')  # 写入空字符串


    with open(args.audio_path, "r", encoding="utf-8") as f:
        audio_files = f.readlines()

    starting_time = time.time()

    total_num = len(audio_files)
    
    current_num = 0
    success_num = 0



    for idx in tqdm(range(total_num)):

        print(f"Current Progress: {idx}/{total_num}", f" Elapsed time: {str(timedelta(seconds=time.time() - starting_time)).split('.')[0]}")
        audio_path = audio_files[idx].strip()
        print(audio_path)
        if tts(audio_path=audio_path, suffix=args.suffix):
            success_num += 1
            with open(result_path, "a") as file:
                file.write(audio_path + "\n")
        current_num += 1

        print("#" * 50)
        print(f"Success / Total : {success_num} / {current_num}")
        print("#" * 50)


