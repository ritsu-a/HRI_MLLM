import argparse
import numpy as np
import time
from tqdm import tqdm
import os
import pickle
import soundfile as sf
import torch
import yaml
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="/root/pengyang/codebase/HRI_MLLM/data/beat_english_v0.2.1/all.txt")
    parser.add_argument("--motion_root", type=str, default="/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1")
    parser.add_argument("--data_path", type=str, default="/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1_kimi")
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B")


    args = parser.parse_args()
    data_list_path = os.path.join(args.data_path, "all.txt")
    save_path = os.path.join(args.data_path, "beat_v1_full.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write('')  

    user_prompt = "Please repeat the following spoken content with a corresponding full-body motion sequence."

    with open(data_list_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    metadata = []

    for line in tqdm(lines):
        audio_path = line.strip()

        wav_path = os.path.join("/root/pengyang/codebase/HRI_MLLM/data/beat_english_v0.2.1", audio_path.split('/')[-1].split('_')[0], audio_path.split('/')[-1].replace('_audio_tokens.pt', '.wav'))
        motion_path = line.strip().replace("audio_tokens", "motion_tokens")

        audio_tokens = torch.load(audio_path).numpy().reshape(-1).tolist()
        motion_tokens = torch.load(motion_path).numpy().reshape(-1).tolist()

        audio_tokens = audio_tokens[:len(motion_tokens)]
        motion_tokens = motion_tokens[:len(audio_tokens)]

        if len(audio_tokens) != len(motion_tokens):
            print(f"Skipping {line.strip()} due to length mismatch between audio and motion tokens.")
            continue

        metadata.append({
                    "task_type": "s2s",
                    
                    "conversation": [
                        {
                            "role": "user",
                            'message_type': 'text',
                            "content": user_prompt
                        },
                        {
                            "role": "user",
                            'message_type': 'audio', 
                            'content': wav_path,
                            'audio_tokens': audio_tokens
                        },
                        {
                            "role": "assistant",
                            "message_type": "audio_motion",
                            'audio_tokens': audio_tokens,
                            'content': wav_path,
                            'motion_tokens': motion_tokens
                        },
                    ]
                })

    with open(save_path, 'w', encoding="utf-8") as f:
        for data in metadata:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Completed processing LibriSpeech dataset. Metadata saved to {save_path}")