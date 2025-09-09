from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np
import os
import wandb
import math
from tqdm import tqdm
from HRI_mllm.datasets.BEATAudioMotionDataset import BEATAudioMotionDataset

### loading motion adaptor
motion_adaptor = GPT2LMHeadModel.from_pretrained("output/motion_adaptor_v1/kimi_audio_motion_gpt2_v2", device_map="auto")


# 创建数据集
config = {
    "beat_tts_root": "/root/pengyang/codebase/HRI_MLLM/data/BEAT_v1_kimi",
    "audio_vocab_size": 16384,
    "motion_vocab_size": 512,
    "total_vocab_size": 16384 + 512,
    "max_seq_length": 10240000,
    "min_seq_length": 128,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "epochs": 100,
    "sliding_window_step": 32,
    "pad_token_id": 16384 + 512,
    "interleave_ratio": [1, 1],
}
config = SimpleNamespace(**config)
dataset = BEATAudioMotionDataset(config)

import ipdb;ipdb.set_trace()