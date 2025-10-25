import argparse
import numpy as np
import time
from tqdm import tqdm
import os
import pickle
import soundfile as sf
import torch
import yaml


import torch.nn.functional as F

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.torch_utils.diff_quat import vec6d_to_quat
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.brainco_representation import data_pkl_to_vec

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni.streamers import QwenTextStreamer

from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm.model.motion_encoder.vqvae_body_hand import VQVaeBodyHand

from HRI_mllm import ROOT

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager


def load_token2wav():
    model_path = "Qwen/Qwen2.5-Omni-3B"
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    speaker_params = model.speaker_map["Chelsie"]

    
    token2wav = model.token2wav
    model.talker = None
    model.thinker = None
    del model

    torch.cuda.empty_cache()  # 清空 GPU 缓存

    def token2wav_adapter(*args, **kwargs):
        for arg in args:
            arg = arg.to("cuda:0")
        kwargs["conditioning"] = speaker_params["cond"].float().to("cuda:0")
        kwargs["reference_mel"] = speaker_params["ref_mel"].float().to("cuda:0")
        return token2wav(*args, **kwargs)


    return token2wav_adapter



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="/root/workspace/HRI_MLLM/data/beat_english_v0.2.1/all.txt")
    parser.add_argument("--motion_root", type=str, default="/root/workspace/HRI_MLLM/data/BEAT_v2_kimi/new_joint_vecs")
    parser.add_argument("--save_path", type=str, default="/root/workspace/HRI_MLLM/data/BEAT_v2_kimi")
    parser.add_argument("--model_name_or_path", type=str, default="moonshotai/Kimi-Audio-7B")
    # VQ-VAE相关参数
    parser.add_argument("--vqvae_config", type=str, default="g1_vqvae_semantic_enhanced.yaml",
                       help="VQ-VAE config file name")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                       help="VQ-VAE checkpoint path. If not provided, will use the one in config file")


    args = parser.parse_args()
    save_split_path = os.path.join(args.save_path, "all.txt")
    save_path = os.path.join(args.save_path, "data")

    with open(save_split_path, "w", encoding="utf-8") as f:
        f.write('')  




    with open(args.audio_path, "r", encoding="utf-8") as f:
        audio_files = f.readlines()


    total_num = len(audio_files)

    ### loading token2wav
    # token2wav = load_token2wav()


    ### Loading motion VQ-VAE (Semantic Enhanced)
    def open_yaml(path):
        with open(path, 'r', encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return data
    
    # 加载配置文件
    config_path = os.path.join(ROOT, "model", "motion_encoder", args.vqvae_config)
    print(f"Loading VQ-VAE config from: {config_path}")
    motion_config = open_yaml(config_path)
    
    # 确定checkpoint路径
    if args.vqvae_checkpoint:
        checkpoint_path = args.vqvae_checkpoint
    elif "ckpt" in motion_config and motion_config["ckpt"]:
        checkpoint_path = motion_config["ckpt"]
    else:
        # 使用默认路径
        if "semantic_enhanced" in args.vqvae_config:
            checkpoint_path = "output/vqvae_semantic_enhanced/checkpoints/vqvae_final.pt"
        else:
            checkpoint_path = "output/vqvae_semantic_training/checkpoints/vqvae_semantic_final.pt"
        print(f"⚠️  No checkpoint specified in config, using default: {checkpoint_path}")
    
    # 检查checkpoint是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"\n可用的checkpoint路径示例:")
        print(f"  - output/vqvae_semantic_enhanced/checkpoints/vqvae_final.pt")
        print(f"  - output/vqvae_semantic_training/checkpoints/vqvae_semantic_final.pt")
        print(f"\n请使用 --vqvae_checkpoint 参数指定正确的路径")
        exit(1)
    
    print(f"Loading VQ-VAE checkpoint from: {checkpoint_path}")
    motion_vae = VQVaeBodyHand(**motion_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_vae.load_state_dict(state_dict, strict=True)
    motion_vae.eval()
    motion_vae.to(device="cuda")
    print(f"✅ VQ-VAE model loaded successfully!")

    ### loading kimi audio tokenizer
    if os.path.exists(args.model_name_or_path):
        # local path
        cache_path = args.model_name_or_path
    else:
        # cache everything if model_path is a model-id
        cache_path = snapshot_download(args.model_name_or_path)

    # load model config
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)

    prompt_manager = KimiAPromptManager(
            model_path=cache_path, kimia_token_offset=model_config.kimia_token_offset, kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
        )


    for idx in tqdm(range(total_num)):

        source_audio_path = audio_files[idx].strip()
        motion_path = os.path.join(args.motion_root, source_audio_path.split("/")[-1].replace(".wav", ".npy"))
        filename = source_audio_path.split("/")[-1]

        audio_path = source_audio_path
        # audio_token_path = audio_path.replace(".wav", "_tokens.pt")
        # audio_tokens = torch.load(audio_token_path)
        # audio, _ = sf.read(audio_path)
        # source_audio, _ = sf.read(source_audio_path)

        kimi_tokens = torch.from_numpy(np.array(prompt_manager._tokenize_audio(audio_path))).unsqueeze(0)
        

        


        g1ml3d_vec = np.load(motion_path)

        # np.save(os.path.join(joint_vecs_save_path, filename.replace(".wav", "_joint_vecs.npy")), g1ml3d_features)


        from HRI_mllm.utils.motion_utils.g1ml3d import normalize_vec
        
        body_tokens, hand_tokens = motion_vae.encode(normalize_vec(torch.from_numpy(g1ml3d_vec).unsqueeze(0).to("cuda:0")))[0]

        motion_tokens = torch.zeros(body_tokens.shape[0], body_tokens.shape[1] + hand_tokens.shape[1])
        motion_tokens[:, 0::2] = body_tokens
        motion_tokens[:, 1::2] = hand_tokens + motion_config["code_num"]


        audio_token_save_path = os.path.join(save_path, filename.replace(".wav", "_audio_tokens.pt"))
        motion_token_save_path = os.path.join(save_path, filename.replace(".wav", "_motion_tokens.pt"))


        torch.save(kimi_tokens, audio_token_save_path)
        torch.save(motion_tokens, motion_token_save_path)


        with open(save_split_path, "a", encoding="utf-8") as f:
            f.write(audio_token_save_path + "\n")  



        


