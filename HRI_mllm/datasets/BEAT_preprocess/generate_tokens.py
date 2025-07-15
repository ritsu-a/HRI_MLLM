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
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.g1_29_humanml3d_representation import data_pkl_to_vec

from HRI_mllm.model.qwen2_5omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from HRI_mllm.utils.qwen_omni_utils import process_mm_info, process_audio_info
from HRI_mllm.model.qwen2_5omni.streamers import QwenTextStreamer

from HRI_mllm.model.motion_encoder.vqvae import VQVae, VQVAE_Trans
from HRI_mllm import ROOT

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


def interpolate_tensor(tensor, target_length):
    ### tensor (num_frames, channels)
    input_tensor = tensor.permute(1, 0).unsqueeze(0)
    resized_tensor = F.interpolate(
        input_tensor, 
        size=target_length, 
        mode='linear',  # 线性插值
        align_corners=True
    )
    result = resized_tensor.squeeze(0).permute(1, 0)
    return result

def interpolate_quat(quat, target_length):
    ### tensor(num_frames, xyzw)
    quats = quat.numpy() if torch.is_tensor(quat) else quat

    old_times = np.linspace(0, 1, len(quats))
    slerp = Slerp(old_times, R.from_quat(quats))

    new_times = np.linspace(0, 1, target_length)

    interp_rots = slerp(new_times).as_matrix() 
    interp_rots_tensor = torch.tensor(interp_rots, dtype=quat.dtype)
    return interp_rots_tensor[:, :, :2]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS/tts_qwen1_result.txt")
    parser.add_argument("--motion_root", type=str, default="/root/pengyang/codebase/HRI_MLLM/data/motion/g1/BEAT")
    parser.add_argument("--save_path", type=str, default="/root/pengyang/codebase/HRI_MLLM/data/BEAT_TTS/data")

    args = parser.parse_args()



    with open(args.audio_path, "r", encoding="utf-8") as f:
        audio_files = f.readlines()


    total_num = len(audio_files)

    ### loading token2wav
    # token2wav = load_token2wav()


    ### loading motion VQVAE
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


    for idx in tqdm(range(total_num)):

        source_audio_path = audio_files[idx].strip()
        motion_path = os.path.join(args.motion_root, source_audio_path.split("/")[-1].replace(".wav", ".pickle"))
        filename = source_audio_path.split("/")[-1]

        audio_path = source_audio_path.replace(".wav", "_qwen1.wav")
        audio_token_path = audio_path.replace(".wav", "_tokens.pt")
        audio_tokens = torch.load(audio_token_path)
        audio, _ = sf.read(audio_path)
        source_audio, _ = sf.read(source_audio_path)

        
        with open(motion_path, "rb") as file:
            motion_pkl = pickle.load(file)


        ### resample motion pkl to match the tts audio length
        target_motion_frames = int(audio_tokens.shape[1] / 50 * 20)

        resampled_angles = interpolate_tensor(torch.from_numpy(motion_pkl["angles"]), target_motion_frames)
        motion_pkl_resampled = {
            "fps": 20,
            "angles": torch.cat([resampled_angles[:, :22], resampled_angles[:, 34:41]], dim=1),
            "global_rotation": interpolate_quat(vec6d_to_quat(torch.from_numpy(motion_pkl["global_rotation"])), target_motion_frames),
            "global_translation": interpolate_tensor(torch.from_numpy(motion_pkl["global_translation"]).squeeze(-1), target_motion_frames).unsqueeze(-1),
            "robot_name": "g1_29",
            "scale": torch.ones(3),
        }

        g1ml3d_features = data_pkl_to_vec(motion_pkl_resampled)

        from HRI_mllm.utils.motion_utils.g1ml3d import normalize_features, feats2datapkl
        
        motion_tokens = motion_vae.encode(normalize_features(torch.from_numpy(g1ml3d_features).unsqueeze(0).to("cuda:0")))[0].detach().cpu()


        audio_token_save_path = os.path.join(args.save_path, filename.replace(".wav", "_audio_tokens.pt"))
        motion_token_save_path = os.path.join(args.save_path, filename.replace(".wav", "_motion_tokens.pt"))

        torch.save(audio_tokens, audio_token_save_path)
        torch.save(motion_tokens, motion_token_save_path)


        # decoded_features = motion_vae.decode(motion_tokens.to("cuda:0")).detach().cpu()
        # data_dict_decoded = feats2datapkl(decoded_features)


        # with open(os.path.join(args.save_path, filename.replace(".wav", "_decoded.pkl")), 'wb') as f:
        #     pickle.dump(data_dict_decoded, f)

        # with torch.no_grad():
        #     decoded_audio = token2wav(audio_tokens.to("cuda:0"))

        # sf.write(os.path.join(args.save_path, filename.replace(".wav", "_decoded.wav")), decoded_audio.detach().cpu().numpy(), 24000)
        # import ipdb;ipdb.set_trace()


        


