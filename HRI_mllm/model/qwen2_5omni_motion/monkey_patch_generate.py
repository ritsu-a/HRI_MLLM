from transformers import TextStreamer
import torch
from typing import Any, Callable, Optional, Union

def monkey_patch_qwen2_5omni_for_motion(model, streamer: Optional[TextStreamer] = None):

    def generate_adapter(*args, **kwargs):
        """
        Monkey patch the generate method of Qwen2.5-Omni to support motion generation.
        """
        if streamer is not None:
            kwargs["streamer"] = streamer
        return qwen2_5omni_motion_generate(*args, **kwargs)
    model.generate = generate_adapter


def monkey_patch_qwen2_5omni_for_motion_tts(model, streamer: Optional[TextStreamer] = None):

    def generate_adapter(*args, **kwargs):
        """
        Monkey patch the generate method of Qwen2.5-Omni to support motion generation.
        """
        if streamer is not None:
            kwargs["streamer"] = streamer
        return qwen2_5omni_motion_generate_tts(*args, **kwargs)
    model.generate = generate_adapter


@torch.no_grad()
def qwen2_5omni_motion_generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
    r"""
    Generate text response and audio from input.

    Args:
        input_ids (`Optional[torch.Tensor]`, *optional*):
            Input ids, should obtain from processor.
        speaker (`str` , defaults to "Chelsie"):
            Which speaker should be used in audio response.
        use_audio_in_video (`bool`, defaults to False):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        return_audio (`Optional[bool]`, *optional*):
            Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
        kwargs (*optional*):
            - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
            - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
            thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
    Returns:
        When `return_audio=False`:
            - **Text** (`torch.Tensor`): Generated text token sequence.
        When `return_audio=True`:
            - **Text** (`torch.Tensor`): Generated text token sequence.
            - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
    """
    if speaker not in self.speaker_map:
        raise ValueError(f"{speaker} is not available, available speakers: {self.speaker_map.keys()}")
    if return_audio and not self.has_talker:
        raise ValueError(
            "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
        )
    if return_audio is None:
        return_audio = self.has_talker
    if input_ids.shape[0] != 1 and return_audio:
        raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

    shared_kwargs = {"use_audio_in_video": use_audio_in_video}
    thinker_kwargs = {
        "max_new_tokens": thinker_max_new_tokens,
    }
    talker_kwargs = {
        "max_new_tokens": talker_max_new_tokens,
        "do_sample": talker_do_sample,
        "top_k": talker_top_k,
        "top_p": talker_top_p,
        "temperature": talker_temperature,
        "eos_token_id": talker_eos_token_id,
        "repetition_penalty": talker_repetition_penalty,
    }
    token2wav_kwargs = {}

    for key, value in kwargs.items():
        if key.startswith("thinker_"):
            thinker_kwargs[key[len("thinker_") :]] = value
        elif key.startswith("talker_"):
            talker_kwargs[key[len("talker_") :]] = value
        elif key.startswith("token2wav_"):
            token2wav_kwargs[key[len("token2wav_") :]] = value
        # Process special input values
        elif key == "feature_attention_mask":
            thinker_kwargs[key] = value
            talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
        elif key == "input_features" or key == "attention_mask":
            thinker_kwargs[key] = value
        # Put other key to shared kwargs
        elif key == "streamer":
            streamer = value
        else:
            shared_kwargs[key] = value

    # Merge kwargs
    for key, value in shared_kwargs.items():
        if key not in thinker_kwargs:
            thinker_kwargs[key] = value
        if key not in talker_kwargs:
            talker_kwargs[key] = value
        if key not in token2wav_kwargs:
            token2wav_kwargs[key] = value
    speaker_params = self.speaker_map[speaker]

    # 1. Generate from thinker module
    generate_audio = return_audio and self.has_talker
    if generate_audio:
        thinker_kwargs["output_hidden_states"] = True
        thinker_kwargs["return_dict_in_generate"] = True

    thinker_result = self.thinker.generate(input_ids=input_ids,streamer=streamer, **thinker_kwargs)
    if not generate_audio:
        return thinker_result

    # 2. Generate speech tokens from talker module
    embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(input_ids.device)
    if thinker_kwargs.get("input_features", None) is not None:
        audio_ids_mask = input_ids == self.config.thinker_config.audio_token_index
        audio_mask = audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        audio_mask_tensor = torch.zeros(
            [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype,
            device=input_ids.device,
        )
        embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)
    if thinker_kwargs.get("pixel_values", None) is not None:
        image_ids_mask = input_ids == self.config.thinker_config.image_token_index
        image_mask = image_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        image_mask_tensor = torch.zeros(
            [image_ids_mask.sum(), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype,
            device=input_ids.device,
        )
        embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)
    if thinker_kwargs.get("pixel_values_videos", None) is not None:
        video_ids_mask = input_ids == self.config.thinker_config.video_token_index
        video_mask = video_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        video_mask_tensor = torch.zeros(
            [video_ids_mask.sum(), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype,
            device=input_ids.device,
        )
        embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)

    processed_thinker_hidden = (
        (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
    ) + thinker_result.hidden_states[1:]
    thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(input_ids.device)
    thinker_token_embeds = [
        token_hidden_states[0].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
    ]
    thinker_hidden_states = [
        token_hidden_states[-1].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
    ]

    talker_text_bos_token = speaker_params["bos_token"]
    talker_input_text_ids = torch.cat(
        [
            input_ids,
            torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device),
            thinker_generate_ids[:, :1],
        ],
        dim=-1,
    )

    talker_input_ids = torch.cat(
        [
            torch.full_like(input_ids, fill_value=self.talker.codec_mask_token),
            torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=input_ids.device),
            torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=input_ids.device),
        ],
        dim=1,
    )

    thinker_embed_tokens = self.thinker.get_input_embeddings()
    thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
    talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
    talker_text_bos_token = torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device)
    talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(input_ids.device)
    talker_inputs_embeds = torch.cat(
        [
            talker_inputs_embeds,
            talker_text_bos_embed,
            thinker_reply_part[:, :1, :],
        ],
        dim=1,
    )

    eos_embedding = thinker_embed_tokens(
        torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=input_ids.device)
    ).to(input_ids.device)

    pad_embedding = thinker_embed_tokens(
        torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=input_ids.device)
    ).to(input_ids.device)   

    thinker_reply_part = torch.cat(
        [
            thinker_reply_part[:, 1:, :],
            eos_embedding,
            pad_embedding,
        ],
        dim=1,
    )

    talker_attention_mask = None
    if "attention_mask" in kwargs:
        talker_attention_mask = torch.cat(
            [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
        ).to(input_ids.device)

    


    talker_result = self.talker.generate(
        input_ids=talker_input_ids,
        input_text_ids=talker_input_text_ids,
        thinker_reply_part=thinker_reply_part,
        inputs_embeds=talker_inputs_embeds,
        attention_mask=talker_attention_mask,
        suppress_tokens=[self.talker.codec_bos_token],
        **{k: (v.to(input_ids.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
    )
    talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

    # 3. Generate wavs from code
    if self.token2wav.dtype != torch.float:
        self.token2wav.float()

    wav = self.token2wav(
        talker_generate_codes.to(input_ids.device),
        conditioning=speaker_params["cond"].to(input_ids.device).float(),
        reference_mel=speaker_params["ref_mel"].to(input_ids.device).float(),
        **token2wav_kwargs,
    )

    return thinker_result.sequences, wav.float(), talker_generate_codes.detach().cpu()


@torch.no_grad()
def qwen2_5omni_motion_generate_tts(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
    r"""
    Generate text response and audio from input.

    Args:
        input_ids (`Optional[torch.Tensor]`, *optional*):
            Input ids, should obtain from processor.
        speaker (`str` , defaults to "Chelsie"):
            Which speaker should be used in audio response.
        use_audio_in_video (`bool`, defaults to False):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        return_audio (`Optional[bool]`, *optional*):
            Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
        kwargs (*optional*):
            - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
            - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
            thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
    Returns:
        When `return_audio=False`:
            - **Text** (`torch.Tensor`): Generated text token sequence.
        When `return_audio=True`:
            - **Text** (`torch.Tensor`): Generated text token sequence.
            - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
    """
    if speaker not in self.speaker_map:
        raise ValueError(f"{speaker} is not available, available speakers: {self.speaker_map.keys()}")
    if return_audio and not self.has_talker:
        raise ValueError(
            "Cannot use talker when talker module not initialized. Use `enable_talker` method or set enable_talker in config to enable talker."
        )
    if return_audio is None:
        return_audio = self.has_talker
    if input_ids.shape[0] != 1 and return_audio:
        raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")

    shared_kwargs = {"use_audio_in_video": use_audio_in_video}
    thinker_kwargs = {
        "max_new_tokens": thinker_max_new_tokens,
    }
    talker_kwargs = {
        "max_new_tokens": talker_max_new_tokens,
        "do_sample": talker_do_sample,
        "top_k": talker_top_k,
        "top_p": talker_top_p,
        "temperature": talker_temperature,
        "eos_token_id": talker_eos_token_id,
        "repetition_penalty": talker_repetition_penalty,
    }
    token2wav_kwargs = {}

    for key, value in kwargs.items():
        if key.startswith("thinker_"):
            thinker_kwargs[key[len("thinker_") :]] = value
        elif key.startswith("talker_"):
            talker_kwargs[key[len("talker_") :]] = value
        elif key.startswith("token2wav_"):
            token2wav_kwargs[key[len("token2wav_") :]] = value
        # Process special input values
        elif key == "feature_attention_mask":
            thinker_kwargs[key] = value
            talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
        elif key == "input_features" or key == "attention_mask":
            thinker_kwargs[key] = value
        # Put other key to shared kwargs
        elif key == "streamer":
            streamer = value
        else:
            shared_kwargs[key] = value

    # Merge kwargs
    for key, value in shared_kwargs.items():
        if key not in thinker_kwargs:
            thinker_kwargs[key] = value
        if key not in talker_kwargs:
            talker_kwargs[key] = value
        if key not in token2wav_kwargs:
            token2wav_kwargs[key] = value
    speaker_params = self.speaker_map[speaker]

    # 1. Generate from thinker module
    generate_audio = return_audio and self.has_talker
    if generate_audio:
        thinker_kwargs["output_hidden_states"] = True
        thinker_kwargs["return_dict_in_generate"] = True

    thinker_result = self.thinker.generate(input_ids=input_ids,streamer=streamer, **thinker_kwargs)
    if not generate_audio:
        return thinker_result
    
    if not streamer.check_tts_quality():
        raise ValueError("Lower quality TTS, discard TTS result.")

    # 2. Generate speech tokens from talker module
    embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(input_ids.device)
    if thinker_kwargs.get("input_features", None) is not None:
        audio_ids_mask = input_ids == self.config.thinker_config.audio_token_index
        audio_mask = audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        audio_mask_tensor = torch.zeros(
            [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype,
            device=input_ids.device,
        )
        embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)
    if thinker_kwargs.get("pixel_values", None) is not None:
        image_ids_mask = input_ids == self.config.thinker_config.image_token_index
        image_mask = image_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        image_mask_tensor = torch.zeros(
            [image_ids_mask.sum(), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype,
            device=input_ids.device,
        )
        embeds_to_talker.masked_scatter_(image_mask, image_mask_tensor)
    if thinker_kwargs.get("pixel_values_videos", None) is not None:
        video_ids_mask = input_ids == self.config.thinker_config.video_token_index
        video_mask = video_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
        video_mask_tensor = torch.zeros(
            [video_ids_mask.sum(), embeds_to_talker.shape[-1]],
            dtype=embeds_to_talker.dtype,
            device=input_ids.device,
        )
        embeds_to_talker.masked_scatter_(video_mask, video_mask_tensor)

    processed_thinker_hidden = (
        (embeds_to_talker,) + thinker_result.hidden_states[0][1:],
    ) + thinker_result.hidden_states[1:]
    thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(input_ids.device)
    thinker_token_embeds = [
        token_hidden_states[0].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
    ]
    thinker_hidden_states = [
        token_hidden_states[-1].to(input_ids.device) for token_hidden_states in processed_thinker_hidden
    ]

    talker_text_bos_token = speaker_params["bos_token"]
    talker_input_text_ids = torch.cat(
        [
            input_ids,
            torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device),
            thinker_generate_ids[:, :1],
        ],
        dim=-1,
    )

    talker_input_ids = torch.cat(
        [
            torch.full_like(input_ids, fill_value=self.talker.codec_mask_token),
            torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=input_ids.device),
            torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=input_ids.device),
        ],
        dim=1,
    )

    thinker_embed_tokens = self.thinker.get_input_embeddings()
    thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
    talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
    talker_text_bos_token = torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=input_ids.device)
    talker_text_bos_embed = thinker_embed_tokens(talker_text_bos_token).to(input_ids.device)
    talker_inputs_embeds = torch.cat(
        [
            talker_inputs_embeds,
            talker_text_bos_embed,
            thinker_reply_part[:, :1, :],
        ],
        dim=1,
    )

    eos_embedding = thinker_embed_tokens(
        torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=input_ids.device)
    ).to(input_ids.device)

    pad_embedding = thinker_embed_tokens(
        torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=input_ids.device)
    ).to(input_ids.device)   

    thinker_reply_part = torch.cat(
        [
            thinker_reply_part[:, 1:, :],
            eos_embedding,
            pad_embedding,
        ],
        dim=1,
    )

    talker_attention_mask = None
    if "attention_mask" in kwargs:
        talker_attention_mask = torch.cat(
            [kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1
        ).to(input_ids.device)

    


    talker_result = self.talker.generate(
        input_ids=talker_input_ids,
        input_text_ids=talker_input_text_ids,
        thinker_reply_part=thinker_reply_part,
        inputs_embeds=talker_inputs_embeds,
        attention_mask=talker_attention_mask,
        suppress_tokens=[self.talker.codec_bos_token],
        **{k: (v.to(input_ids.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
    )
    talker_generate_codes = talker_result[:, talker_input_ids.shape[1] : -1]

    # 3. Generate wavs from code
    if self.token2wav.dtype != torch.float:
        self.token2wav.float()

    wav = self.token2wav(
        talker_generate_codes.to(input_ids.device),
        conditioning=speaker_params["cond"].to(input_ids.device).float(),
        reference_mel=speaker_params["ref_mel"].to(input_ids.device).float(),
        **token2wav_kwargs,
    )

    return thinker_result.sequences, wav.float(), talker_generate_codes.detach().cpu()