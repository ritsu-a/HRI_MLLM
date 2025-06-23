class Qwen2_5OmniTalkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = Qwen2_5OmniTalkerConfig
    base_model_prefix = "talker"

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)

        self.thinker_to_talker_proj = nn.Linear(config.embedding_size, config.hidden_size)

        self.model = Qwen2_5OmniTalkerModel(config)
        self.codebook_size = config.vocab_size
        self.codec_head = nn.Linear(config.hidden_size, self.codebook_size, bias=False)

        self.codec_bos_token = config.tts_codec_start_token_id
        self.codec_eos_token = config.tts_codec_end_token_id
        self.codec_pad_token = config.tts_codec_pad_token_id
        self.codec_mask_token = config.tts_codec_mask_token_id

        self.text_bos_token = config.tts_text_start_token_id
        self.text_eos_token = config.tts_text_end_token_id
        self.text_pad_token = config.tts_text_pad_token_id

        self.spatial_merge_size = self.config.spatial_merge_size
        self.rope_deltas = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        input_text_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_audio_in_video: Optional[bool] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2_5OmniTalkerCausalLMOutputWithPast]:
        r"""
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        thinker_reply_part (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Hidden states from the thinker model's output that represent the text reply part to be processed.
        input_text_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Input token IDs for text-only content, used for position calculation in multimodal contexts.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, Qwen2_5OmniTalkerForConditionalGeneration

        >>> model = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_text_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )

                inputs_embeds[:, -1, :] += self.get_input_embeddings()(
                    torch.tensor([self.codec_bos_token], dtype=torch.long, device=inputs_embeds.device)
                )
                inputs_embeds[:, -2, :] += self.get_input_embeddings()(
                    torch.tensor([self.codec_pad_token], dtype=torch.long, device=inputs_embeds.device)
                )
                self.rope_deltas = rope_deltas

            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Inference tokens after second token
            codec_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]

        talker_lm_input = self.thinker_to_talker_proj(inputs_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=talker_lm_input,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.codec_head(hidden_states)
        logits = logits.float()

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            thinker_reply_part=thinker_reply_part,
        )

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        # Talker needs to calculate cache_position with input_ids, so pop inputs_embeds temporarily
        inputs_embeds = model_kwargs.pop("inputs_embeds")
        model_kwargs = super()._get_initial_cache_position(seq_length, device, model_kwargs)
        model_kwargs["inputs_embeds"] = inputs_embeds
        return model_kwargs

    # prepare inputs for talker lm generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            cache_position,
            use_cache=use_cache,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "thinker_reply_part", None) is not None:
            model_kwargs["thinker_reply_part"] = outputs.thinker_reply_part

        return model_kwargs

