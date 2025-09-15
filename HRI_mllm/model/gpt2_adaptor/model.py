from typing import Union
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class MixedInputGPT2(GPT2LMHeadModel):
    def __init__(self, config, audio_hidden_size=3584):
        super().__init__(config)
        
        # 如果输入的hidden size与GPT-2的hidden size不一致，添加线性层进行对齐
        if audio_hidden_size is not None and audio_hidden_size != config.hidden_size:
            self.input_projection = nn.Linear(audio_hidden_size, config.hidden_size)
        else:
            self.input_projection = nn.Identity()  # 使用恒等映射
            
        # 存储tokenizer用于处理token输入
        self.motion_tokenizer = nn.Embedding(config.vocab_size, config.hidden_size)

        saved_weight = torch.load("/root/pengyang/codebase/HRI_MLLM/output/motion_adaptor_v1/embed_tokens_weight.pt")
        self.audio_tokenizer = torch.nn.Embedding.from_pretrained(
            saved_weight,
            padding_idx=152063,
            freeze=True,
        ).to(self.device)
        for param in self.audio_tokenizer.parameters():
            param.requires_grad = False
        

        
    
    def forward(self, input_data, attention_mask=None, labels=None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                 **kwargs):
        """
        input_data: 包含token IDs和hidden states的混合输入
        input_types: 与input_data相同形状的张量，指示每个位置的输入类型
                    0表示token ID，1表示hidden state
        """

        batch_size, seq_length = input_data.shape[:2]
        # 创建初始的hidden states张量
        
        hidden_states = torch.zeros(
            batch_size, seq_length, self.config.hidden_size,
            device=input_data.device, dtype=torch.float32
        )
        
        # 处理audio tokens
        audio_mask = (labels == -100) * (attention_mask == 1)
        if audio_mask.any():
            audio_tokens = input_data[audio_mask]

            audio_embeds = self.audio_tokenizer(audio_tokens.long()).detach()

            projected_hidden = self.input_projection(audio_embeds.float())

            # 将token embeddings放入对应位置
            hidden_states[audio_mask] = projected_hidden
        
        # 处理motion tokens
        motion_mask = ~audio_mask
        if motion_mask.any():
            motion_tokens = input_data[motion_mask]
            motion_embeds = self.transformer.wte(motion_tokens.long())

            # 将projected hidden states放入对应位置
            hidden_states[motion_mask] = motion_embeds
        
        # # 添加位置编码
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_data.device)
        # position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # position_embeds = self.transformer.wpe(position_ids)
        # hidden_states = hidden_states + position_embeds
        
        # # 应用层归一化和dropout
        # hidden_states = self.transformer.drop(hidden_states)
        # hidden_states = self.transformer.ln_f(hidden_states)
        
        # 通过transformer层
        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return_dict = self.config.use_return_dict

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )