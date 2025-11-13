"""
Hidden State Based Adaptor - 直接使用Kimi的hidden states

原理：
- 不使用离散的audio tokens
- 直接将Kimi最后一层的hidden states传给GPT2 adaptor
- 完全可微，无需任何trick

优点：
- 梯度流畅，无偏估计
- 不丢失信息（相比离散化）
- 最简单直接

缺点：
- 需要Kimi model支持输出hidden states
- 如果Kimi很大，可能需要使用gradient checkpointing节省内存
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class HiddenStateGPT2Adaptor(GPT2LMHeadModel):
    """
    接受hidden states作为输入的GPT2 Adaptor
    
    训练模式：接受Kimi的hidden states（连续，可微）
    推理模式：接受离散token ids（与原来一样）
    """
    
    def __init__(self, config, kimi_hidden_size=3584, use_hidden_projection=True):
        super().__init__(config)
        
        # Hidden state projection layer
        # 将Kimi的hidden size对齐到GPT2的hidden size
        if use_hidden_projection and kimi_hidden_size != config.hidden_size:
            self.hidden_projection = nn.Linear(kimi_hidden_size, config.hidden_size)
            print(f"✅ Using hidden projection: {kimi_hidden_size} → {config.hidden_size}")
        else:
            self.hidden_projection = nn.Identity()
        
        # Motion tokenizer（用于motion tokens）
        self.motion_tokenizer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 可选：audio tokenizer（仅用于推理时的离散tokens）
        self.audio_tokenizer = None
        self.audio_vocab_size = None
        
        # Special tokens
        self.audio_gesture_start_token_id = getattr(config, 'audio_gesture_start_token_id', 512*2 + 3)
        self.audio_gesture_end_token_id = getattr(config, 'audio_gesture_end_token_id', 512*2 + 5)
        self.gesture_start_token_id = getattr(config, 'gesture_start_token_id', 512*2 + 2)
        self.gesture_end_token_id = getattr(config, 'gesture_end_token_id', 512*2 + 4)
    
    def load_audio_tokenizer_for_inference(self, weight_path, audio_vocab_size=152064, freeze=True):
        """加载audio tokenizer（仅用于推理模式）"""
        saved_weight = torch.load(weight_path, weights_only=True)
        self.audio_tokenizer = nn.Embedding.from_pretrained(saved_weight, freeze=freeze)
        self.audio_vocab_size = audio_vocab_size
        
        if freeze:
            for param in self.audio_tokenizer.parameters():
                param.requires_grad = False
        
        print(f"✅ Loaded audio tokenizer for inference (vocab_size={audio_vocab_size})")
    
    def forward(
        self,
        input_data=None,  # 离散token ids [batch, seq_len]（推理时使用）
        audio_hidden_states=None,  # 🔥 Kimi的hidden states [batch, seq_len, hidden_dim]（训练时使用）
        attention_mask=None,
        labels=None,
        position_ids=None,
        **kwargs
    ):
        """
        前向传播
        
        两种模式：
        1. 训练模式：传入audio_hidden_states（来自Kimi的连续表示）
        2. 推理模式：传入input_data（离散token ids）
        
        Args:
            input_data: 离散token ids（推理用）
            audio_hidden_states: Kimi的hidden states（训练用，可微！）
            attention_mask: attention mask
            labels: ground truth labels（motion tokens）
        """
        
        # 🔥 模式1：训练模式 - 使用hidden states（完全可微）
        if audio_hidden_states is not None and self.training:
            batch_size, seq_len, kimi_hidden_dim = audio_hidden_states.shape
            device = audio_hidden_states.device
            
            print(f"🔥 Using hidden state mode (shape={audio_hidden_states.shape})")
            
            # Project Kimi hidden states to GPT2 hidden size
            projected_audio_hidden = self.hidden_projection(audio_hidden_states)
            # [batch, seq_len, gpt2_hidden_dim]
            
            # 初始化inputs_embeds
            inputs_embeds = torch.zeros(
                batch_size, seq_len, self.config.hidden_size,
                dtype=projected_audio_hidden.dtype,
                device=device
            )
            
            # 识别audio和motion的位置
            if labels is not None:
                audio_mask = (labels == -100)  # audio positions
                motion_mask = (labels != -100)  # motion positions
                
                # Audio positions: 使用projected hidden states
                if audio_mask.any():
                    inputs_embeds[audio_mask] = projected_audio_hidden[audio_mask]
                
                # Motion positions: 使用motion token embeddings
                if motion_mask.any() and input_data is not None:
                    motion_tokens = input_data[motion_mask]
                    motion_embeds = self.transformer.wte(motion_tokens.long())
                    inputs_embeds[motion_mask] = motion_embeds
            else:
                # 如果没有labels，假设全是audio
                inputs_embeds = projected_audio_hidden
        
        # 🔥 模式2：推理模式 - 使用离散tokens
        else:
            if input_data is None:
                raise ValueError("input_data or audio_hidden_states is required")
            
            batch_size, seq_len = input_data.shape
            device = input_data.device
            
            inputs_embeds = torch.zeros(
                batch_size, seq_len, self.config.hidden_size,
                dtype=torch.float32,
                device=device
            )
            
            if attention_mask is None or labels is None:
                # 没有明确标记，假设都是audio tokens
                audio_mask = torch.ones_like(input_data, dtype=torch.bool)
            else:
                audio_mask = (labels == -100)
            
            # 处理audio tokens
            if audio_mask.any() and self.audio_tokenizer is not None:
                audio_tokens = input_data[audio_mask]
                valid_mask = (audio_tokens < self.audio_vocab_size) & (audio_tokens >= 0)
                
                if valid_mask.any():
                    valid_tokens = audio_tokens[valid_mask]
                    audio_embeds = self.audio_tokenizer(valid_tokens.long())
                    projected = self.hidden_projection(audio_embeds)
                    
                    # 需要创建一个临时mask来正确索引
                    temp_embeds = torch.zeros(
                        audio_tokens.shape[0], self.config.hidden_size,
                        device=device, dtype=projected.dtype
                    )
                    temp_embeds[valid_mask] = projected
                    inputs_embeds[audio_mask] = temp_embeds
            
            # 处理motion tokens
            motion_mask = ~audio_mask & (attention_mask == 1) if attention_mask is not None else ~audio_mask
            if motion_mask.any():
                motion_tokens = input_data[motion_mask]
                motion_embeds = self.transformer.wte(motion_tokens.long())
                inputs_embeds[motion_mask] = motion_embeds
        
        # 通过GPT2 transformer
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        
        # 计算loss
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class EndToEndHiddenStateModel(nn.Module):
    """
    端到端的Kimi + Hidden State Adaptor
    
    这是最简单、最直接的方案
    """
    
    def __init__(
        self,
        kimi_model,
        gpt2_adaptor: HiddenStateGPT2Adaptor,
        freeze_kimi=False,
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        
        self.kimi_model = kimi_model
        self.gpt2_adaptor = gpt2_adaptor
        
        # 冻结策略
        if freeze_kimi:
            for param in self.kimi_model.parameters():
                param.requires_grad = False
            print("❄️  Kimi model frozen")
        else:
            print("🔥 Kimi model trainable")
        
        # Gradient checkpointing（节省显存）
        if use_gradient_checkpointing:
            if hasattr(kimi_model, 'gradient_checkpointing_enable'):
                kimi_model.gradient_checkpointing_enable()
                print("✅ Enabled gradient checkpointing for Kimi")
            if hasattr(gpt2_adaptor, 'gradient_checkpointing_enable'):
                gpt2_adaptor.gradient_checkpointing_enable()
                print("✅ Enabled gradient checkpointing for GPT2 Adaptor")
    
    def forward(
        self,
        user_audio_tokens,  # [batch, seq_len]
        user_audio_features=None,  # Optional continuous features
        motion_tokens_gt=None,  # [batch, motion_seq_len]
        attention_mask=None,
        **kwargs
    ):
        """
        端到端前向传播
        
        流程：
        user_audio → Kimi(output_hidden_states=True) → hidden states → GPT2 Adaptor → motion loss
        
        ✅ 完全可微，梯度顺畅流动
        """
        
        # 1️⃣ Kimi model生成assistant audio的hidden states
        kimi_outputs = self.kimi_model(
            audio_input_ids=user_audio_tokens,
            continous_feature=user_audio_features,
            output_hidden_states=True,  # 🔥 关键：输出hidden states
            return_dict=True,
            **kwargs
        )
        
        # 获取最后一层的hidden states
        # kimi_outputs.hidden_states是一个tuple: (layer0, layer1, ..., layerN)
        assistant_audio_hidden = kimi_outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        
        print(f"✅ Kimi hidden states shape: {assistant_audio_hidden.shape}")
        print(f"   requires_grad: {assistant_audio_hidden.requires_grad}")
        
        # 2️⃣ 构建interleaved sequence的labels
        # 这里需要根据你的数据格式构建
        # 简化版本：假设已经构建好了
        
        # 3️⃣ GPT2 adaptor处理（直接使用hidden states！）
        adaptor_outputs = self.gpt2_adaptor(
            audio_hidden_states=assistant_audio_hidden,  # 🔥 传入hidden states
            input_data=motion_tokens_gt,  # motion tokens（用于构建完整序列）
            attention_mask=attention_mask,
            labels=kwargs.get('labels'),
        )
        
        motion_loss = adaptor_outputs.loss
        
        # 4️⃣ 可选：添加audio的辅助loss
        # 如果Kimi也需要监督信号
        audio_loss = None
        if hasattr(kimi_outputs, 'loss') and kimi_outputs.loss is not None:
            audio_loss = kimi_outputs.loss
        
        return {
            'motion_loss': motion_loss,
            'audio_loss': audio_loss,
            'total_loss': motion_loss + (audio_loss if audio_loss is not None else 0),
            'logits': adaptor_outputs.logits,
        }


# ==================== 修改Kimi model使其返回hidden states ====================

def patch_kimi_for_hidden_states(kimi_model):
    """
    如果Kimi model默认不返回hidden states，可以使用这个函数patch它
    
    示例用法：
        kimi_model = load_kimi_model(...)
        patch_kimi_for_hidden_states(kimi_model)
    """
    
    original_generate_loop = kimi_model._generate_loop
    
    def new_generate_loop(*args, output_hidden_states=False, **kwargs):
        """Wrapper to enable output_hidden_states"""
        if output_hidden_states:
            kwargs['output_hidden_states'] = True
        return original_generate_loop(*args, **kwargs)
    
    kimi_model._generate_loop = new_generate_loop
    print("✅ Patched Kimi model to support output_hidden_states")


# ==================== LoRA微调方案（推荐） ====================

def setup_lora_finetuning(kimi_model, gpt2_adaptor, lora_r=8, lora_alpha=16):
    """
    使用LoRA只微调adaptor部分，Kimi保持frozen
    这是最经济的方案
    
    Args:
        kimi_model: Kimi model（会被冻结）
        gpt2_adaptor: GPT2 adaptor
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
    """
    from peft import get_peft_model, LoraConfig, TaskType
    
    # 冻结Kimi
    for param in kimi_model.parameters():
        param.requires_grad = False
    print("❄️  Kimi model frozen")
    
    # 冻结GPT2 adaptor的大部分参数
    for param in gpt2_adaptor.parameters():
        param.requires_grad = False
    
    # 只训练projection layer
    if hasattr(gpt2_adaptor, 'hidden_projection'):
        for param in gpt2_adaptor.hidden_projection.parameters():
            param.requires_grad = True
        print("🔥 Hidden projection layer trainable")
    
    # 可选：使用LoRA微调GPT2的attention层
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],  # GPT2的attention modules
    )
    
    gpt2_adaptor = get_peft_model(gpt2_adaptor, lora_config)
    print(f"✅ Applied LoRA to GPT2 adaptor (r={lora_r}, alpha={lora_alpha})")
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in gpt2_adaptor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in gpt2_adaptor.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return kimi_model, gpt2_adaptor


# ==================== 使用示例 ====================

def example_training_script():
    """完整的训练示例"""
    
    print("\n" + "="*60)
    print("Hidden State Based End-to-End Training")
    print("="*60 + "\n")
    
    # 1. 加载模型
    # kimi_model = load_kimi_model(...)
    # gpt2_config = GPT2Config(...)
    # gpt2_adaptor = HiddenStateGPT2Adaptor(gpt2_config, kimi_hidden_size=3584)
    
    # 2. 创建端到端模型
    # model = EndToEndHiddenStateModel(kimi_model, gpt2_adaptor, freeze_kimi=True)
    
    # 3. 或者使用LoRA
    # kimi_model, gpt2_adaptor = setup_lora_finetuning(kimi_model, gpt2_adaptor)
    # model = EndToEndHiddenStateModel(kimi_model, gpt2_adaptor, freeze_kimi=True)
    
    # 4. 训练循环
    # for batch in dataloader:
    #     outputs = model(
    #         user_audio_tokens=batch['user_audio_tokens'],
    #         motion_tokens_gt=batch['motion_tokens'],
    #         labels=batch['labels'],
    #     )
    #     
    #     loss = outputs['total_loss']
    #     loss.backward()  # ✅ 梯度会流向projection layer和LoRA参数
    #     optimizer.step()
    
    print("✅ Training script example completed")


if __name__ == "__main__":
    example_training_script()


