#!/usr/bin/env python3
"""
统一的Kimi-Motion模型
将Kimi模型、MLP映射层和GPT2 adaptor合并成一个模型
使用Kimi模型的text和audio hidden state混合作为adaptor输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from transformers import GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from HRI_mllm.model.kimi_motion.model import MoonshotKimiaForCausalLM
from HRI_mllm.model.kimi_motion.config import KimiAudioConfig


class HiddenStateMixer(nn.Module):
    """
    混合Kimi模型的text和audio hidden states
    """
    def __init__(self, text_hidden_size: int, audio_hidden_size: int, output_hidden_size: int):
        super().__init__()
        self.text_hidden_size = text_hidden_size
        self.audio_hidden_size = audio_hidden_size
        self.output_hidden_size = output_hidden_size
        
        # 将text和audio hidden states映射到相同维度
        self.text_projection = nn.Linear(text_hidden_size, output_hidden_size)
        self.audio_projection = nn.Linear(audio_hidden_size, output_hidden_size)
        
        # 混合权重学习
        self.mix_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # [text_weight, audio_weight]
        
        # 可选的融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_hidden_size * 2, output_hidden_size),
            nn.ReLU(),
            nn.Linear(output_hidden_size, output_hidden_size)
        )
        
    def forward(self, text_hidden_states: torch.Tensor, audio_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_hidden_states: [batch_size, seq_len, text_hidden_size]
            audio_hidden_states: [batch_size, seq_len, audio_hidden_size]
        Returns:
            mixed_hidden_states: [batch_size, seq_len, output_hidden_size]
        """
        # 投影到相同维度
        text_proj = self.text_projection(text_hidden_states)  # [batch_size, seq_len, output_hidden_size]
        audio_proj = self.audio_projection(audio_hidden_states)  # [batch_size, seq_len, output_hidden_size]
        
        # 应用softmax确保权重和为1
        weights = F.softmax(self.mix_weights, dim=0)
        
        # 加权混合
        mixed = weights[0] * text_proj + weights[1] * audio_proj
        
        # 可选：通过融合层进一步处理
        if hasattr(self, 'fusion_layer'):
            # 拼接text和audio特征
            concat_features = torch.cat([text_proj, audio_proj], dim=-1)  # [batch_size, seq_len, output_hidden_size*2]
            fused_features = self.fusion_layer(concat_features)  # [batch_size, seq_len, output_hidden_size]
            
            # 残差连接
            mixed = mixed + fused_features
        
        return mixed


class UnifiedKimiMotionModel(nn.Module):
    """
    统一的Kimi-Motion模型
    包含Kimi模型、hidden state混合器和GPT2 adaptor
    """
    
    def __init__(self, 
                 kimi_model_path: str,
                 gpt2_config: GPT2Config,
                 freeze_kimi: bool = True,
                 freeze_adaptor: bool = True,
                 train_mixer_only: bool = True,
                 motion_loss_weight: float = 1.0,
                 audio_loss_weight: float = 0.1,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 debug: bool = False):
        super().__init__()
        
        self.freeze_kimi = freeze_kimi
        self.freeze_adaptor = freeze_adaptor
        self.train_mixer_only = train_mixer_only
        self.motion_loss_weight = motion_loss_weight  # motion token loss权重（更高）
        self.audio_loss_weight = audio_loss_weight    # audio token loss权重（更低）
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.debug = debug
        
        # 只在非debug模式或主进程打印关键信息
        if not debug or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
            pass  # 移除加载信息，减少输出
        
        # 加载Kimi模型
        self.kimi_model = MoonshotKimiaForCausalLM.from_pretrained(
            kimi_model_path, 
            trust_remote_code=True
        )
        
        # 冻结Kimi模型参数，使用LoRA微调最后2层transformer
        self.use_lora = False
        if self.freeze_kimi:
            # 冻结所有参数
            for param in self.kimi_model.parameters():
                param.requires_grad = False
            
            # 尝试使用LoRA微调最后2层
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                
                # 为PEFT兼容性添加必要的方法（如果不存在）
                if not hasattr(self.kimi_model, 'prepare_inputs_for_generation'):
                    # 定义prepare_inputs_for_generation方法
                    def _prepare_inputs_for_generation(model_self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
                        """占位符方法，用于PEFT兼容性"""
                        if past_key_values is not None:
                            input_ids = input_ids[:, -1:]
                        return {
                            "input_ids": input_ids,
                            "past_key_values": past_key_values,
                            "attention_mask": attention_mask,
                        }
                    # 直接添加到类上
                    from types import MethodType
                    self.kimi_model.prepare_inputs_for_generation = MethodType(_prepare_inputs_for_generation, self.kimi_model)
                
                num_mimo_layers = len(self.kimi_model.model.mimo_layers)
                if num_mimo_layers >= 2:
                    # 为最后2层配置LoRA
                    # 目标模块：attention和MLP的线性层
                    target_modules = []
                    for i in range(num_mimo_layers - 2, num_mimo_layers):
                        layer_idx = i
                        # 添加attention层的目标模块
                        target_modules.extend([
                            f"model.mimo_layers.{layer_idx}.self_attn.q_proj",
                            f"model.mimo_layers.{layer_idx}.self_attn.k_proj",
                            f"model.mimo_layers.{layer_idx}.self_attn.v_proj",
                            f"model.mimo_layers.{layer_idx}.self_attn.o_proj",
                        ])
                        # 添加MLP层的目标模块（Qwen2MLP通常有gate_proj, up_proj, down_proj）
                        target_modules.extend([
                            f"model.mimo_layers.{layer_idx}.mlp.gate_proj",
                            f"model.mimo_layers.{layer_idx}.mlp.up_proj",
                            f"model.mimo_layers.{layer_idx}.mlp.down_proj",
                        ])
                    
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        target_modules=target_modules,
                        bias="none",
                    )
                    
                    # 将模型转换为PEFT模型（只对指定层添加LoRA）
                    self.kimi_model = get_peft_model(self.kimi_model, lora_config)
                    self.use_lora = True
                    
                    # 确保lm_head也是可训练的（用于计算loss）
                    for param in self.kimi_model.lm_head.parameters():
                        param.requires_grad = True
                    
                    print(f"✅ Kimi model using LoRA for last 2 audio transformer layers (layers {num_mimo_layers-2} to {num_mimo_layers-1})")
                    print(f"   - LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
                else:
                    # 如果层数少于2，对所有mimo_layers使用LoRA
                    target_modules = []
                    for i in range(num_mimo_layers):
                        target_modules.extend([
                            f"model.mimo_layers.{i}.self_attn.q_proj",
                            f"model.mimo_layers.{i}.self_attn.k_proj",
                            f"model.mimo_layers.{i}.self_attn.v_proj",
                            f"model.mimo_layers.{i}.self_attn.o_proj",
                            f"model.mimo_layers.{i}.mlp.gate_proj",
                            f"model.mimo_layers.{i}.mlp.up_proj",
                            f"model.mimo_layers.{i}.mlp.down_proj",
                        ])
                    
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        target_modules=target_modules,
                        bias="none",
                    )
                    
                    self.kimi_model = get_peft_model(self.kimi_model, lora_config)
                    self.use_lora = True
                    
                    for param in self.kimi_model.lm_head.parameters():
                        param.requires_grad = True
                    
                    print(f"✅ Kimi model using LoRA for all {num_mimo_layers} audio transformer layers")
                    print(f"   - LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
                    
            except (ImportError, AttributeError, Exception) as e:
                if isinstance(e, ImportError):
                    print("⚠️  peft library not found, falling back to full fine-tuning")
                else:
                    print(f"⚠️  LoRA initialization failed: {e}, falling back to full fine-tuning")
                # 确保use_lora标志为False
                self.use_lora = False
                # 回退到直接解冻最后2层
                num_mimo_layers = len(self.kimi_model.model.mimo_layers)
                if num_mimo_layers >= 2:
                    for layer in self.kimi_model.model.mimo_layers[-2:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    print(f"✅ Kimi model parameters frozen (except last 2 audio transformer layers: {num_mimo_layers-2} to {num_mimo_layers-1})")
                else:
                    for layer in self.kimi_model.model.mimo_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                    print(f"✅ Kimi model parameters frozen (unfroze all {num_mimo_layers} audio transformer layers)")
                
                # 解冻lm_head
                for param in self.kimi_model.lm_head.parameters():
                    param.requires_grad = True
        
        # 获取Kimi模型的hidden state维度
        kimi_config = self.kimi_model.config
        text_hidden_size = kimi_config.hidden_size  # 通常是4096
        audio_hidden_size = kimi_config.hidden_size  # 通常是4096
        
        # 初始化文本tokenizer（使用Kimi模型的tokenizer）
        from transformers import AutoTokenizer
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                kimi_model_path,
                trust_remote_code=True
            )
            print(f"✅ Text tokenizer loaded from {kimi_model_path}")
            
            # 获取extra_tokens（特别是kimia_text_blank）
            try:
                from kimia_infer.utils.special_tokens import instantiate_extra_tokens
                self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
                self.kimia_text_blank = self.extra_tokens.kimia_text_blank
                print(f"✅ Extra tokens loaded, kimia_text_blank={self.kimia_text_blank}")
            except Exception as e:
                print(f"⚠️  Failed to load extra tokens: {e}")
                # 尝试直接获取kimia_text_blank
                if hasattr(self.text_tokenizer, "special_tokens"):
                    try:
                        self.kimia_text_blank = self.text_tokenizer.special_tokens["<|im_kimia_text_blank|>"]
                    except:
                        self.kimia_text_blank = 18  # 默认值
                elif hasattr(self.text_tokenizer, "convert_tokens_to_ids"):
                    try:
                        self.kimia_text_blank = self.text_tokenizer.convert_tokens_to_ids("<|im_kimia_text_blank|>")
                    except:
                        self.kimia_text_blank = 18  # 默认值
                else:
                    self.kimia_text_blank = 18  # 默认值
                print(f"⚠️  Using default kimia_text_blank={self.kimia_text_blank}")
        except Exception as e:
            print(f"⚠️  Failed to load text tokenizer: {e}")
            self.text_tokenizer = None
            self.kimia_text_blank = 18  # 默认值
        
        # 创建hidden state混合器
        self.hidden_state_mixer = HiddenStateMixer(
            text_hidden_size=text_hidden_size,
            audio_hidden_size=audio_hidden_size,
            output_hidden_size=gpt2_config.hidden_size  # GPT2的hidden size，通常是768
        )
        
        # 创建GPT2 adaptor
        self.motion_adaptor = MixedInputGPT2(
            config=gpt2_config,
            audio_hidden_size=gpt2_config.hidden_size  # 现在输入是混合后的hidden state
        )
        
        # 冻结adaptor参数
        if self.freeze_adaptor:
            for param in self.motion_adaptor.parameters():
                param.requires_grad = False
            print("✅ Motion adaptor parameters frozen")
        
        # 设置训练模式
        if self.train_mixer_only:
            # 只训练hidden state mixer
            for param in self.hidden_state_mixer.parameters():
                param.requires_grad = True
            print("✅ Only hidden state mixer will be trained")
        
        print(f"✅ Unified model initialized successfully!")
        print(f"   - Kimi model: {'frozen' if self.freeze_kimi else 'trainable'}")
        print(f"   - Motion adaptor: {'frozen' if self.freeze_adaptor else 'trainable'}")
        print(f"   - Hidden state mixer: trainable")
    
    def forward(self, 
                user_text: Optional[List[str]] = None,
                user_audio_tokens: Optional[torch.Tensor] = None,
                assistant_audio_tokens: Optional[torch.Tensor] = None,
                motion_tokens: Optional[torch.Tensor] = None,
                interleaved_sequences: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):
        """
        前向传播
        不再使用interleaved_sequences中的audio token IDs，只使用Kimi的hidden states
        
        Args:
            user_text: 用户文本 [batch_size] (可选)
            user_audio_tokens: 用户音频token IDs [batch_size, seq_len]
            assistant_audio_tokens: 助手音频token IDs [batch_size, seq_len] (监督信号)
            motion_tokens: 动作token IDs [batch_size, seq_len] (监督信号)
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签 [batch_size, seq_len] (motion tokens)
        
        Returns:
            outputs: 包含logits和loss的输出
        """
        
        # Debug信息已关闭
        
        # 检查输入tensor的有效性
        if user_audio_tokens is None:
            raise ValueError("user_audio_tokens cannot be None")
        if not isinstance(user_audio_tokens, torch.Tensor):
            raise ValueError(f"user_audio_tokens must be a tensor, got {type(user_audio_tokens)}")
        
        # 获取设备（从模型参数或输入tensor）
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else user_audio_tokens.device
        
        # 使用user_audio_tokens作为Kimi模型的输入
        audio_input_ids = user_audio_tokens.to(device)
        
        # 确保assistant_audio_tokens也在正确的设备上
        if assistant_audio_tokens is not None:
            assistant_audio_tokens = assistant_audio_tokens.to(device)
        
        # 为Kimi模型创建正确的attention_mask（基于user_audio_tokens的长度）
        kimi_attention_mask = torch.ones(
            audio_input_ids.shape[0], audio_input_ids.shape[1],
            device=device, dtype=torch.long
        )
        
        # 处理用户文本（如果有的话）
        # 参考kimia_infer/api/prompt_manager.py的_tokenize_text方法
        text_input_ids = None
        if user_text is not None and self.text_tokenizer is not None:
            try:
                # 使用encode方法，参数bos=False, eos=False（与prompt_manager一致）
                if isinstance(user_text, list):
                    # 对列表中的每个文本进行编码
                    encoded_texts = []
                    for text in user_text:
                        if text is None or text.strip() == "":
                            encoded = []
                        else:
                            # 使用与prompt_manager相同的方式：encode(text, bos=False, eos=False)
                            encoded = self.text_tokenizer.encode(text, bos=False, eos=False)
                        # 截断到最大长度（512）
                        if len(encoded) > 512:
                            encoded = encoded[:512]
                        encoded_texts.append(encoded)
                    
                    # Padding到相同长度
                    if encoded_texts:
                        max_len = max(len(enc) for enc in encoded_texts)
                        max_len = min(max_len, 512)
                        
                        # 获取pad_token_id
                        pad_token_id = getattr(self.text_tokenizer, 'pad_token_id', None)
                        if pad_token_id is None:
                            # 如果没有pad_token_id，使用0
                            pad_token_id = 0
                        
                        # Padding所有序列到相同长度
                        padded_texts = []
                        for enc in encoded_texts:
                            pad_len = max_len - len(enc)
                            if pad_len > 0:
                                enc = enc + [pad_token_id] * pad_len
                            padded_texts.append(enc)
                        text_input_ids = torch.tensor(padded_texts, device=device, dtype=torch.long)
                    else:
                        # 如果所有文本都是空的，创建一个空tensor
                        text_input_ids = torch.zeros((len(user_text), 0), device=device, dtype=torch.long)
                else:
                    # 单个文本
                    if user_text is None or user_text.strip() == "":
                        text_input_ids = torch.zeros((1, 0), device=device, dtype=torch.long)
                    else:
                        # 使用与prompt_manager相同的方式：encode(text, bos=False, eos=False)
                        encoded = self.text_tokenizer.encode(user_text, bos=False, eos=False)
                        # 截断到最大长度
                        if len(encoded) > 512:
                            encoded = encoded[:512]
                        text_input_ids = torch.tensor([encoded], device=device, dtype=torch.long)
                
                if self.debug:
                    print(f"   - Text tokenized: {text_input_ids.shape}")
                    if text_input_ids.numel() > 0:
                        print(f"   - Text token IDs range: {text_input_ids.min().item()} to {text_input_ids.max().item()}")
            except Exception as e:
                if self.debug:
                    print(f"   ⚠️  Text tokenization failed: {e}")
                    import traceback
                    traceback.print_exc()
                # 如果tokenization失败，设置为None而不是继续训练
                raise ValueError(f"Text tokenization is required but failed: {e}")
        
        # 1. 通过Kimi模型获取hidden states和生成assistant audio tokens
        if self.debug:
            print(f"🔍 Calling Kimi model...")
        # 注意：即使Kimi模型被冻结，也需要enable_grad()以确保梯度能够通过hidden states传递到mixer
        # Kimi模型的参数已经被设置为requires_grad=False，所以不会更新参数
        with torch.enable_grad():
            try:
                # 构建Kimi模型调用参数
                kimi_kwargs = {
                    'input_ids': audio_input_ids,
                    'attention_mask': kimi_attention_mask,
                    'output_hidden_states': True,
                    'return_dict': True
                }
                
                # 只有当text_input_ids不为None时才添加
                # 注意：text_input_ids的长度必须与audio_input_ids的长度匹配，且位置对齐
                # 因为Kimi模型会将text embeddings加到audio embeddings上
                # 对于没有文本的位置，应该使用kimia_text_blank填充（与prompt_manager一致）
                if text_input_ids is not None:
                    batch_size, audio_seq_len = audio_input_ids.shape
                    text_seq_len = text_input_ids.shape[1]
                    
                    # 如果text序列长度不等于audio序列长度，需要进行padding
                    # 使用kimia_text_blank填充（与Kimi模型的处理方式一致）
                    if text_seq_len != audio_seq_len:
                        # 使用kimia_text_blank作为padding token（与prompt_manager一致）
                        # 在prompt_manager中，对于audio token位置，对应的text token就是kimia_text_blank
                        padded_text_input_ids = torch.full(
                            (batch_size, audio_seq_len),
                            self.kimia_text_blank,
                            device=text_input_ids.device,
                            dtype=text_input_ids.dtype
                        )
                        
                        # 将原始的text_input_ids复制到前面（左对齐）
                        # 剩余位置自动填充为kimia_text_blank
                        actual_len = min(text_seq_len, audio_seq_len)
                        padded_text_input_ids[:, :actual_len] = text_input_ids[:, :actual_len]
                        text_input_ids = padded_text_input_ids
                        
                        if self.debug:
                            print(f"   - Text input IDs padded from {text_seq_len} to {audio_seq_len} using kimia_text_blank={self.kimia_text_blank}")
                    else:
                        # 即使长度匹配，也要确保没有文本的位置使用kimia_text_blank
                        # 但这里我们假设用户提供的text_input_ids已经是正确对齐的
                        if self.debug:
                            print(f"   - Text input IDs length matches audio ({audio_seq_len})")
                    
                    kimi_kwargs['text_input_ids'] = text_input_ids
                else:
                    # 如果没有提供text_input_ids，创建全为kimia_text_blank的tensor
                    # 因为Kimi模型要求text_input_ids必须存在
                    batch_size, audio_seq_len = audio_input_ids.shape
                    text_input_ids = torch.full(
                        (batch_size, audio_seq_len),
                        self.kimia_text_blank,
                        device=device,
                        dtype=torch.long
                    )
                    kimi_kwargs['text_input_ids'] = text_input_ids
                    if self.debug:
                        print(f"   - Created text_input_ids filled with kimia_text_blank={self.kimia_text_blank}")
                
                # 添加labels以计算assistant audio token的loss
                if assistant_audio_tokens is not None:
                    kimi_kwargs['labels'] = assistant_audio_tokens.to(device)
                
                if self.debug:
                    print(f"   - Kimi model kwargs: {list(kimi_kwargs.keys())}")
                
                kimi_outputs = self.kimi_model(**kimi_kwargs)
                if self.debug:
                    print(f"✅ Kimi model call completed")
                    print(f"   - kimi_outputs type: {type(kimi_outputs)}")
                    print(f"   - kimi_outputs keys: {list(kimi_outputs.keys()) if hasattr(kimi_outputs, 'keys') else 'N/A'}")
            except Exception as e:
                print(f"❌ Error in Kimi model call: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # 获取text和audio的hidden states
        if hasattr(kimi_outputs, 'hidden_states') and kimi_outputs.hidden_states:
            # 获取最后一层的hidden states
            last_hidden_states = kimi_outputs.hidden_states[-1]
            
            if isinstance(last_hidden_states, tuple):
                text_hidden_states, audio_hidden_states = last_hidden_states
            else:
                # 如果只有一个hidden state，复制一份作为text和audio
                text_hidden_states = last_hidden_states
                audio_hidden_states = last_hidden_states
        else:
            # 这里需要从logits反推hidden states，或者修改Kimi模型以返回hidden states
            raise NotImplementedError("需要修改Kimi模型以返回hidden states")
        
        # 2. 混合hidden states
        try:
            mixed_hidden_states = self.hidden_state_mixer(text_hidden_states, audio_hidden_states)
        except Exception as e:
            print(f"❌ Error in hidden state mixer: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 3. 通过motion adaptor生成motion tokens
        # 直接使用labels，labels中-100的位置是audio token，其他位置是motion token
        # Adaptor的forward方法会自动处理这两种token类型
        
        # 注意：即使adaptor被冻结，也需要enable_grad()以确保梯度能够通过adaptor传递到mixer
        # Adaptor的参数已经被设置为requires_grad=False，所以不会更新参数
        with torch.enable_grad():
            # 使用MixedInputGPT2的forward方法
            # 它需要input_data（交错序列的token IDs）、labels和attention_mask
            # input_data包含audio tokens和motion tokens的ID
            
            # 但实际上我们不能直接用token IDs，因为我们需要用Kimi的hidden states替换audio token位置
            # 所以我们需要自己处理hidden states的构建
            
            # 获取交错序列的labels来判断哪些是audio，哪些是motion
            if labels is not None:
                audio_mask = (labels == -100)
                motion_mask = (labels != -100)
                
                batch_size, seq_len = labels.shape
                hidden_size = mixed_hidden_states.shape[-1]
                
                # 创建最终的hidden states序列
                final_hidden_states = torch.zeros(
                    batch_size, seq_len, hidden_size,
                    device=mixed_hidden_states.device,
                    dtype=mixed_hidden_states.dtype
                )
                
                # 将Kimi的hidden states放到audio token位置
                if audio_mask.any():
                    # 需要统计每个样本有多少audio tokens
                    for batch_idx in range(batch_size):
                        audio_positions = torch.where(audio_mask[batch_idx])[0]
                        if len(audio_positions) > 0:
                            # 获取该样本的hidden states
                            sample_hidden = mixed_hidden_states[batch_idx]
                            # 确保长度匹配
                            use_len = min(sample_hidden.shape[0], len(audio_positions))
                            final_hidden_states[batch_idx, audio_positions[:use_len]] = sample_hidden[:use_len]
                
                # 将motion token embeddings放到motion token位置
                if motion_mask.any():
                    # 从interleaved sequences中获取motion token IDs
                    if interleaved_sequences is not None:
                        motion_token_ids = interleaved_sequences[motion_mask]
                        motion_embeddings = self.motion_adaptor.transformer.wte(motion_token_ids)
                        final_hidden_states[motion_mask] = motion_embeddings
                    else:
                        # 如果没有interleaved_sequences，使用labels中的motion token IDs
                        motion_tokens_in_labels = labels[motion_mask]
                        motion_embeddings = self.motion_adaptor.transformer.wte(motion_tokens_in_labels)
                        final_hidden_states[motion_mask] = motion_embeddings
                
                # 创建attention mask
                attention_mask = torch.ones(batch_size, seq_len, device=mixed_hidden_states.device, dtype=torch.long)
                
                # 通过adaptor
                adaptor_outputs = self._forward_adaptor_with_hidden_states(
                    final_hidden_states,
                    attention_mask,
                    labels,
                    **kwargs
                )
            else:
                adaptor_outputs = None
        
        # 计算Kimi模型的audio token loss
        kimi_audio_loss = None
        if assistant_audio_tokens is not None and hasattr(kimi_outputs, 'logits'):
            # Kimi模型返回的logits是元组 (text_logits, audio_logits)
            if isinstance(kimi_outputs.logits, tuple) and len(kimi_outputs.logits) >= 2:
                audio_logits = kimi_outputs.logits[1]  # audio_logits
                # 确保assistant_audio_tokens在正确的设备上
                assistant_audio_tokens_device = assistant_audio_tokens.to(audio_logits.device)
                # 计算audio token的loss
                shift_logits = audio_logits[..., :-1, :].contiguous()
                shift_labels = assistant_audio_tokens_device[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                kimi_audio_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            elif hasattr(kimi_outputs, 'loss') and kimi_outputs.loss is not None:
                # 如果Kimi模型已经计算了loss，直接使用
                kimi_audio_loss = kimi_outputs.loss
        
        # 组合Kimi的loss和Adaptor的loss（使用权重）
        total_loss = None
        if adaptor_outputs is not None:
            # Motion token loss（更重要，权重更高）
            motion_loss = adaptor_outputs.loss if adaptor_outputs.loss is not None else None
            if motion_loss is not None:
                total_loss = self.motion_loss_weight * motion_loss
            
            # Audio token loss（权重较低）
            if kimi_audio_loss is not None:
                audio_weighted_loss = self.audio_loss_weight * kimi_audio_loss
                if total_loss is None:
                    total_loss = audio_weighted_loss
                else:
                    total_loss = total_loss + audio_weighted_loss
        
        # 返回组合后的输出（同时保存motion_loss和audio_loss的原始值，用于日志显示）
        if adaptor_outputs is not None:
            from transformers.modeling_outputs import CausalLMOutputWithPast
            output = CausalLMOutputWithPast(
                loss=total_loss,
                logits=adaptor_outputs.logits,
                past_key_values=adaptor_outputs.past_key_values,
                hidden_states=adaptor_outputs.hidden_states,
                attentions=adaptor_outputs.attentions,
            )
            # 添加motion_loss和audio_loss的原始值（不应用权重）用于日志
            if motion_loss is not None:
                output.motion_loss = motion_loss
            if kimi_audio_loss is not None:
                output.audio_loss = kimi_audio_loss
            return output
        else:
            return adaptor_outputs
    
    def _create_interleaved_attention_mask(self, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        创建交错的attention mask
        奇数位：audio位置，偶数位：motion位置
        """
        if attention_mask is None:
            return None
        
        batch_size, seq_len = attention_mask.shape
        
        # 创建交错的attention mask
        interleaved_attention_mask = torch.zeros(
            batch_size, seq_len * 2,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        
        # 填充奇数位（audio位置）
        interleaved_attention_mask[:, 1::2] = attention_mask
        
        # 填充偶数位（motion位置）
        if labels is not None:
            motion_mask = (labels != -100)
            motion_positions = torch.where(motion_mask)[1]
            interleaved_positions = motion_positions * 2
            
            valid_positions = interleaved_positions < interleaved_attention_mask.shape[1]
            if valid_positions.any():
                interleaved_attention_mask[:, interleaved_positions[valid_positions]] = 1
        
        return interleaved_attention_mask
    
    def _create_interleaved_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        创建交错的labels
        奇数位：-100（audio位置），偶数位：motion token ID
        """
        if labels is None:
            return None
        
        batch_size, seq_len = labels.shape
        
        # 创建交错的labels
        interleaved_labels = torch.full(
            (batch_size, seq_len * 2), -100,
            device=labels.device,
            dtype=labels.dtype
        )
        
        # 填充偶数位（motion tokens）
        motion_mask = (labels != -100)
        motion_tokens = labels[motion_mask]
        motion_positions = torch.where(motion_mask)[1]
        interleaved_positions = motion_positions * 2
        
        valid_positions = interleaved_positions < interleaved_labels.shape[1]
        if valid_positions.any():
            interleaved_labels[:, interleaved_positions[valid_positions]] = motion_tokens[valid_positions]
        
        return interleaved_labels
    
    def _forward_adaptor_with_hidden_states(self, 
                                           hidden_states: torch.Tensor,
                                           attention_mask: Optional[torch.Tensor],
                                           labels: Optional[torch.Tensor],
                                           **kwargs):
        """
        使用hidden states作为输入调用adaptor
        绕过MixedInputGPT2的token处理逻辑
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # 直接使用hidden states，跳过token embedding步骤
        # 通过transformer层
        transformer_outputs = self.motion_adaptor.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 获取最终的hidden states
        final_hidden_states = transformer_outputs[0]
        
        # 计算logits
        logits = self.motion_adaptor.lm_head(final_hidden_states)
        
        # 计算损失（如果有labels）
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 只计算motion token位置的损失
            motion_mask = (shift_labels != -100)
            if motion_mask.any():
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        
        # 返回类似CausalLMOutputWithPast的输出
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values if len(transformer_outputs) > 1 else None,
            hidden_states=transformer_outputs.hidden_states if len(transformer_outputs) > 2 else None,
            attentions=transformer_outputs.attentions if len(transformer_outputs) > 3 else None,
        )
    
    def generate_motion_tokens(self, 
                             audio_input_ids: torch.Tensor,
                             text_input_ids: Optional[torch.Tensor] = None,
                             whisper_input_feature: Optional[torch.Tensor] = None,
                             attention_mask: Optional[torch.Tensor] = None,
                             max_new_tokens: int = 256,
                             temperature: float = 0.8,
                             top_k: int = 50,
                             repetition_penalty: float = 1.1,
                             **kwargs):
        """
        生成motion tokens
        
        Args:
            audio_input_ids: 音频token IDs
            text_input_ids: 文本token IDs (可选)
            whisper_input_feature: Whisper特征 (可选)
            attention_mask: 注意力掩码
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: top-k采样
            repetition_penalty: 重复惩罚
        
        Returns:
            generated_motion_tokens: 生成的motion token序列
        """
        self.eval()
        
        with torch.no_grad():
            # 获取Kimi模型的hidden states
            kimi_outputs = self.kimi_model(
                input_ids=audio_input_ids,
                text_input_ids=text_input_ids,
                whisper_input_feature=whisper_input_feature,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 获取text和audio的hidden states
            if hasattr(kimi_outputs, 'hidden_states') and kimi_outputs.hidden_states:
                last_hidden_states = kimi_outputs.hidden_states[-1]
                if isinstance(last_hidden_states, tuple):
                    text_hidden_states, audio_hidden_states = last_hidden_states
                else:
                    text_hidden_states = last_hidden_states
                    audio_hidden_states = last_hidden_states
            
            # 混合hidden states
            mixed_hidden_states = self.hidden_state_mixer(text_hidden_states, audio_hidden_states)
            
            # 使用adaptor生成motion tokens
            # 这里需要实现类似test_gpt2_adaptor.py中的生成逻辑
            # 但是使用交错的hidden states序列
            generated_tokens = self._generate_from_interleaved_hidden_states(
                mixed_hidden_states,
                attention_mask,
                max_new_tokens,
                temperature,
                top_k,
                repetition_penalty
            )
        
        return generated_tokens
    
    def _generate_from_interleaved_hidden_states(self, 
                                               mixed_hidden_states: torch.Tensor,
                                               attention_mask: Optional[torch.Tensor],
                                               max_new_tokens: int,
                                               temperature: float,
                                               top_k: int,
                                               repetition_penalty: float):
        """
        从交错的hidden states生成motion tokens
        奇数位：mixed_hidden_states，偶数位：生成的motion token embeddings
        """
        batch_size, seq_len, hidden_size = mixed_hidden_states.shape
        
        generated_tokens = []
        current_interleaved_states = torch.zeros(
            batch_size, seq_len * 2, hidden_size,
            device=mixed_hidden_states.device,
            dtype=mixed_hidden_states.dtype
        )
        
        # 填充奇数位（mixed_hidden_states）
        current_interleaved_states[:, 1::2, :] = mixed_hidden_states
        
        # 模拟交错生成过程
        interleave_audios, interleave_motions = 1, 1
        motion_token_count = 0
        
        for i in range(seq_len):
            # 每interleave_audios个audio位置后，生成interleave_motions个motion token
            if (i + 1) % interleave_audios == 0:
                for j in range(interleave_motions):
                    # 使用当前交错序列的最后一个位置生成下一个token
                    current_length = i * 2 + 1 + j  # 当前序列长度
                    
                    if current_length >= current_interleaved_states.shape[1]:
                        break
                    
                    # 获取当前序列
                    current_seq = current_interleaved_states[:, :current_length, :]
                    
                    # 创建attention mask
                    current_attention_mask = torch.ones(
                        batch_size, current_length,
                        device=current_interleaved_states.device
                    )
                    
                    # 通过adaptor生成logits
                    with torch.no_grad():
                        # 这里需要修改adaptor的forward方法以接受hidden states
                        # 暂时使用简化的方法
                        logits = self.motion_adaptor.lm_head(current_seq[:, -1, :])
                    
                    # 应用重复惩罚
                    if repetition_penalty != 1.0 and generated_tokens:
                        for token_id in set(generated_tokens[-100:]):
                            if token_id < logits.size(-1):
                                logits[0, token_id] = logits[0, token_id] / repetition_penalty
                    
                    # 应用temperature
                    logits = logits / temperature
                    
                    # Top-k采样
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = torch.full_like(logits, float('-inf'))
                        logits[0, top_k_indices] = top_k_logits
                    
                    # 采样
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # 确保motion token在正确范围内
                    if motion_token_count % 2 == 0:
                        # body token [0, 511]
                        next_token = int(next_token % 512)
                    else:
                        # hand token [512, 1023]
                        next_token = int(512 + (next_token % 512))
                    
                    motion_token_count += 1
                    generated_tokens.append(next_token)
                    
                    # 将生成的motion token embedding添加到交错序列的偶数位
                    motion_embedding = self.motion_adaptor.transformer.wte(
                        torch.tensor([next_token], device=mixed_hidden_states.device)
                    )
                    
                    if current_length < current_interleaved_states.shape[1]:
                        current_interleaved_states[:, current_length, :] = motion_embedding
                    
                    # 检查是否达到最大motion token数量
                    if len(generated_tokens) >= max_new_tokens:
                        break
                
                if len(generated_tokens) >= max_new_tokens:
                    break
        
        return generated_tokens
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params
    
    def save_model(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'kimi_model_path': getattr(self, 'kimi_model_path', None),
            'freeze_kimi': self.freeze_kimi,
            'freeze_adaptor': self.freeze_adaptor,
            'train_mixer_only': self.train_mixer_only,
            'use_lora': self.use_lora,
        }
        torch.save(save_dict, save_path)
        
        # 如果使用了LoRA，也保存LoRA适配器
        if self.use_lora and hasattr(self.kimi_model, 'save_pretrained'):
            lora_save_dir = save_path.replace('.pt', '_lora') if save_path.endswith('.pt') else f"{save_path}_lora"
            self.kimi_model.save_pretrained(lora_save_dir)
            print(f"✅ LoRA adapter saved to: {lora_save_dir}")
        
        print(f"✅ Model saved to: {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, device: str = "cuda"):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=device)
        
        # 从checkpoint中恢复配置
        kimi_model_path = checkpoint.get('kimi_model_path')
        freeze_kimi = checkpoint.get('freeze_kimi', True)
        freeze_adaptor = checkpoint.get('freeze_adaptor', True)
        train_mixer_only = checkpoint.get('train_mixer_only', True)
        
        # 创建模型实例（需要提供kimi_model_path和gpt2_config）
        # 这里需要根据实际情况调整
        model = cls(
            kimi_model_path=kimi_model_path,
            gpt2_config=GPT2Config(),  # 需要从checkpoint中恢复
            freeze_kimi=freeze_kimi,
            freeze_adaptor=freeze_adaptor,
            train_mixer_only=train_mixer_only
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from: {load_path}")
        return model


def create_unified_model(kimi_model_path: str,
                        gpt2_config_path: Optional[str] = None,
                        freeze_kimi: bool = True,
                        freeze_adaptor: bool = True,
                        train_mixer_only: bool = True,
                        motion_loss_weight: float = 1.0,
                        audio_loss_weight: float = 0.1,
                        lora_r: int = 16,
                        lora_alpha: int = 32,
                        lora_dropout: float = 0.1,
                        debug: bool = False) -> UnifiedKimiMotionModel:
    """
    创建统一的Kimi-Motion模型
    
    Args:
        kimi_model_path: Kimi模型路径
        gpt2_config_path: GPT2配置文件路径（可选）
        freeze_kimi: 是否冻结Kimi模型
        freeze_adaptor: 是否冻结adaptor
        train_mixer_only: 是否只训练mixer
        motion_loss_weight: Motion token loss权重（默认1.0，更高）
        audio_loss_weight: Audio token loss权重（默认0.1，较低）
        lora_r: LoRA rank（默认16）
        lora_alpha: LoRA alpha缩放因子（默认32）
        lora_dropout: LoRA dropout率（默认0.1）
        debug: 是否启用调试模式
    
    Returns:
        UnifiedKimiMotionModel: 统一模型实例
    """
    
    # 创建GPT2配置
    if gpt2_config_path:
        gpt2_config = GPT2Config.from_pretrained(gpt2_config_path)
    else:
        # 使用默认配置
        gpt2_config = GPT2Config(
            vocab_size=1034,  # 512*2 + 10 (motion_vocab_size + pad_token)
            n_positions=4096,  # max_seq_length
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )
    
    # 创建统一模型
    model = UnifiedKimiMotionModel(
        kimi_model_path=kimi_model_path,
        gpt2_config=gpt2_config,
        freeze_kimi=freeze_kimi,
        freeze_adaptor=freeze_adaptor,
        train_mixer_only=train_mixer_only,
        motion_loss_weight=motion_loss_weight,
        audio_loss_weight=audio_loss_weight,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        debug=debug
    )
    
    return model
