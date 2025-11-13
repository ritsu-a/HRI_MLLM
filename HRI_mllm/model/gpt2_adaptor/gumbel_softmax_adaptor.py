"""
GPT2 Adaptor with Gumbel-Softmax for differentiable audio token handling
支持端到端训练Kimi + GPT2 Adaptor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GumbelSoftmaxGPT2Adaptor(GPT2LMHeadModel):
    """
    支持Gumbel-Softmax重参数化的GPT2 Adaptor
    
    训练时：接受Kimi的audio logits，使用Gumbel-Softmax生成soft embeddings
    推理时：接受离散的audio token ids，使用常规embedding
    
    这样可以让motion loss的梯度回传到Kimi model
    """
    
    def __init__(self, config, audio_vocab_size=152064, audio_hidden_size=3584):
        super().__init__(config)
        
        # Audio tokenizer embedding (from Kimi)
        self.audio_tokenizer = nn.Embedding(audio_vocab_size, audio_hidden_size)
        
        # Projection layer to align audio hidden size with GPT2 hidden size
        if audio_hidden_size != config.hidden_size:
            self.input_projection = nn.Linear(audio_hidden_size, config.hidden_size)
        else:
            self.input_projection = nn.Identity()
        
        # Motion tokenizer (trainable)
        self.motion_tokenizer = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Special tokens
        self.audio_gesture_start_token_id = getattr(config, 'audio_gesture_start_token_id', 512*2 + 3)
        self.audio_gesture_end_token_id = getattr(config, 'audio_gesture_end_token_id', 512*2 + 5)
        self.gesture_start_token_id = getattr(config, 'gesture_start_token_id', 512*2 + 2)
        self.gesture_end_token_id = getattr(config, 'gesture_end_token_id', 512*2 + 4)
        
        self.audio_vocab_size = audio_vocab_size
        
    def load_pretrained_audio_tokenizer(self, weight_path, freeze=True):
        """加载预训练的audio tokenizer权重"""
        saved_weight = torch.load(weight_path, weights_only=True)
        self.audio_tokenizer = nn.Embedding.from_pretrained(
            saved_weight,
            freeze=freeze,
        )
        if freeze:
            for param in self.audio_tokenizer.parameters():
                param.requires_grad = False
        print(f"✅ Loaded audio tokenizer from {weight_path} (freeze={freeze})")
    
    def gumbel_softmax_sample(self, logits, tau=1.0, hard=False, dim=-1):
        """
        Gumbel-Softmax采样
        
        Args:
            logits: [batch_size, seq_len, vocab_size] 原始logits
            tau: 温度参数，越小越接近one-hot（推荐训练初期用1.0，后期降到0.5）
            hard: 是否使用straight-through estimator（前向one-hot，反向soft）
            dim: softmax的维度
            
        Returns:
            soft_tokens: [batch_size, seq_len, vocab_size] soft token分布
        """
        # 添加Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        gumbel_logits = (logits + gumbel_noise) / tau
        
        # Softmax得到soft distribution
        soft_tokens = F.softmax(gumbel_logits, dim=dim)
        
        if hard:
            # Straight-through estimator
            # 前向传播：使用one-hot（hard）
            # 反向传播：使用soft gradient
            hard_tokens = torch.zeros_like(soft_tokens)
            hard_tokens.scatter_(dim, soft_tokens.argmax(dim=dim, keepdim=True), 1.0)
            
            # Straight-through: 前向hard，反向soft
            soft_tokens = hard_tokens - soft_tokens.detach() + soft_tokens
        
        return soft_tokens
    
    def forward(
        self, 
        input_data=None,
        audio_logits=None,  # 🔥 新增：来自Kimi的audio logits [batch_size, seq_len, audio_vocab_size]
        attention_mask=None,
        labels=None,
        gumbel_tau=1.0,  # Gumbel-Softmax温度
        use_gumbel_hard=False,  # 是否使用straight-through
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ):
        """
        前向传播
        
        两种使用模式：
        1. 训练模式（端到端）：传入audio_logits，使用Gumbel-Softmax
        2. 推理模式：传入input_data（离散token ids），使用常规embedding
        
        Args:
            input_data: 离散token ids [batch_size, seq_len]（推理时使用）
            audio_logits: 来自Kimi的audio logits [batch_size, seq_len, audio_vocab_size]（训练时使用）
            attention_mask: attention mask [batch_size, seq_len]
            labels: ground truth labels [batch_size, seq_len]
            gumbel_tau: Gumbel-Softmax温度参数
            use_gumbel_hard: 是否使用straight-through estimator
        """
        
        batch_size = (audio_logits.shape[0] if audio_logits is not None else input_data.shape[0])
        seq_len = (audio_logits.shape[1] if audio_logits is not None else input_data.shape[1])
        device = (audio_logits.device if audio_logits is not None else input_data.device)
        
        # 初始化hidden states
        hidden_states = torch.zeros(
            batch_size, seq_len, self.config.hidden_size,
            dtype=torch.float32, device=device
        )
        
        # 🔥 模式1：训练模式 - 使用Gumbel-Softmax处理audio logits
        if audio_logits is not None and self.training:
            print(f"🔥 Using Gumbel-Softmax mode (tau={gumbel_tau}, hard={use_gumbel_hard})")
            
            # 检测audio token的位置（通过attention_mask或labels）
            if labels is not None:
                audio_mask = (labels == -100)  # audio tokens的label为-100
            else:
                audio_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            
            # 应用Gumbel-Softmax
            soft_audio_tokens = self.gumbel_softmax_sample(
                audio_logits, 
                tau=gumbel_tau, 
                hard=use_gumbel_hard
            )  # [batch_size, seq_len, audio_vocab_size]
            
            # 通过soft distribution获取soft embeddings
            # soft_embeds = soft_tokens @ embedding_matrix
            audio_embeddings = self.audio_tokenizer.weight  # [audio_vocab_size, audio_hidden_size]
            soft_audio_embeds = torch.matmul(
                soft_audio_tokens, 
                audio_embeddings
            )  # [batch_size, seq_len, audio_hidden_size]
            
            # Project to GPT2 hidden size
            projected_audio_embeds = self.input_projection(soft_audio_embeds)
            
            # 将audio embeddings放入对应位置
            hidden_states[audio_mask] = projected_audio_embeds[audio_mask]
            
            # 处理motion tokens（如果有）
            if labels is not None:
                motion_mask = (labels != -100)
                if motion_mask.any() and input_data is not None:
                    motion_tokens = input_data[motion_mask]
                    motion_embeds = self.transformer.wte(motion_tokens.long())
                    hidden_states[motion_mask] = motion_embeds
        
        # 🔥 模式2：推理模式 - 使用离散tokens
        else:
            if input_data is None:
                raise ValueError("input_data is required for inference mode")
            
            if attention_mask is None or labels is None:
                # 如果没有attention_mask/labels，假设所有都是audio tokens
                audio_mask = torch.ones_like(input_data, dtype=torch.bool)
            else:
                audio_mask = (labels == -100)
            
            # 识别特殊audio token
            special_audio_tokens_mask = torch.zeros_like(input_data, dtype=torch.bool)
            if self.audio_gesture_start_token_id is not None:
                special_audio_tokens_mask |= (input_data == self.audio_gesture_start_token_id)
            if self.audio_gesture_end_token_id is not None:
                special_audio_tokens_mask |= (input_data == self.audio_gesture_end_token_id)
            
            # 普通audio tokens
            normal_audio_mask = audio_mask & ~special_audio_tokens_mask
            
            if normal_audio_mask.any():
                valid_audio_mask = normal_audio_mask & (input_data < self.audio_vocab_size) & (input_data >= 0)
                
                if valid_audio_mask.any():
                    valid_audio_tokens = input_data[valid_audio_mask]
                    audio_embeds = self.audio_tokenizer(valid_audio_tokens.long())
                    projected_hidden = self.input_projection(audio_embeds.float())
                    hidden_states[valid_audio_mask] = projected_hidden
                
                invalid_audio_mask = normal_audio_mask & ~valid_audio_mask
                if invalid_audio_mask.any():
                    invalid_audio_tokens = input_data[invalid_audio_mask]
                    invalid_audio_embeds = self.transformer.wte(invalid_audio_tokens.long())
                    hidden_states[invalid_audio_mask] = invalid_audio_embeds
            
            # 处理special audio tokens
            if special_audio_tokens_mask.any():
                special_audio_tokens = input_data[special_audio_tokens_mask]
                special_audio_embeds = self.transformer.wte(special_audio_tokens.long())
                hidden_states[special_audio_tokens_mask] = special_audio_embeds
            
            # 处理motion tokens
            motion_mask = ~audio_mask & (attention_mask == 1)
            if motion_mask.any():
                motion_tokens = input_data[motion_mask]
                motion_embeds = self.transformer.wte(motion_tokens.long())
                hidden_states[motion_mask] = motion_embeds
        
        # 通过GPT2 transformer
        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = transformer_outputs[0]
        
        # 计算logits
        lm_logits = self.lm_head(hidden_states)
        
        # 计算loss
        loss = None
        if labels is not None:
            # 只计算motion token的loss（labels != -100的位置）
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


class EndToEndKimiMotionModel(nn.Module):
    """
    端到端的Kimi + GPT2 Adaptor模型
    支持联合训练
    """
    
    def __init__(
        self,
        kimi_model,  # Kimi audio model
        gpt2_adaptor,  # GumbelSoftmaxGPT2Adaptor
        freeze_kimi=False,
        freeze_adaptor=False,
        gumbel_tau_schedule=None,  # 温度调度函数
    ):
        super().__init__()
        
        self.kimi_model = kimi_model
        self.gpt2_adaptor = gpt2_adaptor
        
        # 冻结策略
        if freeze_kimi:
            for param in self.kimi_model.parameters():
                param.requires_grad = False
            print("❄️  Kimi model frozen")
        
        if freeze_adaptor:
            for param in self.gpt2_adaptor.parameters():
                param.requires_grad = False
            print("❄️  GPT2 Adaptor frozen")
        
        self.gumbel_tau_schedule = gumbel_tau_schedule
        self.current_step = 0
    
    def get_current_tau(self):
        """获取当前的Gumbel温度"""
        if self.gumbel_tau_schedule is not None:
            return self.gumbel_tau_schedule(self.current_step)
        else:
            # 默认：从1.0指数衰减到0.5
            return max(0.5, 1.0 * (0.95 ** (self.current_step / 1000)))
    
    def forward(
        self,
        user_audio_features,  # Kimi的输入音频特征
        user_audio_tokens,  # 用户音频的token ids（用于构建interleaved sequence）
        motion_tokens_gt,  # Ground truth motion tokens
        use_gumbel_hard=False,
        **kwargs
    ):
        """
        端到端前向传播
        
        Args:
            user_audio_features: Kimi编码后的音频特征
            user_audio_tokens: 用户音频token ids
            motion_tokens_gt: 标注的motion tokens
        """
        
        # 1️⃣ Kimi model生成assistant audio logits
        # 这里需要修改Kimi的generate方法，让它返回logits而不是采样的tokens
        kimi_outputs = self.kimi_model(
            audio_input_ids=user_audio_tokens,
            continous_feature=user_audio_features,
            output_logits=True,  # 🔥 关键：返回logits
            **kwargs
        )
        
        assistant_audio_logits = kimi_outputs.logits  # [batch, seq_len, audio_vocab_size]
        
        # 2️⃣ 构建interleaved sequence用于GPT2 adaptor
        # 这里需要将audio logits和motion tokens交错排列
        # 简化版本：假设已经构建好了
        
        # 构建labels：audio位置为-100，motion位置为真实token
        batch_size, seq_len = assistant_audio_logits.shape[:2]
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=assistant_audio_logits.device)
        # 这里需要根据实际的interleave策略填充motion token的labels
        
        # 3️⃣ GPT2 adaptor处理（使用Gumbel-Softmax）
        current_tau = self.get_current_tau()
        
        adaptor_outputs = self.gpt2_adaptor(
            audio_logits=assistant_audio_logits,  # 🔥 传入logits
            attention_mask=None,
            labels=labels,
            gumbel_tau=current_tau,
            use_gumbel_hard=use_gumbel_hard,
        )
        
        motion_loss = adaptor_outputs.loss
        
        # 4️⃣ 可选：添加audio token的辅助loss
        # 如果有ground truth的assistant audio tokens
        # audio_loss = F.cross_entropy(...)
        
        self.current_step += 1
        
        return {
            'motion_loss': motion_loss,
            'logits': adaptor_outputs.logits,
            'gumbel_tau': current_tau,
        }


