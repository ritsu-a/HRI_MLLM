from typing import Union
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from HRI_mllm import OUTPUT_ROOT

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

        saved_weight = torch.load(f"{OUTPUT_ROOT}/motion_adaptor_v1/embed_tokens_weight.pt")
        self.audio_tokenizer = torch.nn.Embedding.from_pretrained(
            saved_weight,
            padding_idx=152063,
            freeze=True,
        )
        for param in self.audio_tokenizer.parameters():
            param.requires_grad = False
        
        # 获取特殊token ID（从config中读取，如果不存在则使用默认值）
        self.audio_gesture_start_token_id = getattr(config, 'audio_gesture_start_token_id', None)
        self.audio_gesture_end_token_id = getattr(config, 'audio_gesture_end_token_id', None)
        self.gesture_start_token_id = getattr(config, 'gesture_start_token_id', None)
        self.gesture_end_token_id = getattr(config, 'gesture_end_token_id', None)
        
        # 获取audio_tokenizer的vocab_size
        self.audio_vocab_size = self.audio_tokenizer.num_embeddings
        
        # 添加9分类多层网络（用于预测label_tokens）
        # 使用下采样和ReLU激活
        hidden_dim = config.hidden_size
        intermediate_dim = hidden_dim // 2  # 下采样到一半维度
        
        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim // 2, 9)  # 最终输出9分类
        )
        
        # 将9分类logit转换为embedding的线性层（用于添加到下一个audio token）
        self.label_logit_to_embedding = nn.Linear(9, hidden_dim)
        # 初始化权重和偏置为0，避免影响初始训练
        nn.init.zeros_(self.label_logit_to_embedding.weight)
        nn.init.zeros_(self.label_logit_to_embedding.bias)
    
    def forward(self, input_data, attention_mask=None, labels=None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                label_tokens=None,  # 新增：label_tokens用于分类任务
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
        if labels is not None and attention_mask is not None:
            audio_mask = (labels == -100) & (attention_mask == 1)
            
            # 检测audio特殊token（audio_gesture_start和audio_gesture_end）
            # 这些token虽然被标记为audio类型，但不在audio_tokenizer的vocab范围内
            special_audio_tokens_mask = torch.zeros_like(input_data, dtype=torch.bool)
            if self.audio_gesture_start_token_id is not None:
                special_audio_tokens_mask |= (input_data == self.audio_gesture_start_token_id)
            if self.audio_gesture_end_token_id is not None:
                special_audio_tokens_mask |= (input_data == self.audio_gesture_end_token_id)
            
            # 确保特殊token只在audio_mask范围内
            special_audio_tokens_mask = special_audio_tokens_mask & audio_mask
            
            # 从audio_mask中分离出特殊token和普通audio token
            normal_audio_mask = audio_mask & ~special_audio_tokens_mask
            
            # 处理普通audio tokens
            if normal_audio_mask.any():
                # 检查token是否在audio_tokenizer的vocab范围内
                valid_audio_mask = normal_audio_mask & (input_data < self.audio_vocab_size) & (input_data >= 0)
                
                if valid_audio_mask.any():
                    valid_audio_tokens = input_data[valid_audio_mask]
                    audio_embeds = self.audio_tokenizer(valid_audio_tokens.long()).detach()
                    projected_hidden = self.input_projection(audio_embeds.float())
                    hidden_states[valid_audio_mask] = projected_hidden
                
                # 处理超出audio_tokenizer范围的audio tokens，使用motion_tokenizer
                invalid_audio_mask = normal_audio_mask & ~valid_audio_mask
                if invalid_audio_mask.any():
                    invalid_audio_tokens = input_data[invalid_audio_mask]
                    invalid_audio_embeds = self.transformer.wte(invalid_audio_tokens.long())
                    hidden_states[invalid_audio_mask] = invalid_audio_embeds
            
            # 处理audio特殊token（audio_gesture_start和audio_gesture_end），使用motion_tokenizer
            if special_audio_tokens_mask.any():
                special_audio_tokens = input_data[special_audio_tokens_mask]
                special_audio_embeds = self.transformer.wte(special_audio_tokens.long())
                hidden_states[special_audio_tokens_mask] = special_audio_embeds
            
            # 处理motion tokens
            motion_mask = ~audio_mask & (attention_mask == 1)
            if motion_mask.any():
                motion_tokens = input_data[motion_mask]
                motion_embeds = self.transformer.wte(motion_tokens.long())
                
                # 将projected hidden states放入对应位置
                hidden_states[motion_mask] = motion_embeds
        else:
            # 如果没有labels和attention_mask，假设所有输入都是motion tokens
            motion_tokens = input_data
            motion_embeds = self.transformer.wte(motion_tokens.long())
            hidden_states = motion_embeds
        
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
        
        # 将motion token位置的9分类logit转换为embedding，并添加到下一个audio token的embedding上
        # 注意：这里是在transformer之后添加label embedding，只影响最终logits的计算
        # 如果需要影响transformer内部的attention，应该在transformer之前添加
        # 优化：只对motion token位置计算label_logits，使用向量化操作
        
        # 调试输出标志（需要在前面定义，因为后面会使用）
        debug_label_loss = hasattr(self, '_debug_label_loss') and self._debug_label_loss
        
        motion_label_logits_for_loss = None  # 初始化为None，确保变量存在
        motion_mask_for_embedding = None  # 保存用于后续验证
        motion_hidden_before_modification = None  # 保存修改前的motion hidden states，用于重新计算label_logits
        motion_mask = None  # 初始化为None，确保变量存在
        label_embedding_additions = None  # 保存label embedding additions用于验证
        if labels is not None and attention_mask is not None:
            audio_mask = (labels == -100) & (attention_mask == 1)
            motion_mask = ~audio_mask & (attention_mask == 1)
            motion_mask_for_embedding = motion_mask  # 保存用于后续验证
            
            if motion_mask.any():
                # 只对motion token位置计算label_logits（避免对全部位置计算）
                motion_hidden = hidden_states[motion_mask]  # [num_motion_tokens, hidden_dim]
                # 保存修改前的motion hidden states，用于后续重新计算（如果需要）
                motion_hidden_before_modification = motion_hidden.clone()
                motion_label_logits = self.label_classifier(motion_hidden)  # [num_motion_tokens, 9]
                
                # 保存用于后续loss计算
                motion_label_logits_for_loss = motion_label_logits
                
                # Vector Quantization: 将label_logits转换为one-hot向量
                # 训练时使用straight-through estimator（argmax + detach + one-hot）
                # 推理时直接使用argmax
                if self.training:
                    # 训练时：使用straight-through estimator
                    # 1. 计算argmax（用于前向传播）
                    label_indices = torch.argmax(motion_label_logits, dim=-1)  # [num_motion_tokens]
                    # 2. 创建one-hot向量
                    motion_label_one_hot = torch.zeros_like(motion_label_logits)
                    motion_label_one_hot.scatter_(1, label_indices.unsqueeze(1), 1.0)
                    # 3. 使用straight-through: 前向用one-hot，反向传播用原始logits
                    motion_label_one_hot = motion_label_one_hot + motion_label_logits - motion_label_logits.detach()
                else:
                    # 推理时：直接使用argmax + one-hot
                    label_indices = torch.argmax(motion_label_logits, dim=-1)  # [num_motion_tokens]
                    motion_label_one_hot = torch.zeros_like(motion_label_logits)
                    motion_label_one_hot.scatter_(1, label_indices.unsqueeze(1), 1.0)
                
                # 将one-hot向量转换为embedding
                motion_label_embeds = self.label_logit_to_embedding(motion_label_one_hot)  # [num_motion_tokens, hidden_dim]
                
                # 使用向量化操作找到每个motion token对应的下一个audio token位置
                batch_size, seq_len = input_data.shape
                device = input_data.device
                
                # 获取motion token的batch和位置索引
                motion_batch_indices, motion_pos_indices = torch.where(motion_mask)
                
                # 为每个motion token找到下一个audio token位置（向量化）
                next_audio_positions = torch.full((len(motion_batch_indices),), -1, dtype=torch.long, device=device)
                
                # 对每个batch分别处理（避免跨batch的复杂逻辑）
                for b in range(batch_size):
                    batch_mask = motion_batch_indices == b
                    if not batch_mask.any():
                        continue
                    
                    batch_motion_pos = motion_pos_indices[batch_mask]
                    batch_audio_mask = audio_mask[b]
                    
                    # 找到所有audio token位置
                    audio_positions = torch.where(batch_audio_mask)[0]
                    
                    if len(audio_positions) > 0:
                        # 对每个motion token，找到下一个audio token位置
                        for i, motion_pos in enumerate(batch_motion_pos):
                            # 找到第一个大于motion_pos的audio位置
                            next_audio_mask = audio_positions > motion_pos
                            if next_audio_mask.any():
                                next_audio_pos = audio_positions[next_audio_mask][0]
                                # 找到在motion_batch_indices中的索引
                                motion_idx = torch.where((motion_batch_indices == b) & (motion_pos_indices == motion_pos))[0]
                                if len(motion_idx) > 0:
                                    next_audio_positions[motion_idx[0]] = next_audio_pos
                
                # 只处理有效的映射（next_audio_positions != -1）
                valid_mask = next_audio_positions >= 0
                if valid_mask.any():
                    valid_batch = motion_batch_indices[valid_mask]
                    valid_audio_pos = next_audio_positions[valid_mask]
                    valid_embeds = motion_label_embeds[valid_mask]
                    
                    # 使用scatter_add_进行高效的累积操作（向量化，避免Python循环）
                    label_embedding_additions = torch.zeros_like(hidden_states)
                    # 使用index_add_在batch维度上累积
                    for b in range(batch_size):
                        batch_mask = valid_batch == b
                        if batch_mask.any():
                            batch_audio_pos = valid_audio_pos[batch_mask]
                            batch_embeds = valid_embeds[batch_mask]
                            # 对同一batch内的多个audio位置，使用index_add_累积
                            label_embedding_additions[b].index_add_(
                                0, batch_audio_pos, batch_embeds
                            )
                    
                    # 使用非inplace操作添加label embedding
                    # 注意：这里是在transformer之后添加，只影响最终logits，不影响transformer内部的attention
                    hidden_states = hidden_states + label_embedding_additions
        
        # 在修改hidden_states后计算logits（这样logits能反映label embedding的影响）
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        label_loss = None
        
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # 计算分类loss（只在motion token位置计算）
        label_logits = None
        label_loss = None
        # 注意：不要重新赋值motion_mask = None，因为前面已经计算过了
        # motion_mask = None  # 已在前面的代码中定义，不要覆盖
        motion_label_logits = None
        
        # 注意：debug_label_loss已在前面定义，这里不需要重新定义
        
        if label_tokens is not None and labels is not None and attention_mask is not None:
            # 重新计算motion_mask，确保与前面一致
            # 注意：必须使用与前面相同的计算方式
            audio_mask_loss = (labels == -100) & (attention_mask == 1)
            motion_mask_loss = ~audio_mask_loss & (attention_mask == 1)
            
            if motion_mask_loss.any():
                # 如果已经计算过motion_label_logits，直接使用；否则现在计算
                if motion_label_logits_for_loss is not None:
                    # 验证motion_mask是否匹配
                    expected_num_motion = motion_mask_loss.sum().item()
                    actual_num_logits = motion_label_logits_for_loss.shape[0]
                    if expected_num_motion == actual_num_logits:
                        motion_label_logits = motion_label_logits_for_loss
                    else:
                        # 如果不匹配，重新计算
                        motion_hidden = hidden_states[motion_mask_loss]
                        motion_label_logits = self.label_classifier(motion_hidden)
                else:
                    motion_hidden = hidden_states[motion_mask_loss]
                    motion_label_logits = self.label_classifier(motion_hidden)
                
                motion_label_tokens = label_tokens[motion_mask_loss]
                
                # 检查是否有有效的label（>=0，因为-1表示无效/无label，0-8是有效类别）
                valid_label_mask = motion_label_tokens >= 0
                num_valid_labels = valid_label_mask.sum().item()
                
                if num_valid_labels > 0:
                    # 计算分类loss（CrossEntropyLoss）
                    # 忽略label=-1的位置（padding、audio token、特殊token或BEAT数据，没有label）
                    label_loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
                    # 确保输入形状正确
                    logits_flat = motion_label_logits.view(-1, 9)  # [num_motion_tokens, 9]
                    labels_flat = motion_label_tokens.view(-1)  # [num_motion_tokens]
                    
                    label_loss = label_loss_fct(logits_flat, labels_flat)
                    
                    # 计算准确率
                    with torch.no_grad():
                        predicted_labels = torch.argmax(logits_flat, dim=-1)
                        valid_predicted = predicted_labels[valid_label_mask]
                        valid_gt = labels_flat[valid_label_mask]
                        label_accuracy = (valid_predicted == valid_gt).float().mean().item()
                        # 存储准确率用于输出
                        self._last_label_accuracy = label_accuracy
                    
                    # 确保label_loss是tensor且有效
                    if label_loss is not None:
                        if not isinstance(label_loss, torch.Tensor):
                            label_loss = torch.tensor(label_loss, device=motion_label_logits.device, dtype=motion_label_logits.dtype)
                        # 检查loss是否为nan或inf
                        if torch.isnan(label_loss) or torch.isinf(label_loss):
                            label_loss = None
                else:
                    # 如果没有有效的label（所有label都是-1，如只有BEAT数据），loss为None
                    label_loss = None
                    self._last_label_accuracy = None
        
        # 创建完整的label_logits用于输出（只motion位置有值）
        # 注意：即使label_tokens=None（推理时），也应该计算并返回label_logits
        if label_logits is None:
            # 确定使用哪个motion_mask和motion_label_logits
            # 优先使用motion_mask_loss（训练时），否则使用motion_mask（推理时）
            if 'motion_mask_loss' in locals() and motion_mask_loss is not None:
                final_motion_mask = motion_mask_loss
                final_motion_label_logits = motion_label_logits if 'motion_label_logits' in locals() and motion_label_logits is not None else None
            elif 'motion_mask' in locals() and motion_mask is not None:
                final_motion_mask = motion_mask
                # 如果motion_label_logits_for_loss存在（在推理时已计算），使用它
                final_motion_label_logits = motion_label_logits_for_loss if motion_label_logits_for_loss is not None else None
            else:
                final_motion_mask = None
                final_motion_label_logits = None
            
            # 填充label_logits
            if final_motion_mask is not None and final_motion_mask.any() and final_motion_label_logits is not None:
                # 验证mask和logits的匹配
                expected_num = final_motion_mask.sum().item()
                actual_num = final_motion_label_logits.shape[0]
                if expected_num != actual_num:
                    # 如果数量不匹配，使用保存的修改前的motion hidden states重新计算
                    # 注意：不能使用修改后的hidden_states，因为已经添加了label embedding
                    if motion_hidden_before_modification is not None:
                        # 如果mask匹配，直接使用保存的hidden states
                        if motion_hidden_before_modification.shape[0] == expected_num:
                            final_motion_label_logits = self.label_classifier(motion_hidden_before_modification)
                        else:
                            # 如果mask不匹配，需要重新从原始hidden_states提取（但此时hidden_states已被修改）
                            # 这种情况下，我们应该使用已计算的motion_label_logits_for_loss
                            # 如果motion_label_logits_for_loss存在且数量匹配，使用它
                            if motion_label_logits_for_loss is not None and motion_label_logits_for_loss.shape[0] == expected_num:
                                final_motion_label_logits = motion_label_logits_for_loss
                            else:
                                # 如果都不匹配，将final_motion_label_logits设置为None，让代码进入else分支创建全零label_logits
                                final_motion_label_logits = None
                    else:
                        # 如果没有保存的motion_hidden，尝试使用motion_label_logits_for_loss
                        if motion_label_logits_for_loss is not None and motion_label_logits_for_loss.shape[0] == expected_num:
                            final_motion_label_logits = motion_label_logits_for_loss
                        else:
                            final_motion_label_logits = None
                
                # 再次检查final_motion_label_logits是否为None（可能在上述逻辑中被设置为None）
                if final_motion_label_logits is not None:
                    label_logits = torch.zeros(
                        hidden_states.shape[0], hidden_states.shape[1], 9,
                        device=hidden_states.device, dtype=final_motion_label_logits.dtype
                    )
                    label_logits[final_motion_mask] = final_motion_label_logits
                else:
                    # 如果final_motion_label_logits为None，创建全零的label_logits
                    label_logits = torch.zeros(
                        hidden_states.shape[0], hidden_states.shape[1], 9,
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
            elif hidden_states is not None:
                # 如果没有motion_label_logits，创建一个全零的label_logits
                label_logits = torch.zeros(
                    hidden_states.shape[0], hidden_states.shape[1], 9,
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            else:
                label_logits = None

        # 将label_logits和label_loss存储为额外属性
        output = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        # 添加额外的属性（确保label_loss是tensor或None）
        output.label_logits = label_logits
        
        if label_loss is not None:
            # 确保label_loss是tensor
            if not isinstance(label_loss, torch.Tensor):
                device = logits.device if logits is not None else hidden_states.device
                label_loss = torch.tensor(label_loss, device=device)
            output.label_loss = label_loss
            # 添加准确率
            if hasattr(self, '_last_label_accuracy'):
                output.label_accuracy = self._last_label_accuracy
        else:
            output.label_loss = None
            output.label_accuracy = None
        
        # 多卡训练时，DDP可能不会传递自定义属性，保存到模型属性中作为备份
        # 这样在训练循环中可以从model.module._last_label_loss获取
        if label_loss is not None:
            self._last_label_loss = label_loss
            self._last_label_logits = label_logits
        else:
            self._last_label_loss = None
            self._last_label_logits = label_logits
        
        if not return_dict:
            return (output.loss, output.logits, output.label_logits, output.label_loss) + transformer_outputs[1:]
        
        return output