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
        self.motion_blank_token_id = getattr(config, 'motion_blank_token_id', 513)  # 默认值为513（根据special_tokens.py注释）
        
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
                use_label_prediction_mode=False,  # 新增：是否使用label预测模式（使用motion_blank）
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
        # 从kwargs中提取output_attentions，如果没有则默认为False
        output_attentions = kwargs.pop('output_attentions', False)
        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
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
            
            # 如果使用label预测模式，将motion token替换为motion_blank
            if use_label_prediction_mode and motion_mask.any():
                input_data = input_data.clone()
                input_data[motion_mask] = self.motion_blank_token_id
            
            if motion_mask.any():
                # 如果使用label预测模式，计算label_logits（基于motion_blank序列）
                # 否则，计算label_logits用于添加到下一个audio token（基于真实motion序列）
                if use_label_prediction_mode:
                    # label预测模式：使用motion_blank位置的hidden_states计算label_logits
                    # 注意：即使使用GT label计算loss，我们仍然需要通过label_classifier预测logits
                    # 因为CrossEntropyLoss需要logits和GT labels
                    motion_hidden_for_label = hidden_states[motion_mask]  # [num_motion_tokens, hidden_dim]
                    motion_label_logits = self.label_classifier(motion_hidden_for_label)  # [num_motion_tokens, 9]
                    motion_label_logits_for_loss = motion_label_logits
                    motion_hidden_before_modification = motion_hidden_for_label.clone()
                else:
                    # 正常模式：计算label_logits用于添加到下一个audio token（推理时使用）
                    # 注意：如果模型没有label_classifier（如motion_model），直接跳过
                    if hasattr(self, 'label_classifier'):
                        if motion_mask.any():
                            motion_hidden_for_label = hidden_states[motion_mask]  # [num_motion_tokens, hidden_dim]
                            # 计算label_logits（用于添加到下一个audio token）
                            motion_label_logits = self.label_classifier(motion_hidden_for_label)  # [num_motion_tokens, 9]
                            motion_hidden_before_modification = motion_hidden_for_label.clone()
                        else:
                            # 如果没有motion tokens，使用一个dummy输入（取hidden_states的第一个token）
                            # 这样确保label_classifier的参数在每次forward中都被标记
                            motion_hidden_for_label = hidden_states[:1]  # [1, hidden_dim]
                            # 为了DDP静态图兼容性，也"使用"label_classifier，但用detach避免梯度
                            motion_label_logits_dummy = self.label_classifier(motion_hidden_for_label)
                            # 立即detach结果并删除，不保留梯度
                            motion_label_logits_dummy = motion_label_logits_dummy.detach()
                            del motion_label_logits_dummy
                            motion_label_logits = None
                            motion_hidden_before_modification = None
                    else:
                        motion_label_logits = None
                        motion_hidden_before_modification = None
                    
                    motion_label_logits_for_loss = None  # 正常模式不计算loss
                
                # 在label预测模式或正常模式下进行Vector Quantization和embedding添加
                # 正常模式下也添加label embedding，这样推理时可以使用预测的label影响生成
                if motion_label_logits is not None:
                    # Vector Quantization: 将label转换为one-hot向量
                    # 训练时：如果使用label预测模式，优先使用GT label（如果存在），否则使用推理的label
                    # 推理时或正常模式：使用推理的label
                    if use_label_prediction_mode and self.training and label_tokens is not None:
                        # 训练时：优先使用GT label
                        # 获取motion token位置对应的GT label
                        motion_label_tokens_gt = label_tokens[motion_mask]  # [num_motion_tokens]
                        # 只对有效的GT label（>=0）使用GT，无效的（-1）使用推理结果
                        valid_gt_mask = motion_label_tokens_gt >= 0
                        
                        # 初始化one-hot向量
                        motion_label_one_hot = torch.zeros_like(motion_label_logits)
                        
                        if valid_gt_mask.any():
                            # 对有效GT label的位置，使用GT label创建one-hot
                            valid_gt_labels = motion_label_tokens_gt[valid_gt_mask]
                            motion_label_one_hot[valid_gt_mask].scatter_(1, valid_gt_labels.unsqueeze(1), 1.0)
                        
                        # 对无效GT label的位置（-1），使用推理的label（straight-through estimator）
                        invalid_gt_mask = ~valid_gt_mask
                        if invalid_gt_mask.any():
                            label_indices_pred = torch.argmax(motion_label_logits[invalid_gt_mask], dim=-1)
                            motion_label_one_hot[invalid_gt_mask].scatter_(1, label_indices_pred.unsqueeze(1), 1.0)
                            # 使用straight-through: 前向用one-hot，反向传播用原始logits
                            motion_label_one_hot[invalid_gt_mask] = (
                                motion_label_one_hot[invalid_gt_mask] + 
                                motion_label_logits[invalid_gt_mask] - 
                                motion_label_logits[invalid_gt_mask].detach()
                            )
                    else:
                        # 推理时或没有GT label时：使用推理的label
                        label_indices = torch.argmax(motion_label_logits, dim=-1)  # [num_motion_tokens]
                        motion_label_one_hot = torch.zeros_like(motion_label_logits)
                        motion_label_one_hot.scatter_(1, label_indices.unsqueeze(1), 1.0)
                    
                    # 将one-hot向量转换为embedding（用于添加到下一个audio token）
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
                else:
                    # 非label预测模式：不添加label embedding
                    label_embedding_additions = None
        
        # 在修改hidden_states后计算logits（这样logits能反映label embedding的影响）
        # 如果使用label预测模式，不计算motion logits（只计算label_logits）以节省显存
        if use_label_prediction_mode:
            logits = None  # label预测模式不计算motion logits，节省显存
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        label_loss = None
        
        if labels is not None and not use_label_prediction_mode:
            # 只在非label预测模式下计算motion loss
            if logits is not None:
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
        
        # 只在label预测模式下计算label_loss
        # 注意：在训练时，我们需要通过label_classifier预测label_logits来计算loss
        # 但在label预测模式下，我们仍然需要label_classifier来预测，因为CrossEntropyLoss需要logits
        if use_label_prediction_mode and label_tokens is not None and labels is not None and attention_mask is not None:
            # 重新计算motion_mask，确保与前面一致
            # 注意：必须使用与前面相同的计算方式
            audio_mask_loss = (labels == -100) & (attention_mask == 1)
            motion_mask_loss = ~audio_mask_loss & (attention_mask == 1)
            
            if motion_mask_loss.any() and motion_label_logits_for_loss is not None:
                # 使用已经计算好的motion_label_logits（基于motion_blank序列）
                # 注意：即使使用GT label，我们仍然需要通过label_classifier预测logits来计算loss
                motion_label_logits = motion_label_logits_for_loss
                
                motion_label_tokens = label_tokens[motion_mask_loss]
                
                # 检查是否有有效的label（>=0，因为-1表示无效/无label，0-8是有效类别）
                valid_label_mask = motion_label_tokens >= 0
                num_valid_labels = valid_label_mask.sum().item()
                
                if num_valid_labels > 0:
                    # 计算分类loss（CrossEntropyLoss）
                    # 注意：即使使用GT label，我们仍然需要logits来计算CrossEntropyLoss
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
        # 注意：只在label预测模式下创建label_logits，避免在正常模式下使用label_classifier
        if label_logits is None:
            if use_label_prediction_mode and motion_label_logits_for_loss is not None:
                # label预测模式：使用已计算的label_logits
                if motion_mask is not None and motion_mask.any():
                    label_logits = torch.zeros(
                        hidden_states.shape[0], hidden_states.shape[1], 9,
                        device=hidden_states.device, dtype=motion_label_logits_for_loss.dtype
                    )
                    label_logits[motion_mask] = motion_label_logits_for_loss
                else:
                    label_logits = torch.zeros(
                        hidden_states.shape[0], hidden_states.shape[1], 9,
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
            else:
                # 非label预测模式：创建全零的label_logits（不计算，避免使用label_classifier）
                if hidden_states is not None:
                    label_logits = torch.zeros(
                        hidden_states.shape[0], hidden_states.shape[1], 9,
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                else:
                    label_logits = None

        # 将label_logits和label_loss存储为额外属性
        # 确保attentions被正确传递
        attentions = None
        if hasattr(transformer_outputs, 'attentions'):
            attentions = transformer_outputs.attentions
        
        output = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values if hasattr(transformer_outputs, 'past_key_values') else None,
            hidden_states=transformer_outputs.hidden_states if hasattr(transformer_outputs, 'hidden_states') else None,
            attentions=attentions,
            cross_attentions=transformer_outputs.cross_attentions if hasattr(transformer_outputs, 'cross_attentions') else None,
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