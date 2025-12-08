# 训练设计总结

## 一、整体架构

### 1.1 双模型设计
训练采用**双模型架构**，将 motion prediction 和 label prediction 分离：

- **motion_model**：
  - 不包含 `label_classifier` 和 `label_logit_to_embedding`
  - 用于计算 motion token prediction loss
  - 作为主模型，被 DDP 包装

- **label_model**：
  - 包含 `label_classifier`（9分类网络）和 `label_logit_to_embedding`
  - 用于计算 label classification loss
  - 不进行 DDP 包装

### 1.2 参数共享机制
两个模型**共享核心 transformer 参数**：
- `transformer`：GPT-2 transformer 层
- `input_projection`：输入投影层
- `motion_tokenizer`：motion token embedding
- `audio_tokenizer`：audio token embedding
- `lm_head`：语言模型头

**独立参数**（仅在 label_model 中）：
- `label_classifier`：9分类网络（3层，带 ReLU 和 Dropout）
- `label_logit_to_embedding`：将 label logits 转换为 embedding 的线性层

## 二、模型结构

### 2.1 MixedInputGPT2 模型
基于 GPT-2 的适配器模型，支持混合输入（audio tokens + motion tokens）

**关键组件**：
- Audio tokenizer：预训练的 audio embedding（冻结）
- Motion tokenizer：可训练的 motion embedding
- Transformer：12层 GPT-2 transformer（n_positions=512）
- LM Head：输出 motion token logits

**Label 相关组件**（仅在 label_model 中）：
- `label_classifier`：3层 MLP（768 → 384 → 192 → 9）
- `label_logit_to_embedding`：线性层（9 → 768）

### 2.2 Label Classifier 结构
```python
nn.Sequential(
    nn.Linear(768, 384),      # 下采样到一半
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(384, 192),      # 再次下采样
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(192, 9)         # 9分类输出
)
```

## 三、训练流程

### 3.1 数据准备
- **BEAT 数据集**：不包含 label tokens（label_tokens 全为 -1）
- **single_motion 数据集**：包含 label tokens（0-8 为有效类别，-1 为无效）
- **序列长度**：统一为 512（BEAT 过长时随机裁剪）
- **Batch 大小**：64
- **数据采样**：加权采样（BEAT:single_motion = 1.0:1.0）

### 3.2 Forward Pass（每个 batch）

#### 第一次 Forward：Motion Prediction
```python
outputs = motion_model(
    inputs=inputs,
    labels=labels,
    attention_mask=attn_mask,
    label_tokens=None,
    use_label_prediction_mode=False
)
motion_loss = outputs.loss / accum_steps
```

**特点**：
- 使用真实的 motion tokens
- 不计算 label loss
- 计算 motion token prediction loss

#### 第二次 Forward：Label Prediction
```python
if label_tokens_batch is not None:
    outputs_label = label_model(
        inputs=inputs,
        labels=labels,
        attention_mask=attn_mask,
        label_tokens=label_tokens_batch,
        use_label_prediction_mode=True
    )
    label_loss = outputs_label.label_loss
```

**特点**：
- 使用 `motion_blank_token_id`（513）替换 motion tokens
- 只基于 audio tokens 预测 label
- 计算 9 分类 CrossEntropyLoss（ignore_index=-1）

### 3.3 Loss 计算与 Backward

```python
# 合并 loss
label_loss_weight = 0.5
combined_loss = motion_loss + label_loss_weight * label_loss

# 一次性 backward
combined_loss.backward()
```

**关键点**：
- 两个 loss 合并后一次性 backward
- `label_loss` 可以监督 transformer 参数（因为 label_model 共享 transformer）
- 使用梯度累积（accum_steps）

### 3.4 Label Embedding 机制
在 `use_label_prediction_mode=True` 时：
1. 计算 motion token 位置的 label logits
2. 使用 Straight-Through Estimator（STE）转换为 one-hot
3. 通过 `label_logit_to_embedding` 转换为 embedding
4. 添加到**下一个** audio token 的 hidden state

**训练时**：使用 GT labels 创建 one-hot（如果可用）
**推理时**：使用预测的 labels（argmax）

## 四、分布式训练（DDP）

### 4.1 DDP 配置
```python
model = DistributedDataParallel(
    motion_model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False
)
model._set_static_graph()  # 启用静态图模式
```

**关键设计**：
- 只包装 `motion_model`（主模型）
- `label_model` 不进行 DDP 包装
- 使用 `_set_static_graph()` 因为 motion_model 的计算图结构一致

### 4.2 为什么可以避免 DDP 静态图错误？
- `motion_model` 不包含 `label_classifier`，计算图结构在每次 forward 中一致
- `label_model` 不进行 DDP 包装，不会触发 DDP 的静态图检查
- 两个模型分离，避免了参数使用模式变化的问题

## 五、优化器配置

```python
optimizer = AdamW(
    list(motion_model.parameters()) + 
    list(label_model.label_classifier.parameters()) + 
    list(label_model.label_logit_to_embedding.parameters()),
    lr=1e-4
)
```

**参数包含**：
- `motion_model` 的所有参数（包括共享的 transformer）
- `label_model` 的 `label_classifier` 参数
- `label_model` 的 `label_logit_to_embedding` 参数

**注意**：由于参数共享，transformer 参数实际上只优化一次，但两个模型的梯度都会更新它。

## 六、Checkpoint 管理

### 6.1 保存
```python
checkpoint = {
    'epoch': epoch,
    'model_state': {
        # motion_model 的所有参数（包括共享的 transformer）
        **motion_model.state_dict(),
        # label_model 的独立参数
        **label_classifier_state_dict,
        **label_logit_to_embedding_state_dict
    },
    'optimizer': optimizer.state_dict()
}
```

### 6.2 加载
1. 加载 `motion_model` 的 state_dict（包含共享的 transformer 参数）
2. 单独加载 `label_model` 的 `label_classifier` 和 `label_logit_to_embedding`
3. 由于参数共享，transformer 参数会自动同步到 `label_model`

## 七、训练超参数

- **学习率**：1e-4
- **Batch Size**：64
- **Max Sequence Length**：512
- **Gradient Accumulation Steps**：1
- **Epochs**：4000
- **Label Loss Weight**：0.5
- **Optimizer**：AdamW
- **Gradient Clipping**：1.0

## 八、关键设计优势

### 8.1 解决 DDP 静态图问题
- **问题**：之前单模型设计导致 `label_classifier` 参数使用模式不一致
- **解决**：分离为两个模型，`motion_model` 的计算图结构一致

### 8.2 参数共享保证一致性
- 两个模型共享 transformer 参数，确保训练一致性
- `label_loss` 可以监督 transformer 参数

### 8.3 内存优化
- 两次 forward 之间清理 CUDA 缓存
- 删除不需要的 logits 张量

### 8.4 灵活的 Loss 权重
- `motion_loss` 和 `label_loss` 可以独立调整权重
- 当前设置：`label_loss_weight = 0.5`

## 九、数据流

```
输入序列: [audio_token, motion_token, audio_token, motion_token, ...]
                ↓
第一次 Forward (motion_model):
  - 使用真实 motion tokens
  - 计算 motion_loss
                ↓
第二次 Forward (label_model):
  - 将 motion tokens 替换为 motion_blank (513)
  - 基于 audio tokens 预测 labels
  - 计算 label_loss
  - 将 label embedding 添加到下一个 audio token
                ↓
合并 Loss: motion_loss + 0.5 * label_loss
                ↓
Backward & Update
```

## 十、训练监控

- **Motion Loss**：motion token prediction loss
- **Label Loss**：9 分类 CrossEntropyLoss
- **Label Accuracy**：9 分类准确率（仅计算有效 labels）
- **Combined Loss**：总损失
- **Learning Rate**：当前学习率
- **Memory Usage**：GPU 内存使用情况

## 十一、注意事项

1. **Label Tokens 处理**：
   - BEAT 数据：label_tokens 全为 -1（无效）
   - single_motion 数据：label_tokens 为 0-8（有效类别）或 -1（无效）

2. **Motion Blank Token**：
   - 默认 ID：513
   - 用于 label prediction 时替换 motion tokens

3. **梯度同步**：
   - 由于参数共享，transformer 参数的梯度会从两个 forward 中累积
   - DDP 会自动同步所有参数的梯度

4. **Checkpoint 兼容性**：
   - 旧 checkpoint 可能不包含 `label_classifier` 参数
   - 使用 `strict=False` 加载，缺失参数使用默认初始化


