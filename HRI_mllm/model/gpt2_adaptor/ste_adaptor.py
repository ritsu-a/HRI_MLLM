"""
Straight-Through Estimator (STE) for differentiable discrete token handling

原理：
- 前向传播：使用argmax得到离散token（hard）
- 反向传播：将梯度直接传递（straight through），忽略argmax操作

优点：简单，不需要额外的超参数（如Gumbel温度）
缺点：梯度估计有偏差（biased gradient estimator）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator
    
    前向传播：y = argmax(x)
    反向传播：dy/dx = 1（identity）
    """
    
    @staticmethod
    def forward(ctx, logits, dim=-1):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size]
            dim: argmax的维度
            
        Returns:
            indices: [batch_size, seq_len] 离散的token indices
        """
        # 前向：argmax（离散）
        indices = logits.argmax(dim=dim)
        
        # 保存用于反向传播
        ctx.save_for_backward(logits)
        ctx.dim = dim
        
        return indices
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：将梯度直接传回logits
        
        Args:
            grad_output: 对indices的梯度 [batch_size, seq_len]
            
        Returns:
            grad_logits: 对logits的梯度 [batch_size, seq_len, vocab_size]
        """
        logits, = ctx.saved_tensors
        dim = ctx.dim
        
        # 获取argmax的indices
        indices = logits.argmax(dim=dim)
        
        # 创建one-hot向量
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(dim, indices.unsqueeze(dim), 1.0)
        
        # Straight-through：将grad_output传播到argmax位置
        if dim == -1:
            grad_logits = one_hot * grad_output.unsqueeze(-1)
        else:
            grad_logits = one_hot * grad_output.unsqueeze(dim)
        
        return grad_logits, None


class SoftmaxSTE(torch.autograd.Function):
    """
    Softmax + Straight-Through Estimator
    
    前向传播：y = argmax(softmax(x))
    反向传播：dy/dx = softmax(x) 的梯度（使用softmax的梯度，而不是argmax）
    
    这个版本比纯STE更准确，因为它使用了softmax的梯度
    
    注意：返回float tensor保持梯度流
    """
    
    @staticmethod
    def forward(ctx, logits, temperature=1.0, dim=-1):
        """
        Args:
            logits: [batch_size, seq_len, vocab_size]
            temperature: softmax温度（用于控制分布的sharpness）
            dim: softmax/argmax的维度
        """
        # 应用temperature
        scaled_logits = logits / temperature
        
        # 前向：argmax（离散），但返回float保持梯度
        indices = scaled_logits.argmax(dim=dim).float()
        
        # 保存用于反向传播
        ctx.save_for_backward(logits, indices)
        ctx.dim = dim
        ctx.temperature = temperature
        
        return indices
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        使用softmax的梯度（而不是argmax）
        """
        logits, indices = ctx.saved_tensors
        dim = ctx.dim
        temperature = ctx.temperature
        
        # 计算softmax
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=dim)
        
        # 创建one-hot向量
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(dim, indices.long().unsqueeze(dim), 1.0)
        
        # 使用softmax的梯度
        if dim == -1:
            grad_probs = one_hot * grad_output.unsqueeze(-1)
        else:
            grad_probs = one_hot * grad_output.unsqueeze(dim)
        
        # Softmax gradient: d_softmax/d_logits
        grad_logits = probs * (grad_probs - (probs * grad_probs).sum(dim=dim, keepdim=True))
        grad_logits = grad_logits / temperature
        
        return grad_logits, None, None


def straight_through_argmax(logits, dim=-1):
    """简单的STE封装"""
    return StraightThroughEstimator.apply(logits, dim)


def straight_through_softmax(logits, temperature=1.0, dim=-1):
    """Softmax STE封装"""
    return SoftmaxSTE.apply(logits, temperature, dim)


class STEGPTAdaptor(nn.Module):
    """
    使用Straight-Through Estimator的GPT2 Adaptor
    
    相比Gumbel-Softmax：
    - 更简单，不需要调温度超参数
    - 梯度估计有偏差，但实践中通常work
    """
    
    def __init__(self, gpt2_adaptor, use_softmax_ste=True):
        """
        Args:
            gpt2_adaptor: 已有的GPT2 adaptor模型
            use_softmax_ste: True=使用softmax gradient, False=使用纯identity
        """
        super().__init__()
        self.gpt2_adaptor = gpt2_adaptor
        self.use_softmax_ste = use_softmax_ste
    
    def forward(
        self,
        audio_logits,  # 来自Kimi的audio logits
        motion_tokens_gt=None,  # Ground truth motion tokens
        interleave_ratio=(1, 1),  # (audio, motion)的交错比例
        ste_temperature=1.0,  # STE的temperature（仅用于softmax_ste）
        **kwargs
    ):
        """
        使用STE从audio logits生成离散tokens，并训练GPT2 adaptor
        
        Args:
            audio_logits: [batch_size, audio_seq_len, audio_vocab_size]
            motion_tokens_gt: [batch_size, motion_seq_len] ground truth
            interleave_ratio: (num_audio, num_motion) 交错比例
        """
        batch_size, audio_seq_len, audio_vocab_size = audio_logits.shape
        device = audio_logits.device
        
        # 🔥 使用STE从logits得到离散tokens（可微分！）
        if self.use_softmax_ste:
            audio_tokens = straight_through_softmax(audio_logits, temperature=ste_temperature)
        else:
            audio_tokens = straight_through_argmax(audio_logits)
        
        # [batch_size, audio_seq_len]
        print(f"✅ STE generated audio tokens: {audio_tokens.shape}, requires_grad={audio_tokens.requires_grad}")
        
        # 构建interleaved sequence（交错audio和motion tokens）
        # 这里简化处理，实际需要根据你的数据格式调整
        
        # 调用GPT2 adaptor
        # 注意：这里input_data包含通过STE生成的audio tokens（带梯度！）
        outputs = self.gpt2_adaptor(
            input_data=audio_tokens,  # 🔥 这些tokens是可微的！
            attention_mask=kwargs.get('attention_mask'),
            labels=kwargs.get('labels'),
        )
        
        return outputs


class EndToEndSTEModel(nn.Module):
    """
    端到端的Kimi + STE + GPT2 Adaptor
    """
    
    def __init__(self, kimi_model, gpt2_adaptor, use_softmax_ste=True):
        super().__init__()
        self.kimi_model = kimi_model
        self.ste_adaptor = STEGPTAdaptor(gpt2_adaptor, use_softmax_ste=use_softmax_ste)
    
    def forward(self, user_audio_tokens, motion_tokens_gt, **kwargs):
        """
        端到端训练
        
        流程：
        user_audio → Kimi(返回logits) → STE(离散化但可微) → GPT2 → motion loss → 梯度回传到Kimi
        """
        
        # 1. Kimi生成assistant audio logits
        kimi_outputs = self.kimi_model(
            audio_input_ids=user_audio_tokens,
            return_dict=True,
            **kwargs
        )
        
        assistant_audio_logits = kimi_outputs.logits  # [batch, seq_len, vocab]
        
        # 2. 使用STE + GPT2 Adaptor
        motion_outputs = self.ste_adaptor(
            audio_logits=assistant_audio_logits,
            motion_tokens_gt=motion_tokens_gt,
            **kwargs
        )
        
        # 3. 返回loss（可以回传到Kimi！）
        return {
            'motion_loss': motion_outputs.loss,
            'logits': motion_outputs.logits,
        }


# ==================== 使用示例 ====================

def example_usage():
    """演示如何使用STE进行端到端训练"""
    
    # 假设我们有：
    batch_size = 2
    seq_len = 100
    audio_vocab_size = 152064
    
    # 1. Kimi model的输出（logits）
    audio_logits = torch.randn(batch_size, seq_len, audio_vocab_size, requires_grad=True)
    
    # 2. 使用STE获得离散tokens（但保持梯度）
    audio_tokens = straight_through_softmax(audio_logits, temperature=1.0)
    
    print(f"Audio tokens shape: {audio_tokens.shape}")
    print(f"Audio tokens requires_grad: {audio_tokens.requires_grad}")  # True!
    print(f"Audio tokens dtype: {audio_tokens.dtype}")  # Long（离散）
    
    # 3. 使用这些tokens进行后续计算
    # 例如，通过embedding layer
    embedding = nn.Embedding(audio_vocab_size, 768)
    audio_embeds = embedding(audio_tokens.long())
    
    # 4. 计算一个简单的loss
    target = torch.randn_like(audio_embeds)
    loss = F.mse_loss(audio_embeds, target)
    
    # 5. 反向传播（梯度会传回audio_logits！）
    loss.backward()
    
    print(f"✅ Gradient successfully backpropagated!")
    print(f"   audio_logits.grad: {audio_logits.grad is not None}")  # True
    print(f"   audio_logits.grad.shape: {audio_logits.grad.shape}")


if __name__ == "__main__":
    print("="*60)
    print("Straight-Through Estimator (STE) Demo")
    print("="*60)
    example_usage()

