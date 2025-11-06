#!/usr/bin/env python3
"""
统一的Kimi-Motion模型
将Kimi模型、MLP映射层和GPT2 adaptor合并成一个模型
使用Kimi模型的text和audio hidden state混合作为adaptor输入
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Dict
from transformers import GPT2Config
from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
from HRI_mllm.model.kimi_motion.model import MoonshotKimiaForCausalLM
from HRI_mllm.model.kimi_motion.config import KimiAudioConfig


class HiddenStateMixer(nn.Module):
    """
    简单的Hidden State Mixer
    使用downsample和upsample结构，参考多模态大模型的适配器设计
    """
    def __init__(self, text_hidden_size: int, audio_hidden_size: int, output_hidden_size: int,
                 intermediate_size: int = None, num_layers: int = 2):
        super().__init__()
        self.text_hidden_size = text_hidden_size
        self.audio_hidden_size = audio_hidden_size
        self.output_hidden_size = output_hidden_size
        
        # 计算中间维度：如果未指定，使用text和audio的平均值作为中间维度
        if intermediate_size is None:
            # 使用较大的中间维度以保持表达能力
            intermediate_size = max((text_hidden_size + audio_hidden_size) // 2, output_hidden_size * 2)
        
        self.intermediate_size = intermediate_size
        
        # 第一步：Downsample - 将text和audio特征分别降维到中间维度
        self.text_downsample = nn.Sequential(
            nn.Linear(text_hidden_size, intermediate_size),
            nn.GELU()
        )
        self.audio_downsample = nn.Sequential(
            nn.Linear(audio_hidden_size, intermediate_size),
            nn.GELU()
        )
        
        # 第二步：融合层 - 将text和audio特征融合
        # 使用可学习的混合权重
        self.mix_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # [text_weight, audio_weight]
        
        # 第三步：中间MLP层进行特征处理（可选，用于增强表达能力）
        mlp_layers = []
        for i in range(num_layers):
            mlp_layers.append(nn.Linear(intermediate_size, intermediate_size))
            mlp_layers.append(nn.GELU())
        
        if num_layers > 0:
            self.fusion_mlp = nn.Sequential(*mlp_layers)
        else:
            self.fusion_mlp = nn.Identity()
        
        # 第四步：Upsample - 将中间维度升维到目标维度
        self.upsample = nn.Sequential(
            nn.Linear(intermediate_size, output_hidden_size),
            nn.GELU(),
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
        # 第一步：Downsample
        text_down = self.text_downsample(text_hidden_states)  # [batch_size, seq_len, intermediate_size]
        audio_down = self.audio_downsample(audio_hidden_states)  # [batch_size, seq_len, intermediate_size]
        
        # 第二步：融合 - 使用可学习的权重加权混合
        weights = F.softmax(self.mix_weights, dim=0)
        fused = weights[0] * text_down + weights[1] * audio_down
        
        # 第三步：通过MLP进一步处理
        fused = self.fusion_mlp(fused)
        
        # 第四步：Upsample
        output = self.upsample(fused)  # [batch_size, seq_len, output_hidden_size]
        
        return output


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
                 debug: bool = False,
                 skip_kimi_model: bool = False,
                 skip_adaptor: bool = False,  # 新增：是否跳过adaptor加载
                 adaptor_checkpoint_path: Optional[str] = None,
                 preprocessed_hidden_states_dir: Optional[str] = None,
                 mixer_intermediate_size: Optional[int] = None,
                 mixer_num_layers: int = 2):
        super().__init__()
        
        self.freeze_kimi = freeze_kimi
        self.freeze_adaptor = freeze_adaptor
        self.train_mixer_only = train_mixer_only
        self.motion_loss_weight = motion_loss_weight  # motion token loss权重（更高）
        self.audio_loss_weight = audio_loss_weight    # audio token loss权重（更低）
        self.debug = debug
        self.skip_kimi_model = skip_kimi_model
        self.skip_adaptor = skip_adaptor
        self.kimi_model_path = kimi_model_path
        self.preprocessed_hidden_states_dir = preprocessed_hidden_states_dir
        self.mixer_intermediate_size = mixer_intermediate_size
        self.mixer_num_layers = mixer_num_layers
        
        # 只在非debug模式或主进程打印关键信息
        if not debug or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
            pass  # 移除加载信息，减少输出
        
        # 如果跳过Kimi模型加载（使用预处理的hidden states），则不加载Kimi模型
        if self.skip_kimi_model:
            self.kimi_model = None
            print("✅ Skipping Kimi model loading (using preprocessed hidden states, saving GPU memory)")
            
            # 尝试从预处理数据中检测hidden size
            kimi_hidden_size = None
            if preprocessed_hidden_states_dir and os.path.exists(preprocessed_hidden_states_dir):
                # 尝试从index.json或第一个预处理文件读取hidden size
                index_path = os.path.join(preprocessed_hidden_states_dir, "index.json")
                if os.path.exists(index_path):
                    try:
                        import json
                        with open(index_path, 'r', encoding='utf-8') as f:
                            index_data = json.load(f)
                            samples = index_data.get('samples', [])
                            if samples:
                                # 从第一个样本的metadata或hidden states文件读取
                                first_sample = samples[0]
                                if 'text_hidden_size' in first_sample:
                                    kimi_hidden_size = first_sample['text_hidden_size']
                                    print(f"✅ Detected hidden size from preprocessed data: {kimi_hidden_size}")
                                elif 'hidden_states_path' in first_sample:
                                    # 尝试加载第一个文件来检测维度
                                    hs_path = first_sample['hidden_states_path']
                                    if not os.path.isabs(hs_path):
                                        hs_path = os.path.join(preprocessed_hidden_states_dir, "hidden_states", os.path.basename(hs_path))
                                    if os.path.exists(hs_path):
                                        try:
                                            hs_data = torch.load(hs_path, map_location='cpu', weights_only=False)
                                            if 'text_hidden_states' in hs_data:
                                                kimi_hidden_size = hs_data['text_hidden_states'].shape[-1]
                                                print(f"✅ Detected hidden size from preprocessed file: {kimi_hidden_size}")
                                        except:
                                            pass
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️  Failed to detect hidden size from preprocessed data: {e}")
            
            # 如果无法检测，使用默认值
            if kimi_hidden_size is None:
                kimi_hidden_size = 4096
                print(f"⚠️  Using default hidden_size={kimi_hidden_size} (could not detect from preprocessed data)")
        else:
            # 加载Kimi模型
            self.kimi_model = MoonshotKimiaForCausalLM.from_pretrained(
                kimi_model_path, 
                trust_remote_code=True
            )
            kimi_hidden_size = None  # 将从模型配置中获取
        
        # 冻结Kimi模型参数
        if self.kimi_model is not None:
            if self.freeze_kimi:
                # 冻结所有参数
                for param in self.kimi_model.parameters():
                    param.requires_grad = False
                print("✅ Kimi model parameters fully frozen")
        
        # 获取Kimi模型的hidden state维度
        # 注意：如果使用预处理的hidden states，实际维度可能不同，会在forward中动态适配
        if self.kimi_model is not None:
            kimi_config = self.kimi_model.config
            text_hidden_size = kimi_config.hidden_size  # 通常是4096，但可能是3584
            audio_hidden_size = kimi_config.hidden_size  # 通常是4096，但可能是3584
        elif kimi_hidden_size is not None:
            # 使用提供的hidden size
            text_hidden_size = kimi_hidden_size
            audio_hidden_size = kimi_hidden_size
        else:
            # 默认值（Kimi模型可能是4096或3584）
            # 如果使用预处理的hidden states，会在forward中自动适配
            text_hidden_size = 4096
            audio_hidden_size = 4096
            print(f"⚠️  Using default hidden_size={text_hidden_size} (Kimi model not loaded)")
            print(f"   Note: If using preprocessed hidden states, dimensions will be auto-detected and adjusted")
        
        # 初始化文本tokenizer（使用Kimi模型的tokenizer）
        # 如果跳过Kimi模型加载，则也不加载tokenizer（因为不需要）
        if self.skip_kimi_model:
            self.text_tokenizer = None
            self.kimia_text_blank = 18  # 默认值
            print("✅ Skipping text tokenizer loading (not needed when using preprocessed hidden states)")
        else:
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
        
        # 创建hidden state混合器（使用downsample和upsample结构）
        self.hidden_state_mixer = HiddenStateMixer(
            text_hidden_size=text_hidden_size,
            audio_hidden_size=audio_hidden_size,
            output_hidden_size=gpt2_config.hidden_size,  # GPT2的hidden size，通常是768
            intermediate_size=self.mixer_intermediate_size,
            num_layers=self.mixer_num_layers
        )
        
        # 如果跳过adaptor加载，创建一个轻量级的loss head用于训练
        if self.skip_adaptor:
            # 创建一个简单的projection层用于将mixed_hidden_states映射到motion token logits
            # vocab_size通常为1034 (512*2 + 10)
            vocab_size = gpt2_config.vocab_size if hasattr(gpt2_config, 'vocab_size') else 1034
            self.mixer_loss_head = nn.Linear(gpt2_config.hidden_size, vocab_size)
            self.motion_adaptor = None
            print("✅ Skipping adaptor loading - using lightweight loss head for mixer training")
        else:
            # 创建GPT2 adaptor
            self.motion_adaptor = MixedInputGPT2(
                config=gpt2_config,
                audio_hidden_size=gpt2_config.hidden_size  # 现在输入是混合后的hidden state
            )
            self.mixer_loss_head = None
        
        # 如果跳过了adaptor加载，不需要加载adaptor权重
        if self.skip_adaptor:
            if adaptor_checkpoint_path is not None:
                print("⚠️  Skipping adaptor checkpoint loading (adaptor not loaded)")
        
        # 如果提供了预训练adaptor的checkpoint，加载权重
        elif adaptor_checkpoint_path is not None and os.path.exists(adaptor_checkpoint_path):
            print(f"🔄 Loading pre-trained adaptor from: {adaptor_checkpoint_path}")
            try:
                checkpoint = torch.load(adaptor_checkpoint_path, map_location="cpu", weights_only=False)
                
                # 检查checkpoint格式
                if 'model_state' in checkpoint:
                    # 从motion_adaptor训练的checkpoint加载
                    adaptor_state_dict = checkpoint['model_state']
                    epoch = checkpoint.get('epoch', 0)
                    print(f"   - Checkpoint epoch: {epoch}")
                elif 'model_state_dict' in checkpoint:
                    # 从unified模型checkpoint加载（需要提取adaptor部分）
                    full_state_dict = checkpoint['model_state_dict']
                    adaptor_state_dict = {}
                    for key, value in full_state_dict.items():
                        if key.startswith('motion_adaptor.'):
                            # 移除'motion_adaptor.'前缀
                            new_key = key[len('motion_adaptor.'):]
                            adaptor_state_dict[new_key] = value
                    if not adaptor_state_dict:
                        print("⚠️  No motion_adaptor weights found in checkpoint")
                    else:
                        print(f"   - Extracted adaptor weights from unified model checkpoint")
                else:
                    # 直接是state_dict
                    adaptor_state_dict = checkpoint
                
                if adaptor_state_dict:
                    # 加载权重（使用strict=False以兼容可能的参数不匹配）
                    missing_keys, unexpected_keys = self.motion_adaptor.load_state_dict(
                        adaptor_state_dict, strict=False
                    )
                    if missing_keys:
                        print(f"   ⚠️  Missing keys: {len(missing_keys)} keys (may be normal if config differs)")
                        if self.debug or len(missing_keys) <= 10:
                            # 在debug模式或missing keys较少时显示详细信息
                            for key in missing_keys[:10]:  # 最多显示10个
                                print(f"      - {key}")
                            if len(missing_keys) > 10:
                                print(f"      ... and {len(missing_keys) - 10} more")
                    if unexpected_keys:
                        print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)} keys (may be normal)")
                        # 总是显示unexpected keys的详细信息，因为通常很少
                        for key in unexpected_keys:
                            print(f"      - {key}")
                    print(f"✅ Pre-trained adaptor loaded successfully!")
                else:
                    print(f"⚠️  No adaptor weights found in checkpoint, using random initialization")
            except Exception as e:
                print(f"⚠️  Failed to load adaptor checkpoint: {e}")
                print(f"   Will use random initialization")
                import traceback
                traceback.print_exc()
        else:
            if adaptor_checkpoint_path is not None:
                print(f"⚠️  Adaptor checkpoint not found: {adaptor_checkpoint_path}")
                print(f"   Will use random initialization")
        
        # 冻结adaptor参数（如果adaptor存在）
        if not self.skip_adaptor:
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
        if self.skip_kimi_model:
            print(f"   - Kimi model: Skipped (using preprocessed hidden states)")
        else:
            print(f"   - Kimi model: {'frozen' if self.freeze_kimi else 'trainable'}")
        if self.skip_adaptor:
            print(f"   - Motion adaptor: Skipped (using lightweight loss head)")
        else:
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
                batch: Optional[Dict] = None,
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
        # 检查是否可以使用预处理的hidden states
        text_hidden_states = None
        audio_hidden_states = None
        
        if batch is not None:
            use_preprocessed = batch.get('use_preprocessed', False)
            
            if use_preprocessed:
                # 使用预处理的hidden states（从batch中获取）
                batch_text_hs = batch.get('text_hidden_states')
                batch_audio_hs = batch.get('audio_hidden_states')
                
                if self.debug:
                    print(f"🔍 Checking preprocessed hidden states:")
                    print(f"   - use_preprocessed: {use_preprocessed}")
                    print(f"   - batch_text_hs type: {type(batch_text_hs)}, is None: {batch_text_hs is None}")
                    print(f"   - batch_audio_hs type: {type(batch_audio_hs)}, is None: {batch_audio_hs is None}")
                    if batch_text_hs is not None:
                        print(f"   - batch_text_hs length: {len(batch_text_hs) if isinstance(batch_text_hs, list) else 'N/A'}")
                        if isinstance(batch_text_hs, list) and len(batch_text_hs) > 0:
                            print(f"   - First item type: {type(batch_text_hs[0])}, is None: {batch_text_hs[0] is None}")
                
                if batch_text_hs is not None and batch_audio_hs is not None:
                    # batch_text_hs和batch_audio_hs是列表，每个元素是一个样本的hidden states（可能为None）
                    # 检查有多少样本有预处理数据
                    valid_indices = []
                    text_hs_list = []
                    audio_hs_list = []
                    
                    for i in range(len(batch_text_hs)):
                        text_hs = batch_text_hs[i]
                        audio_hs = batch_audio_hs[i]
                        
                        if text_hs is not None and audio_hs is not None:
                            if isinstance(text_hs, torch.Tensor) and isinstance(audio_hs, torch.Tensor):
                                valid_indices.append(i)
                                # 确保在CPU上
                                text_hs = text_hs.cpu() if text_hs.is_cuda else text_hs
                                audio_hs = audio_hs.cpu() if audio_hs.is_cuda else audio_hs
                                text_hs_list.append(text_hs)
                                audio_hs_list.append(audio_hs)
                    
                    # 如果所有样本都有预处理数据
                    if len(valid_indices) == len(batch_text_hs):
                        if text_hs_list and len(text_hs_list) > 0:
                            # 获取最大序列长度
                            max_seq_len = max(hs.shape[0] if len(hs.shape) > 1 else hs.shape[0] for hs in text_hs_list)
                            
                            if max_seq_len > 0:
                                # 获取hidden size（从实际数据中获取）
                                actual_hidden_size = text_hs_list[0].shape[-1]
                                batch_size = len(text_hs_list)
                                
                                # 检查hidden size是否与mixer的期望匹配
                                expected_hidden_size = self.hidden_state_mixer.text_hidden_size
                                if actual_hidden_size != expected_hidden_size:
                                    print(f"⚠️  WARNING: Hidden states dimension mismatch!")
                                    print(f"   - Expected: {expected_hidden_size} (from mixer)")
                                    print(f"   - Actual: {actual_hidden_size} (from preprocessed data)")
                                    print(f"   - Attempting to adjust mixer input dimensions...")
                                    
                                    # 动态调整mixer的input维度（如果可能）
                                    # 如果mixer已经初始化，我们需要重新创建或调整它
                                    # 这里我们创建一个新的projection层来适配
                                    from torch.nn import Linear
                                    
                                    # 临时创建适配的projection层
                                    if not hasattr(self, '_temp_text_proj') or self._temp_text_proj.in_features != actual_hidden_size:
                                        self._temp_text_proj = Linear(actual_hidden_size, self.hidden_state_mixer.output_hidden_size).to(device)
                                        self._temp_audio_proj = Linear(actual_hidden_size, self.hidden_state_mixer.output_hidden_size).to(device)
                                        print(f"   - Created temporary projection layers: {actual_hidden_size} -> {self.hidden_state_mixer.output_hidden_size}")
                                    
                                    # 使用临时projection层
                                    hidden_size = actual_hidden_size
                                else:
                                    hidden_size = actual_hidden_size
                                
                                # Padding到相同长度并batch
                                batch_text_hs_tensor = torch.zeros(batch_size, max_seq_len, hidden_size, dtype=text_hs_list[0].dtype)
                                batch_audio_hs_tensor = torch.zeros(batch_size, max_seq_len, hidden_size, dtype=audio_hs_list[0].dtype)
                                
                                for i, (text_hs, audio_hs) in enumerate(zip(text_hs_list, audio_hs_list)):
                                    seq_len = text_hs.shape[0]
                                    batch_text_hs_tensor[i, :seq_len] = text_hs
                                    batch_audio_hs_tensor[i, :seq_len] = audio_hs
                                
                                text_hidden_states = batch_text_hs_tensor.to(device)
                                audio_hidden_states = batch_audio_hs_tensor.to(device)
                                
                                # 如果使用了临时projection，标记一下
                                if actual_hidden_size != expected_hidden_size:
                                    self._use_temp_proj = True
                                else:
                                    self._use_temp_proj = False
                    elif len(valid_indices) == 0:
                        # 所有样本都没有预处理数据
                        if self.debug:
                            print(f"⚠️  Batch marked as use_preprocessed=True but no valid preprocessed hidden states found")
                        text_hidden_states = None
                        audio_hidden_states = None
                    else:
                        # 部分样本有预处理数据，部分没有
                        # 如果Kimi模型不可用，报错并提示
                        if self.kimi_model is None:
                            missing_count = len(batch_text_hs) - len(valid_indices)
                            raise ValueError(
                                f"Cannot compute hidden states: {missing_count}/{len(batch_text_hs)} samples in this batch "
                                f"are missing preprocessed hidden states, but Kimi model is not loaded. "
                                f"Please either:\n"
                                f"  1. Ensure all samples have preprocessed hidden states, or\n"
                                f"  2. Load the Kimi model (set skip_kimi_model=False) to compute missing hidden states on-the-fly"
                            )
                        # 如果Kimi模型可用，fallback到Kimi模型计算所有样本
                        if self.debug:
                            print(f"⚠️  Batch has {len(valid_indices)}/{len(batch_text_hs)} samples with preprocessed hidden states, "
                                  f"will compute remaining {len(batch_text_hs) - len(valid_indices)} samples using Kimi model")
                        text_hidden_states = None
                        audio_hidden_states = None
        
        # 如果没有预处理的hidden states，调用Kimi模型
        if text_hidden_states is None or audio_hidden_states is None:
            if self.kimi_model is None:
                raise ValueError(
                    "Cannot compute hidden states: Kimi model is not loaded and preprocessed hidden states are not available. "
                    "Please either load the Kimi model or provide preprocessed hidden states."
                )
            
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
                    
                    # 获取text和audio的hidden states
                    if hasattr(kimi_outputs, 'hidden_states') and kimi_outputs.hidden_states:
                        # 获取最后一层的hidden states
                        last_hidden_states = kimi_outputs.hidden_states[-1]
                        
                        if isinstance(last_hidden_states, tuple):
                            text_hidden_states, audio_hidden_states = last_hidden_states[0], last_hidden_states[1]
                        else:
                            # 如果只有一个hidden state，复制一份作为text和audio
                            text_hidden_states = last_hidden_states
                            audio_hidden_states = last_hidden_states
                    else:
                        # 这里需要从logits反推hidden states，或者修改Kimi模型以返回hidden states
                        raise NotImplementedError("需要修改Kimi模型以返回hidden states")
                    
                    # 保存kimi_outputs用于后续计算audio loss（如果需要）
                    # 注意：这里需要在外部作用域也能访问到kimi_outputs
                    kimi_outputs_for_loss = kimi_outputs
                except Exception as e:
                    print(f"❌ Error in Kimi model call: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        
        # 2. 混合hidden states
        try:
            # 如果使用临时projection（维度不匹配时），先投影再混合
            if hasattr(self, '_use_temp_proj') and self._use_temp_proj:
                # 使用临时projection层投影到mixer的输出维度
                text_proj = self._temp_text_proj(text_hidden_states)
                audio_proj = self._temp_audio_proj(audio_hidden_states)
                # 应用mixer的混合权重
                weights = torch.nn.functional.softmax(self.hidden_state_mixer.mix_weights, dim=0)
                mixed_hidden_states = weights[0] * text_proj + weights[1] * audio_proj
            else:
                mixed_hidden_states = self.hidden_state_mixer(text_hidden_states, audio_hidden_states)
        except Exception as e:
            print(f"❌ Error in hidden state mixer: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 3. 如果跳过adaptor，使用轻量级loss head直接计算loss，或者使用保存的target_hidden_states
        loss = None
        logits = None
        use_target_hidden_states = False
        
        if self.skip_adaptor:
            # 检查batch中是否有target_hidden_states（从预处理的hidden states中加载）
            if batch is not None and batch.get('target_hidden_states') is not None:
                target_hidden_states_list = batch.get('target_hidden_states')
                
                if target_hidden_states_list is not None and len(target_hidden_states_list) > 0:
                    # 检查是否有有效的target_hidden_states
                    valid_targets = [t for t in target_hidden_states_list if t is not None]
                    if valid_targets:
                        # 使用MSE loss，让mixer的输出匹配目标hidden states（在对应位置）
                        # 注意：target_hidden_states是adaptor transformer输出的完整序列
                        # 我们需要对齐mixer输出到audio token位置
                        
                        # 从labels中获取audio token位置
                        if labels is not None:
                            audio_mask = (labels == -100)
                            batch_size, seq_len = labels.shape
                            
                            # 对齐目标hidden states到mixer输出位置
                            loss_list = []
                            for batch_idx in range(batch_size):
                                audio_positions = torch.where(audio_mask[batch_idx])[0]
                                if len(audio_positions) > 0 and batch_idx < len(valid_targets):
                                    target_hs = valid_targets[batch_idx].to(mixed_hidden_states.device)
                                    sample_mixed = mixed_hidden_states[batch_idx]
                                    
                                    # 对齐长度
                                    min_len = min(len(audio_positions), sample_mixed.shape[0], target_hs.shape[0])
                                    if min_len > 0:
                                        # 获取目标位置的hidden states
                                        target_positions = audio_positions[:min_len]
                                        target_subset = target_hs[target_positions]  # [min_len, hidden_size]
                                        mixed_subset = sample_mixed[:min_len]  # [min_len, hidden_size]
                                        
                                        # 计算MSE loss
                                        mse_loss = nn.functional.mse_loss(mixed_subset, target_subset)
                                        loss_list.append(mse_loss)
                            
                            if loss_list:
                                loss = torch.stack(loss_list).mean()
                                use_target_hidden_states = True
            
            # 如果没有target_hidden_states或计算失败，fallback到loss head方式
            if not use_target_hidden_states:
                # 使用轻量级loss head
                logits = self.mixer_loss_head(mixed_hidden_states)  # [batch_size, seq_len, vocab_size]
                
                # 计算loss（如果有labels）
                if labels is not None:
                    # 由于我们只有audio token的hidden states，我们需要一个映射策略
                    # 方案1：使用motion_tokens作为target，通过attention或pooling对齐
                    # 方案2：直接预测每个audio token位置对应的下一个motion token
                    
                    # 这里使用方案2：对于每个audio token的hidden state，预测对应的motion token
                    # 需要从labels中提取motion tokens，并与mixed_hidden_states对齐
                    
                    # 获取motion tokens（从labels或motion_tokens参数）
                    if motion_tokens is not None:
                        # 直接使用motion_tokens作为target
                        batch_size = mixed_hidden_states.shape[0]
                        audio_seq_len = mixed_hidden_states.shape[1]
                        motion_seq_len = motion_tokens.shape[1] if len(motion_tokens.shape) > 1 else len(motion_tokens)
                        
                        # 将motion tokens对齐到audio sequence
                        # 简单的方案：截断或padding到相同长度
                        min_len = min(audio_seq_len, motion_seq_len)
                        
                        # 使用交叉熵损失
                        shift_logits = logits[:, :min_len, :].contiguous()  # [batch_size, min_len, vocab_size]
                        if len(motion_tokens.shape) == 1:
                            shift_labels = motion_tokens[:min_len].unsqueeze(0).expand(batch_size, -1)
                        else:
                            shift_labels = motion_tokens[:, :min_len].contiguous()
                        
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                    else:
                        # 如果没有motion_tokens，尝试从labels中提取
                        # 这是fallback方案
                        motion_mask = (labels != -100)
                        if motion_mask.any():
                            # 提取motion token IDs
                            motion_token_ids = labels[motion_mask]
                            # 这个方案比较复杂，需要复杂的对齐逻辑
                            # 暂时使用简单的loss
                            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                            # 简化：只计算motion token位置的loss
                            motion_positions = torch.where(motion_mask.view(-1))[0]
                            if len(motion_positions) > 0:
                                # 选择对应的logits和labels
                                flat_logits = logits.view(-1, logits.size(-1))
                                selected_logits = flat_logits[motion_positions]
                                selected_labels = labels[motion_mask]
                                loss = loss_fct(selected_logits, selected_labels)
            
            # 返回输出
            from transformers.modeling_outputs import CausalLMOutputWithPast
            output = CausalLMOutputWithPast(
                loss=loss * self.motion_loss_weight if loss is not None else None,
                logits=logits if 'logits' in locals() else None,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
            if loss is not None:
                output.motion_loss = loss
            output.audio_loss = None
            return output
        
        # 3. 通过motion adaptor生成motion tokens（原来的逻辑）
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
                    if self.motion_adaptor is None:
                        raise ValueError("Cannot use adaptor: motion_adaptor is None (skip_adaptor=True)")
                    
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
        # 注意：当使用预处理的hidden states时，没有kimi_outputs，所以无法计算audio loss
        # 如果使用预处理的hidden states，跳过audio loss计算
        kimi_audio_loss = None
        if assistant_audio_tokens is not None and not self.skip_kimi_model:
            # 当调用Kimi模型时，audio loss应该在kimi_outputs中计算
            # 但由于kimi_outputs在with块内，这里无法直接访问
            # 为了简化，如果使用预处理的hidden states（skip_kimi_model=True），则不计算audio loss
            # 如果需要audio loss，可以在调用Kimi模型时通过labels参数自动计算
            pass
        
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
            'mixer_intermediate_size': getattr(self, 'mixer_intermediate_size', None),
            'mixer_num_layers': getattr(self, 'mixer_num_layers', 2),
        }
        torch.save(save_dict, save_path)
        
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
        mixer_intermediate_size = checkpoint.get('mixer_intermediate_size', None)
        mixer_num_layers = checkpoint.get('mixer_num_layers', 2)
        
        # 创建模型实例（需要提供kimi_model_path和gpt2_config）
        # 这里需要根据实际情况调整
        model = cls(
            kimi_model_path=kimi_model_path,
            gpt2_config=GPT2Config(),  # 需要从checkpoint中恢复
            freeze_kimi=freeze_kimi,
            freeze_adaptor=freeze_adaptor,
            train_mixer_only=train_mixer_only,
            mixer_intermediate_size=mixer_intermediate_size,
            mixer_num_layers=mixer_num_layers
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
                        debug: bool = False,
                        skip_kimi_model: bool = False,
                        skip_adaptor: bool = False,  # 新增：是否跳过adaptor加载
                        adaptor_checkpoint_path: Optional[str] = None,
                        preprocessed_hidden_states_dir: Optional[str] = None,
                        mixer_intermediate_size: Optional[int] = None,
                        mixer_num_layers: int = 2) -> UnifiedKimiMotionModel:
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
        debug: 是否启用调试模式
        skip_kimi_model: 是否跳过Kimi模型加载（当使用预处理的hidden states时，可以设为True以节省显存）
        skip_adaptor: 是否跳过adaptor加载（当只训练mixer时，可以设为True以节省显存，使用轻量级loss head）
        adaptor_checkpoint_path: 预训练adaptor的checkpoint路径（可选，如果提供将加载预训练权重）
        preprocessed_hidden_states_dir: 预处理hidden states目录（用于自动检测hidden size）
        mixer_intermediate_size: Mixer中间维度（默认None，自动计算）
        mixer_num_layers: Mixer的MLP层数（默认2）
    
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
        debug=debug,
        skip_kimi_model=skip_kimi_model,
        skip_adaptor=skip_adaptor,
        adaptor_checkpoint_path=adaptor_checkpoint_path,
        preprocessed_hidden_states_dir=preprocessed_hidden_states_dir,
        mixer_intermediate_size=mixer_intermediate_size,
        mixer_num_layers=mixer_num_layers
    )
    
    return model
