#!/usr/bin/env python3
"""
为mixer准备训练数据
遍历beat数据集，使用Kimi模型获取hidden states，存储mixer需要的输入和输出

存储格式：
- text_hidden_states: [seq_len, hidden_size]
- audio_hidden_states: [seq_len, hidden_size]  
- motion_tokens: [motion_len]
- user_text: str
- user_audio_tokens: [user_audio_len]
- assistant_audio_tokens: [assistant_audio_len]
- interleaved_sequence: [total_seq_len]
- token_labels: [total_seq_len]  (用于训练时区分audio和motion位置)

如果提供了adaptor checkpoint，还会计算并保存：
- target_hidden_states: [seq_len, adaptor_hidden_size] (adaptor transformer输出的目标hidden states)
  这样训练mixer时就不需要额外的loss head，只需要让mixer的输出匹配这个目标
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import argparse

# 设置MuJoCo渲染
os.environ['MUJOCO_GL'] = 'egl'

from kimia_infer.api.kimia import KimiAudio
from HRI_mllm.model.kimi_motion.model import MoonshotKimiaForCausalLM
from transformers import AutoTokenizer, GPT2Config
from HRI_mllm.datasets.json_audio_motion_dataset import JSONAudioMotionDataset


def load_kimi_model(kimi_model_path: str, device: str = "cuda"):
    """加载Kimi模型"""
    print(f"🔄 Loading Kimi model from: {kimi_model_path}")
    
    model = MoonshotKimiaForCausalLM.from_pretrained(
        kimi_model_path,
        trust_remote_code=True
    )
    model.eval()
    model.to(device)
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"✅ Kimi model loaded successfully")
    return model


def load_text_tokenizer(kimi_model_path: str):
    """加载文本tokenizer"""
    print(f"🔄 Loading text tokenizer from: {kimi_model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            kimi_model_path,
            trust_remote_code=True
        )
        
        # 获取extra_tokens（特别是kimia_text_blank）
        try:
            from kimia_infer.utils.special_tokens import instantiate_extra_tokens
            extra_tokens = instantiate_extra_tokens(tokenizer)
            kimia_text_blank = extra_tokens.kimia_text_blank
            print(f"✅ Text tokenizer loaded, kimia_text_blank={kimia_text_blank}")
        except Exception as e:
            print(f"⚠️  Failed to load extra tokens: {e}")
            # 尝试直接获取kimia_text_blank
            if hasattr(tokenizer, "special_tokens"):
                try:
                    kimia_text_blank = tokenizer.special_tokens["<|im_kimia_text_blank|>"]
                except:
                    kimia_text_blank = 18  # 默认值
            elif hasattr(tokenizer, "convert_tokens_to_ids"):
                try:
                    kimia_text_blank = tokenizer.convert_tokens_to_ids("<|im_kimia_text_blank|>")
                except:
                    kimia_text_blank = 18  # 默认值
            else:
                kimia_text_blank = 18  # 默认值
            print(f"⚠️  Using default kimia_text_blank={kimia_text_blank}")
        
        return tokenizer, kimia_text_blank
    except Exception as e:
        print(f"⚠️  Failed to load text tokenizer: {e}")
        return None, 18


def get_hidden_states_from_kimi(
    kimi_model,
    text_tokenizer,
    kimia_text_blank: int,
    user_text: Optional[str],
    user_audio_tokens: torch.Tensor,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从Kimi模型获取hidden states
    
    Args:
        kimi_model: Kimi模型
        text_tokenizer: 文本tokenizer
        kimia_text_blank: kimia_text_blank token ID
        user_text: 用户文本（可选）
        user_audio_tokens: 用户音频tokens [seq_len]
        device: 设备
    
    Returns:
        text_hidden_states: [seq_len, hidden_size]
        audio_hidden_states: [seq_len, hidden_size]
    """
    with torch.no_grad():
        # 准备输入
        batch_size = 1
        audio_seq_len = len(user_audio_tokens)
        
        # 将audio tokens转换为batch格式
        audio_input_ids = user_audio_tokens.unsqueeze(0).to(device)  # [1, seq_len]
        
        # 创建attention mask
        attention_mask = torch.ones(
            batch_size, audio_seq_len,
            device=device, dtype=torch.long
        )
        
        # 处理文本输入
        text_input_ids = None
        if user_text is not None and user_text.strip() != "" and text_tokenizer is not None:
            try:
                encoded = text_tokenizer.encode(user_text, bos=False, eos=False)
                # 截断到最大长度（512）
                if len(encoded) > 512:
                    encoded = encoded[:512]
                text_input_ids = torch.tensor([encoded], device=device, dtype=torch.long)
            except Exception as e:
                print(f"⚠️  Text tokenization failed: {e}")
                text_input_ids = None
        
        # 如果text_input_ids为None或长度不匹配，创建全为kimia_text_blank的tensor
        if text_input_ids is None:
            text_input_ids = torch.full(
                (batch_size, audio_seq_len),
                kimia_text_blank,
                device=device,
                dtype=torch.long
            )
        else:
            # 如果text序列长度不等于audio序列长度，需要进行padding
            text_seq_len = text_input_ids.shape[1]
            if text_seq_len != audio_seq_len:
                padded_text_input_ids = torch.full(
                    (batch_size, audio_seq_len),
                    kimia_text_blank,
                    device=device,
                    dtype=torch.long
                )
                actual_len = min(text_seq_len, audio_seq_len)
                padded_text_input_ids[:, :actual_len] = text_input_ids[:, :actual_len]
                text_input_ids = padded_text_input_ids
        
        # 调用Kimi模型获取hidden states
        kimi_outputs = kimi_model(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 提取hidden states
        if hasattr(kimi_outputs, 'hidden_states') and kimi_outputs.hidden_states:
            # 获取最后一层的hidden states
            last_hidden_states = kimi_outputs.hidden_states[-1]
            
            if isinstance(last_hidden_states, tuple):
                # 如果有多个hidden states（text和audio分开）
                text_hidden_states = last_hidden_states[0].squeeze(0)  # [seq_len, hidden_size]
                audio_hidden_states = last_hidden_states[1].squeeze(0)  # [seq_len, hidden_size]
            else:
                # 如果只有一个hidden state，复制一份作为text和audio
                hidden_states = last_hidden_states.squeeze(0)  # [seq_len, hidden_size]
                text_hidden_states = hidden_states
                audio_hidden_states = hidden_states
        else:
            raise ValueError("Kimi model did not return hidden_states")
        
        # 移动到CPU以节省显存
        text_hidden_states = text_hidden_states.cpu()
        audio_hidden_states = audio_hidden_states.cpu()
        
        return text_hidden_states, audio_hidden_states


def load_adaptor(adaptor_checkpoint_path: str, device: str = "cuda"):
    """加载adaptor模型"""
    print(f"🔄 Loading adaptor from: {adaptor_checkpoint_path}")
    
    from HRI_mllm.model.gpt2_adaptor.model import MixedInputGPT2
    
    try:
        checkpoint = torch.load(adaptor_checkpoint_path, map_location=device, weights_only=False)
        
        # 检查checkpoint格式
        if 'model_state' in checkpoint:
            adaptor_state_dict = checkpoint['model_state']
        elif 'model_state_dict' in checkpoint:
            full_state_dict = checkpoint['model_state_dict']
            adaptor_state_dict = {}
            for key, value in full_state_dict.items():
                if key.startswith('motion_adaptor.'):
                    new_key = key[len('motion_adaptor.'):]
                    adaptor_state_dict[new_key] = value
                elif not key.startswith('hidden_state_mixer.'):
                    # 如果没有前缀，直接使用
                    adaptor_state_dict[key] = value
        else:
            adaptor_state_dict = checkpoint
        
        # 尝试从checkpoint中获取config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # 使用默认GPT2配置
            config = GPT2Config(
                vocab_size=1034,
                n_positions=4096,
                n_embd=768,
                n_layer=12,
                n_head=12,
            )
        
        # 创建adaptor模型
        adaptor = MixedInputGPT2(config=config, audio_hidden_size=768)
        adaptor.load_state_dict(adaptor_state_dict, strict=False)
        adaptor.eval()
        adaptor.to(device)
        
        # 冻结所有参数
        for param in adaptor.parameters():
            param.requires_grad = False
        
        print(f"✅ Adaptor loaded successfully")
        return adaptor, config
    except Exception as e:
        print(f"❌ Failed to load adaptor: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compute_target_hidden_states(
    text_hidden_states: torch.Tensor,  # [seq_len, kimi_hidden_size]
    audio_hidden_states: torch.Tensor,  # [seq_len, kimi_hidden_size]
    motion_tokens: torch.Tensor,  # [motion_len]
    interleaved_sequence: torch.Tensor,  # [total_seq_len]
    token_labels: torch.Tensor,  # [total_seq_len]
    adaptor,
    device: str = "cuda",
    mixer_output_size: int = 768  # mixer的输出维度，通常是adaptor的hidden_size
) -> torch.Tensor:
    """
    计算adaptor transformer输出的目标hidden states
    
    这个函数：
    1. 将text和audio hidden states混合（使用简单的加权平均作为mixer的近似）
    2. 投影到adaptor输入维度
    3. 通过adaptor transformer处理
    4. 返回transformer输出的hidden states作为mixer训练的目标
    
    注意：这里不使用实际的mixer，而是使用简单的混合方式作为初始目标
    """
    batch_size = 1
    seq_len = len(interleaved_sequence)
    kimi_hidden_size = text_hidden_states.shape[-1]
    
    # 1. 简单的混合text和audio hidden states（作为mixer的近似）
    # 这里使用平均混合，实际训练时mixer会学习更好的混合方式
    mixed_hidden_states = 0.5 * text_hidden_states + 0.5 * audio_hidden_states  # [seq_len, kimi_hidden_size]
    mixed_hidden_states = mixed_hidden_states.to(device)
    
    # 2. 投影到adaptor输入维度（adaptor.config.hidden_size，通常是768）
    # 注意：adaptor的input_projection是从audio_hidden_size到config.hidden_size
    # 我们需要确保投影到adaptor.config.hidden_size
    target_hidden_size = adaptor.config.hidden_size
    
    # 如果维度已经匹配，不需要投影
    if kimi_hidden_size == target_hidden_size:
        projected_mixed = mixed_hidden_states
    else:
        # 创建/使用投影层，确保维度正确
        # 创建唯一的投影层名称，避免冲突
        proj_key = f'_temp_kimi_to_adaptor_proj_{kimi_hidden_size}_to_{target_hidden_size}'
        
        if not hasattr(adaptor, proj_key):
            # 创建新的投影层
            proj_layer = torch.nn.Linear(kimi_hidden_size, target_hidden_size).to(device)
            torch.nn.init.xavier_uniform_(proj_layer.weight)
            for param in proj_layer.parameters():
                param.requires_grad = False
            setattr(adaptor, proj_key, proj_layer)
        
        # 使用投影层
        projection_layer = getattr(adaptor, proj_key)
        projected_mixed = projection_layer(mixed_hidden_states)
    
    # 验证投影后的维度
    actual_hidden_size = projected_mixed.shape[-1]
    if actual_hidden_size != target_hidden_size:
        # 如果投影失败，强制再次投影
        print(f"⚠️  Warning: Projection mismatch detected. Expected {target_hidden_size}, got {actual_hidden_size}")
        print(f"   - kimi_hidden_size: {kimi_hidden_size}")
        print(f"   - target_hidden_size: {target_hidden_size}")
        print(f"   - Creating new projection layer...")
        # 强制创建新的投影层
        if hasattr(adaptor, '_temp_kimi_to_adaptor_proj'):
            # 删除旧的
            del adaptor._temp_kimi_to_adaptor_proj
        adaptor._temp_kimi_to_adaptor_proj = torch.nn.Linear(
            actual_hidden_size, target_hidden_size
        ).to(device)
        torch.nn.init.xavier_uniform_(adaptor._temp_kimi_to_adaptor_proj.weight)
        for param in adaptor._temp_kimi_to_adaptor_proj.parameters():
            param.requires_grad = False
        projected_mixed = adaptor._temp_kimi_to_adaptor_proj(projected_mixed)
    
    # 最终验证
    assert projected_mixed.shape[-1] == target_hidden_size, \
        f"Projection failed: expected {target_hidden_size}, got {projected_mixed.shape[-1]}"
    
    # 3. 创建final hidden states序列（包含mixed hidden states和motion embeddings）
    final_hidden_states = torch.zeros(
        batch_size, seq_len, adaptor.config.hidden_size,
        device=device, dtype=projected_mixed.dtype
    )
    
    # 获取audio和motion mask
    audio_mask = (token_labels == -100)
    motion_mask = (token_labels != -100)
    
    # 将混合后的hidden states放到audio token位置
    if audio_mask.any():
        audio_positions = torch.where(audio_mask)[0]
        if len(audio_positions) > 0:
            use_len = min(len(audio_positions), projected_mixed.shape[0])
            final_hidden_states[0, audio_positions[:use_len]] = projected_mixed[:use_len]
    
    # 将motion token embeddings放到motion token位置
    if motion_mask.any():
        motion_token_ids = interleaved_sequence[motion_mask]
        motion_embeddings = adaptor.transformer.wte(motion_token_ids.long().to(device))
        final_hidden_states[0, motion_mask] = motion_embeddings
    
    # 4. 通过adaptor transformer（只到transformer层，不包括lm_head）
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    
    with torch.no_grad():
        transformer_outputs = adaptor.transformer(
            inputs_embeds=final_hidden_states,
            attention_mask=attention_mask
        )
        
        # 返回transformer输出的hidden states（在audio token位置）
        # 这是mixer训练的目标：让mixer输出的投影结果匹配这个
        target_hidden_states = transformer_outputs[0].squeeze(0)  # [seq_len, adaptor_hidden_size]
    
    return target_hidden_states


def process_dataset(
    json_paths: List[str],
    kimi_model_path: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    debug: bool = False,
    adaptor_checkpoint_path: Optional[str] = None,
    mixer_checkpoint_path: Optional[str] = None
):
    """
    处理数据集，生成mixer训练数据
    
    Args:
        json_paths: JSON文件路径列表
        kimi_model_path: Kimi模型路径
        output_dir: 输出目录
        device: 设备
        batch_size: 批次大小（当前仅支持1）
        max_samples: 最大处理样本数（None表示处理所有）
        debug: 是否开启调试模式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    hidden_states_dir = os.path.join(output_dir, "hidden_states")
    os.makedirs(hidden_states_dir, exist_ok=True)
    
    # 加载Kimi模型和tokenizer
    kimi_model = load_kimi_model(kimi_model_path, device)
    text_tokenizer, kimia_text_blank = load_text_tokenizer(kimi_model_path)
    
    # 可选：加载adaptor（如果提供）
    adaptor = None
    adaptor_config = None
    if adaptor_checkpoint_path and os.path.exists(adaptor_checkpoint_path):
        adaptor, adaptor_config = load_adaptor(adaptor_checkpoint_path, device)
        if adaptor is not None:
            print("💡 Note: Will compute target_hidden_states using adaptor (without mixer, using simple mixing as approximation)")
    
    # 加载数据集
    print(f"🔄 Loading datasets from {len(json_paths)} files...")
    all_samples = []
    for json_path in json_paths:
        dataset = JSONAudioMotionDataset(
            json_path=json_path,
            max_audio_length=2048,
            max_motion_length=1024,
            interleave_ratio=(1, 1),
            debug=debug
        )
        all_samples.extend(dataset.samples)
    
    print(f"📊 Total samples: {len(all_samples)}")
    if max_samples is not None:
        all_samples = all_samples[:max_samples]
        print(f"📊 Processing first {len(all_samples)} samples")
    
    # 处理每个样本
    processed_samples = []
    failed_samples = []
    
    for sample_idx, sample in enumerate(tqdm(all_samples, desc="Processing samples")):
        try:
            # 提取数据
            user_text = sample.get('user_text')
            user_audio_tokens = sample['user_audio_tokens']  # [user_audio_len]
            assistant_audio_tokens = sample['assistant_audio_tokens']  # [assistant_audio_len]
            motion_tokens = sample['motion_tokens']  # [motion_len]
            interleaved_sequence = sample['interleaved_sequence']  # [total_seq_len]
            token_labels = sample['token_labels']  # [total_seq_len]
            
            # 使用user_audio_tokens获取hidden states（这是mixer的输入）
            text_hidden_states, audio_hidden_states = get_hidden_states_from_kimi(
                kimi_model=kimi_model,
                text_tokenizer=text_tokenizer,
                kimia_text_blank=kimia_text_blank,
                user_text=user_text,
                user_audio_tokens=user_audio_tokens,
                device=device
            )
            
            # 可选：如果有adaptor，计算目标hidden states
            # 注意：这里不使用mixer（因为mixer还没训练好），而是使用简单的混合作为近似
            target_hidden_states = None
            if adaptor is not None:
                try:
                    # 计算目标hidden states（使用简单的text+audio混合作为mixer的近似）
                    target_hidden_states = compute_target_hidden_states(
                        text_hidden_states=text_hidden_states,
                        audio_hidden_states=audio_hidden_states,
                        motion_tokens=motion_tokens,
                        interleaved_sequence=interleaved_sequence,
                        token_labels=token_labels,
                        adaptor=adaptor,
                        device=device,
                        mixer_output_size=adaptor_config.hidden_size if adaptor_config else 768
                    )
                except Exception as e:
                    print(f"⚠️  Failed to compute target_hidden_states for sample {sample_idx}: {e}")
                    if debug:
                        import traceback
                        traceback.print_exc()
                    target_hidden_states = None
            
            # 保存hidden states到文件
            sample_id = f"sample_{sample_idx:06d}"
            hidden_states_path = os.path.join(hidden_states_dir, f"{sample_id}.pt")
            
            # 保存所有数据
            saved_data = {
                'text_hidden_states': text_hidden_states,  # [seq_len, hidden_size]
                'audio_hidden_states': audio_hidden_states,  # [seq_len, hidden_size]
                'motion_tokens': motion_tokens,  # [motion_len]
                'user_text': user_text,
                'user_audio_tokens': user_audio_tokens.cpu(),  # [user_audio_len]
                'assistant_audio_tokens': assistant_audio_tokens.cpu(),  # [assistant_audio_len]
                'interleaved_sequence': interleaved_sequence.cpu(),  # [total_seq_len]
                'token_labels': token_labels.cpu(),  # [total_seq_len]
                'user_audio_length': sample['user_audio_length'],
                'assistant_audio_length': sample['assistant_audio_length'],
                'motion_length': sample['motion_length'],
                'sequence_length': sample['sequence_length'],
            }
            
            # 如果计算了目标hidden states，也保存
            if target_hidden_states is not None:
                saved_data['target_hidden_states'] = target_hidden_states.cpu()  # [seq_len, adaptor_hidden_size]
            
            torch.save(saved_data, hidden_states_path)
            
            # 添加到索引（确保包含sample_idx以匹配dataset）
            processed_samples.append({
                'sample_idx': sample.get('sample_idx', sample_idx),  # 从sample中获取，如果没有则使用当前索引
                'sample_id': sample_id,
                'hidden_states_path': f"hidden_states/{sample_id}.pt",
                'user_text': user_text,
                'user_audio_length': sample['user_audio_length'],
                'assistant_audio_length': sample['assistant_audio_length'],
                'motion_length': sample['motion_length'],
                'sequence_length': sample['sequence_length'],
                'text_hidden_size': text_hidden_states.shape[-1],
                'audio_hidden_size': audio_hidden_states.shape[-1],
            })
            
            # Debug模式：只处理第一个样本
            if debug and sample_idx == 0:
                print(f"\n🔍 Debug sample {sample_idx}:")
                print(f"   - Text hidden states: {text_hidden_states.shape}")
                print(f"   - Audio hidden states: {audio_hidden_states.shape}")
                print(f"   - Motion tokens: {motion_tokens.shape}")
                print(f"   - Interleaved sequence: {interleaved_sequence.shape}")
                print(f"   - Token labels: {token_labels.shape}")
                break
                
        except Exception as e:
            print(f"❌ Error processing sample {sample_idx}: {e}")
            if debug:
                import traceback
                traceback.print_exc()
            failed_samples.append(sample_idx)
            continue
    
    # 保存索引文件
    index_path = os.path.join(output_dir, "index.json")
    index_data = {
        'total_samples': len(processed_samples),
        'failed_samples': len(failed_samples),
        'text_hidden_size': processed_samples[0]['text_hidden_size'] if processed_samples else None,
        'audio_hidden_size': processed_samples[0]['audio_hidden_size'] if processed_samples else None,
        'samples': processed_samples
    }
    
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processing completed!")
    print(f"   - Processed samples: {len(processed_samples)}")
    print(f"   - Failed samples: {len(failed_samples)}")
    print(f"   - Output directory: {output_dir}")
    print(f"   - Index file: {index_path}")
    
    if failed_samples:
        print(f"   - Failed sample indices: {failed_samples[:10]}{'...' if len(failed_samples) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(description='Prepare mixer training data')
    
    # 数据参数
    parser.add_argument('--json_paths', type=str, nargs='+', required=True,
                       help='JSON文件路径列表')
    parser.add_argument('--kimi_model_path', type=str, 
                       default="moonshotai/Kimi-Audio-7B",
                       help='Kimi模型路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    
    # 其他参数
    parser.add_argument('--device', type=str, default="cuda",
                       help='设备')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小（当前仅支持1）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大处理样本数（None表示处理所有）')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式（只处理第一个样本）')
    parser.add_argument('--adaptor_checkpoint_path', type=str, default=None,
                       help='Adaptor checkpoint路径（可选，如果提供将计算目标hidden states用于训练mixer）')
    
    args = parser.parse_args()
    
    print("🚀 Starting mixer training data preparation")
    print(f"📋 Arguments: {vars(args)}")
    
    process_dataset(
        json_paths=args.json_paths,
        kimi_model_path=args.kimi_model_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        debug=args.debug,
        adaptor_checkpoint_path=args.adaptor_checkpoint_path,
        mixer_checkpoint_path=None  # 准备数据时不需要mixer
    )
    
    print("🎉 All done!")


if __name__ == "__main__":
    main()

