import argparse
import json
import os
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from kimia_infer.api.prompt_manager import KimiAPromptManager


def load_jsonl(file_path):
    """加载jsonl文件"""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def verify_entry(entry, prompt_manager, verbose=True):
    """验证单个jsonl条目"""
    print(f"\n{'='*80}")
    
    # 获取对话内容
    conversation = entry['conversation']
    task_type = entry['task_type']
    
    if verbose:
        print(f"Task Type: {task_type}")
        print(f"Conversation has {len(conversation)} messages")
    
    # 检查结构
    assert len(conversation) == 3, f"Expected 3 messages, got {len(conversation)}"
    
    # 消息1: 文本提示
    msg1 = conversation[0]
    assert msg1['role'] == 'user', f"Message 1 should be user, got {msg1['role']}"
    assert msg1['message_type'] == 'text', f"Message 1 should be text, got {msg1['message_type']}"
    if verbose:
        print(f"\n[User Text Prompt]")
        print(f"  Content: {msg1['content']}")
    
    # 消息2: 用户音频 (A音频)
    msg2 = conversation[1]
    assert msg2['role'] == 'user', f"Message 2 should be user, got {msg2['role']}"
    assert msg2['message_type'] == 'audio', f"Message 2 should be audio, got {msg2['message_type']}"
    
    user_audio_path = msg2['content']
    user_audio_tokens = msg2['audio_tokens']
    
    if verbose:
        print(f"\n[User Audio] (A音频)")
        print(f"  Path: {user_audio_path}")
        print(f"  Tokens count: {len(user_audio_tokens)}")
        print(f"  Token range: [{min(user_audio_tokens)}, {max(user_audio_tokens)}]")
        print(f"  First 10 tokens: {user_audio_tokens[:10]}")
    
    # 验证用户音频tokens
    if os.path.exists(user_audio_path):
        print(f"  ✅ Audio file exists")
        
        # 重新tokenize音频，验证tokens是否一致
        try:
            retokenized = np.array(prompt_manager._tokenize_audio(user_audio_path))
            if np.array_equal(retokenized, np.array(user_audio_tokens)):
                print(f"  ✅ Tokens match original audio")
            else:
                print(f"  ❌ Tokens DO NOT match!")
                print(f"     Expected {len(retokenized)} tokens, got {len(user_audio_tokens)}")
                if len(retokenized) == len(user_audio_tokens):
                    diff_count = np.sum(retokenized != np.array(user_audio_tokens))
                    print(f"     {diff_count} tokens are different")
        except Exception as e:
            print(f"  ⚠️  Could not re-tokenize audio: {e}")
    else:
        print(f"  ❌ Audio file NOT found")
    
    # 消息3: Assistant响应 (B音频 + B动作)
    msg3 = conversation[2]
    assert msg3['role'] == 'assistant', f"Message 3 should be assistant, got {msg3['role']}"
    assert msg3['message_type'] == 'audio_motion', f"Message 3 should be audio_motion, got {msg3['message_type']}"
    
    assistant_audio_path = msg3['content']
    assistant_audio_tokens = msg3['audio_tokens']
    motion_tokens = msg3['motion_tokens']
    
    if verbose:
        print(f"\n[Assistant Response] (B音频 + B动作)")
        print(f"  Audio Path: {assistant_audio_path}")
        print(f"  Audio tokens count: {len(assistant_audio_tokens)}")
        print(f"  Audio token range: [{min(assistant_audio_tokens)}, {max(assistant_audio_tokens)}]")
        print(f"  First 10 audio tokens: {assistant_audio_tokens[:10]}")
        print(f"  Motion tokens count: {len(motion_tokens)}")
        print(f"  Motion token range: [{min(motion_tokens)}, {max(motion_tokens)}]")
        print(f"  First 20 motion tokens: {motion_tokens[:20]}")
    
    # 验证assistant音频tokens
    if os.path.exists(assistant_audio_path):
        print(f"  ✅ Audio file exists")
        
        # 重新tokenize音频，验证tokens是否一致
        try:
            retokenized = np.array(prompt_manager._tokenize_audio(assistant_audio_path))
            if np.array_equal(retokenized, np.array(assistant_audio_tokens)):
                print(f"  ✅ Audio tokens match original audio")
            else:
                print(f"  ❌ Audio tokens DO NOT match!")
                print(f"     Expected {len(retokenized)} tokens, got {len(assistant_audio_tokens)}")
                if len(retokenized) == len(assistant_audio_tokens):
                    diff_count = np.sum(retokenized != np.array(assistant_audio_tokens))
                    print(f"     {diff_count} tokens are different")
        except Exception as e:
            print(f"  ⚠️  Could not re-tokenize audio: {e}")
    else:
        print(f"  ❌ Audio file NOT found")
    
    # 验证motion tokens格式
    # Motion tokens应该是交替的 body 和 hand tokens
    # body tokens: [0, 511], hand tokens: [512, 1023]
    body_tokens = [motion_tokens[i] for i in range(0, len(motion_tokens), 2)]
    hand_tokens = [motion_tokens[i] for i in range(1, len(motion_tokens), 2)]
    
    body_in_range = all(0 <= t < 512 for t in body_tokens)
    hand_in_range = all(512 <= t < 1024 for t in hand_tokens)
    
    print(f"\n[Motion Tokens Analysis]")
    print(f"  Total motion tokens: {len(motion_tokens)}")
    print(f"  Body tokens (even indices): {len(body_tokens)}")
    print(f"    Range: [{min(body_tokens)}, {max(body_tokens)}]")
    print(f"    All in [0, 511]: {'✅' if body_in_range else '❌'}")
    print(f"  Hand tokens (odd indices): {len(hand_tokens)}")
    print(f"    Range: [{min(hand_tokens)}, {max(hand_tokens)}]")
    print(f"    All in [512, 1023]: {'✅' if hand_in_range else '❌'}")
    
    # 检查文件名对应关系
    user_audio_basename = os.path.basename(user_audio_path)
    assistant_audio_basename = os.path.basename(assistant_audio_path)
    
    # 从 user_audio_basename 提取编号，例如 "717_A.wav" -> "717"
    if '_A.wav' in user_audio_basename:
        user_id = user_audio_basename.replace('_A.wav', '')
        expected_assistant = f"{user_id}_B.wav"
        
        print(f"\n[File Correspondence]")
        print(f"  User audio: {user_audio_basename}")
        print(f"  Assistant audio: {assistant_audio_basename}")
        print(f"  Expected assistant: {expected_assistant}")
        
        if assistant_audio_basename == expected_assistant:
            print(f"  ✅ File names correspond correctly")
        else:
            print(f"  ❌ File names DO NOT correspond!")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify HAND_RING jsonl file')
    parser.add_argument("--jsonl_file", type=str, 
                       default="/root/workspace/HRI_MLLM/data/HAND_RING_1111_tokens.jsonl",
                       help="Path to jsonl file")
    parser.add_argument("--model_name_or_path", type=str, 
                       default="moonshotai/Kimi-Audio-7B",
                       help="Kimi model path")
    parser.add_argument("--max_entries", type=int, default=None,
                       help="Maximum number of entries to verify (default: all)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed information")
    
    args = parser.parse_args()
    
    # 加载音频tokenizer
    print(f"Loading audio tokenizer from: {args.model_name_or_path}")
    if os.path.exists(args.model_name_or_path):
        cache_path = args.model_name_or_path
    else:
        cache_path = snapshot_download(args.model_name_or_path)
    
    model_config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    prompt_manager = KimiAPromptManager(
        model_path=cache_path, 
        kimia_token_offset=model_config.kimia_token_offset, 
        kimia_text_audiodelaytokens=model_config.kimia_mimo_audiodelaytokens
    )
    print(f"✅ Audio tokenizer loaded successfully!")
    
    # 加载jsonl文件
    print(f"\nLoading jsonl file: {args.jsonl_file}")
    entries = load_jsonl(args.jsonl_file)
    print(f"Loaded {len(entries)} entries")
    
    # 验证条目
    max_entries = args.max_entries if args.max_entries else len(entries)
    success_count = 0
    
    for i, entry in enumerate(entries[:max_entries]):
        print(f"\n{'#'*80}")
        print(f"# Verifying Entry {i+1}/{min(max_entries, len(entries))}")
        print(f"{'#'*80}")
        
        try:
            verify_entry(entry, prompt_manager, verbose=args.verbose)
            success_count += 1
        except Exception as e:
            print(f"\n❌ Error verifying entry {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Verification Summary")
    print(f"{'='*80}")
    print(f"Total entries: {len(entries)}")
    print(f"Verified: {min(max_entries, len(entries))}")
    print(f"Success: {success_count}")
    print(f"Failed: {min(max_entries, len(entries)) - success_count}")
    print(f"{'='*80}")

