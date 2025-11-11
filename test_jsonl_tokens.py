#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to read the first line of BEAT_v2_1110_tokens.jsonl
and output the shapes of audio tokens and motion tokens.
"""

import json
import numpy as np
from pathlib import Path

def test_jsonl_tokens(jsonl_path):
    """
    Read the first line of JSONL file and output token shapes.
    
    Args:
        jsonl_path: Path to the JSONL file
    """
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"❌ Error: JSONL file not found: {jsonl_path}")
        return
    
    print(f"📖 Reading JSONL file: {jsonl_path}")
    
    # Read the first line
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    
    if not first_line:
        print("❌ Error: JSONL file is empty")
        return
    
    # Parse JSON
    try:
        data = json.loads(first_line)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse JSON: {e}")
        return
    
    print(f"✅ Successfully parsed JSON")
    print(f"📋 Task type: {data.get('task_type', 'N/A')}")
    print(f"📋 Conversation length: {len(data.get('conversation', []))}")
    
    # Extract audio tokens
    conversation = data.get('conversation', [])
    audio_tokens = None
    motion_tokens = None
    
    # Find audio tokens in user message
    for msg in conversation:
        if msg.get('role') == 'user' and msg.get('message_type') == 'audio':
            audio_tokens = msg.get('audio_tokens', [])
            print(f"🎵 Audio file: {msg.get('content', 'N/A')}")
            break
    
    # Find motion tokens in assistant message
    for msg in conversation:
        if msg.get('role') == 'assistant' and msg.get('message_type') == 'audio_motion':
            motion_tokens = msg.get('motion_tokens', [])
            break
    
    if audio_tokens is None:
        print("❌ Error: Audio tokens not found")
        return
    
    if motion_tokens is None:
        print("❌ Error: Motion tokens not found")
        return
    
    # Convert to numpy arrays
    audio_tokens_array = np.array(audio_tokens, dtype=np.int64)
    motion_tokens_array = np.array(motion_tokens, dtype=np.int64)
    
    # Output results
    print("\n" + "="*60)
    print("📊 Token Statistics:")
    print("="*60)
    print(f"🎵 Audio Tokens:")
    print(f"   Shape: {audio_tokens_array.shape}")
    print(f"   Length: {len(audio_tokens)}")
    print(f"   Dtype: {audio_tokens_array.dtype}")
    print(f"   Min value: {audio_tokens_array.min()}")
    print(f"   Max value: {audio_tokens_array.max()}")
    print(f"   Sample (first 10): {audio_tokens[:10]}")
    print(f"   Sample (last 10): {audio_tokens[-10:]}")
    
    print(f"\n🎭 Motion Tokens:")
    print(f"   Shape: {motion_tokens_array.shape}")
    print(f"   Length: {len(motion_tokens)}")
    print(f"   Dtype: {motion_tokens_array.dtype}")
    print(f"   Min value: {motion_tokens_array.min()}")
    print(f"   Max value: {motion_tokens_array.max()}")
    print(f"   Sample (first 10): {motion_tokens[:10]}")
    print(f"   Sample (last 10): {motion_tokens[-10:]}")
    
    # Analyze motion tokens (body and hand tokens are interleaved)
    # Body tokens: [0, 511], Hand tokens: [512, 1023]
    body_tokens = [t for t in motion_tokens if 0 <= t < 512]
    hand_tokens = [t for t in motion_tokens if 512 <= t < 1024]
    
    print(f"\n🔍 Motion Token Analysis:")
    print(f"   Body tokens (0-511): {len(body_tokens)} tokens")
    print(f"   Hand tokens (512-1023): {len(hand_tokens)} tokens")
    if body_tokens:
        print(f"   Body token range: [{min(body_tokens)}, {max(body_tokens)}]")
    else:
        print(f"   Body token range: N/A")
    if hand_tokens:
        print(f"   Hand token range: [{min(hand_tokens)}, {max(hand_tokens)}]")
    else:
        print(f"   Hand token range: N/A")
    
    # Check if tokens are interleaved correctly
    print(f"\n🔗 Token Interleaving:")
    print(f"   Expected pattern: body[0], hand[0], body[1], hand[1], ...")
    if len(motion_tokens) >= 4:
        print(f"   First 4 tokens: {motion_tokens[:4]}")
        print(f"   Pattern check: body={motion_tokens[0] < 512}, hand={motion_tokens[1] >= 512}, body={motion_tokens[2] < 512}, hand={motion_tokens[3] >= 512}")
    
    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    import sys
    from HRI_mllm import DATA_ROOT
    
    # Default JSONL path
    jsonl_path = Path(DATA_ROOT) / "BEAT_v2_1110_tokens.jsonl"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        jsonl_path = Path(sys.argv[1])
    
    test_jsonl_tokens(jsonl_path)

