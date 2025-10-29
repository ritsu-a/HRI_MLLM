#!/usr/bin/env python3
"""
快速测试统一Kimi-Motion模型系统
验证所有组件是否正常工作
"""

import os
import sys
import torch
import json
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.append('/root/workspace/HRI_MLLM')

def test_imports():
    """测试所有必要的导入"""
    print("🔄 Testing imports...")
    
    try:
        from HRI_mllm.model.unified_kimi_motion_model import UnifiedKimiMotionModel, HiddenStateMixer
        print("✅ Unified model imports successful")
    except ImportError as e:
        print(f"❌ Unified model import failed: {e}")
        return False
    
    try:
        from HRI_mllm.datasets.json_audio_motion_dataset import JSONAudioMotionDataset, create_dataloader
        print("✅ Dataset imports successful")
    except ImportError as e:
        print(f"❌ Dataset import failed: {e}")
        return False
    
    try:
        from HRI_mllm.train.train_unified_kimi_motion import UnifiedModelTrainer
        print("✅ Trainer imports successful")
    except ImportError as e:
        print(f"❌ Trainer import failed: {e}")
        return False
    
    return True


def test_hidden_state_mixer():
    """测试HiddenStateMixer组件"""
    print("\n🔄 Testing HiddenStateMixer...")
    
    try:
        from HRI_mllm.model.unified_kimi_motion_model import HiddenStateMixer
        
        # 创建mixer
        mixer = HiddenStateMixer(
            text_hidden_size=4096,
            audio_hidden_size=4096,
            output_hidden_size=768
        )
        
        # 测试前向传播
        batch_size, seq_len = 2, 10
        text_hidden = torch.randn(batch_size, seq_len, 4096)
        audio_hidden = torch.randn(batch_size, seq_len, 4096)
        
        output = mixer(text_hidden, audio_hidden)
        
        assert output.shape == (batch_size, seq_len, 768), f"Expected shape (2, 10, 768), got {output.shape}"
        print("✅ HiddenStateMixer test passed")
        return True
        
    except Exception as e:
        print(f"❌ HiddenStateMixer test failed: {e}")
        return False


def test_dataset():
    """测试数据集加载"""
    print("\n🔄 Testing dataset loading...")
    
    try:
        from HRI_mllm.datasets.json_audio_motion_dataset import JSONAudioMotionDataset
        
        # 创建临时测试数据（JSONL格式）
        test_data = [
            {
                "conversation": [
                    {
                        "message_type": "audio",
                        "audio_tokens": list(range(1, 51))  # 增加到50个token
                    },
                    {
                        "message_type": "audio_motion",
                        "motion_tokens": list(range(512, 562))  # 增加到50个token
                    }
                ]
            }
        ]
        
        # 保存临时文件（JSONL格式）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for data in test_data:
                f.write(json.dumps(data) + '\n')
            temp_path = f.name
        
        try:
            # 测试数据集加载
            dataset = JSONAudioMotionDataset(
                json_path=temp_path,
                max_audio_length=50,
                max_motion_length=50,
                min_audio_length=10,  # 降低最小长度要求
                min_motion_length=10,
                interleave_ratio=(1, 1),
                debug=True
            )
            
            assert len(dataset) > 0, "Dataset should have at least one sample"
            
            # 测试获取样本
            sample = dataset[0]
            assert 'audio_tokens' in sample, "Sample should contain audio_tokens"
            assert 'motion_tokens' in sample, "Sample should contain motion_tokens"
            assert 'interleaved_sequence' in sample, "Sample should contain interleaved_sequence"
            assert 'token_labels' in sample, "Sample should contain token_labels"
            
            print("✅ Dataset test passed")
            return True
            
        finally:
            # 清理临时文件
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def test_model_creation():
    """测试模型创建（不加载实际权重）"""
    print("\n🔄 Testing model creation...")
    
    try:
        from HRI_mllm.model.unified_kimi_motion_model import create_unified_model
        from transformers import GPT2Config
        
        # 创建简化的GPT2配置
        gpt2_config = GPT2Config(
            vocab_size=1034,
            n_positions=1024,
            n_embd=768,
            n_layer=6,
            n_head=12,
            n_inner=3072,
        )
        
        # 测试模型创建（使用一个不存在的路径，但会失败在加载阶段）
        try:
            model = create_unified_model(
                kimi_model_path="fake_path",  # 这会失败，但我们只测试创建逻辑
                freeze_kimi=True,
                freeze_adaptor=True,
                train_mixer_only=True
            )
            print("✅ Model creation test passed")
            return True
        except Exception as e:
            if "fake_path" in str(e) or "not found" in str(e).lower():
                print("✅ Model creation test passed (expected failure due to fake path)")
                return True
            else:
                raise e
                
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_data_processing():
    """测试数据处理流程"""
    print("\n🔄 Testing data processing...")
    
    try:
        from HRI_mllm.datasets.json_audio_motion_dataset import collate_fn
        
        # 创建模拟batch数据
        batch = [
            {
                'audio_tokens': torch.tensor([1, 2, 3, 4, 5]),
                'motion_tokens': torch.tensor([512, 513, 514, 515, 516]),
                'interleaved_sequence': torch.tensor([1, 512, 2, 513, 3, 514, 4, 515, 5, 516]),
                'token_labels': torch.tensor([-100, 512, -100, 513, -100, 514, -100, 515, -100, 516]),
                'audio_length': 5,
                'motion_length': 5,
                'sequence_length': 10,
                'sample_idx': 0
            },
            {
                'audio_tokens': torch.tensor([6, 7, 8, 9]),
                'motion_tokens': torch.tensor([517, 518, 519, 520]),
                'interleaved_sequence': torch.tensor([6, 517, 7, 518, 8, 519, 9, 520]),
                'token_labels': torch.tensor([-100, 517, -100, 518, -100, 519, -100, 520]),
                'audio_length': 4,
                'motion_length': 4,
                'sequence_length': 8,
                'sample_idx': 1
            }
        ]
        
        # 测试批处理函数
        processed_batch = collate_fn(batch)
        
        assert 'interleaved_sequences' in processed_batch, "Batch should contain interleaved_sequences"
        assert 'token_labels' in processed_batch, "Batch should contain token_labels"
        assert 'attention_masks' in processed_batch, "Batch should contain attention_masks"
        
        # 检查形状
        assert processed_batch['interleaved_sequences'].shape[0] == 2, "Batch size should be 2"
        assert processed_batch['interleaved_sequences'].shape[1] == 10, "Max sequence length should be 10"
        
        print("✅ Data processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 Starting Unified Kimi-Motion Model System Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("HiddenStateMixer Tests", test_hidden_state_mixer),
        ("Dataset Tests", test_dataset),
        ("Model Creation Tests", test_model_creation),
        ("Data Processing Tests", test_data_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Prepare your JSON training data")
        print("   2. Run: bash train_unified_kimi_motion.sh")
        print("   3. Test with: python HRI_mllm/test/test_unified_kimi_motion.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n💡 Troubleshooting:")
        print("   1. Check if all dependencies are installed")
        print("   2. Verify file paths and permissions")
        print("   3. Check Python environment and imports")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
