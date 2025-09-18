from transformers import AutoModel, AutoConfig
import torch

from HRI_mllm.model.kimi_motion import MoonshotKimiaForCausalLM, MoonshotKimiaMotionForCausalLM


# 加载模型A的配置
config = AutoConfig.from_pretrained("moonshotai/Kimi-Audio-7B-Instruct", trust_remote_code=True)
config.motion_vocab_size = 512 + 2

# 初始化模型B
model_b = MoonshotKimiaMotionForCausalLM(config)

# 加载模型A的权重
model_a = MoonshotKimiaForCausalLM.from_pretrained("moonshotai/Kimi-Audio-7B-Instruct",trust_remote_code=True)

# 部分加载权重
missing_keys, unexpected_keys = model_b.load_state_dict(
    model_a.state_dict(), 
    strict=False
)

print(f"加载完成。缺失键: {len(missing_keys)}, 意外键: {len(unexpected_keys)}")

import ipdb;ipdb.set_trace()

# # 对新增参数进行初始化
# for name, param in model_b.named_parameters():
#     if any(missing_key in name for missing_key in missing_keys):
#         if 'weight' in name:
#             torch.nn.init.xavier_normal_(param)
#         elif 'bias' in name:
#             torch.nn.init.zeros_(param)

# 保存模型B
model_b.save_pretrained("output/motion_model/Kimi-Audio-Motion-7B")
print("模型B保存完成")



