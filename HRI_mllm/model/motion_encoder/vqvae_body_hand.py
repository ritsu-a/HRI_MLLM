
import torch
import torch.nn as nn
import numpy as np
import copy


from .vqvae import VQVae

class VQVaeBodyHand(nn.Module):
    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
        assert nfeats == 491, f"nfeats should be 491 for body+hand, but got {nfeats}"
        self.body_vae = VQVae(nfeats=263,
                              quantizer=quantizer,
                              code_num=code_num,
                              code_dim=code_dim,
                              output_emb_width=output_emb_width,
                              down_t=down_t,
                              stride_t=stride_t,
                              width=width,
                              depth=depth,
                              dilation_growth_rate=dilation_growth_rate,
                              norm=norm,
                              activation=activation)
        
        self.hand_vae = VQVae(nfeats=228,
                              quantizer=quantizer,
                              code_num=code_num,    
                                code_dim=code_dim,
                                output_emb_width=output_emb_width,
                                down_t=down_t,
                                stride_t=stride_t,
                                width=width,
                                depth=depth,
                                dilation_growth_rate=dilation_growth_rate,
                                norm=norm,
                                activation=activation)
        

    def encode(self, x):
        # x: (B, T, 491)
        body_x = x[:, :, :263]  # (B, T, 263)
        hand_x = x[:, :, 263:]  # (B, T, 228)
        body_code, _ = self.body_vae.encode(body_x)  # body_code: (B, T', 263)
        hand_code, _ = self.hand_vae.encode(hand_x)  # hand_code: (B, T', 228)
        return (body_code, hand_code), None
    
    def decode(self, codes):
        body_code, hand_code = codes
        body_decoded = self.body_vae.decode(body_code)  # (B, T, 263)
        hand_decoded = self.hand_vae.decode(hand_code)  # (B, T, 228)
        x_decoded = torch.cat([body_decoded, hand_decoded], dim=-1)  # (B, T, 491)
        return x_decoded
    
    def forward(self, x):
        body_x = x[:, :, :263]  # (B, T, 263)
        hand_x = x[:, :, 263:]  # (B, T, 228)

        body_recon, body_quant_loss, body_perplexity = self.body_vae(body_x)  # (B, T, 263)
        hand_recon, hand_quant_loss, hand_perplexity = self.hand_vae(hand_x)  # (B, T, 228)
        x_recon = torch.cat([body_recon, hand_recon], dim=-1)  # (B, T, 491)
        total_quant_loss = body_quant_loss + hand_quant_loss
        return x_recon, total_quant_loss, (body_perplexity + hand_perplexity) / 2