import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math
from torch import Tensor
from Linear import Linear
from RMSNorm import RMSNorm
from CausalMultiHeadAttention import CausalMultiHeadAttention
from Embedding import Embedding
from SwiGLU import SwiGLU
from Block import TransformerBlock
from utils import softmax, scaled_dot_product_attention

class LM(nn.Module):
    def __init__(self,
        vocab_size:int,
        context_length:int,
        d_model:int,
        num_layers:int,
        num_heads:int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.model_args = {
            'vocab_size': vocab_size,
            'context_length': context_length,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'rope_theta': rope_theta,
        }
        self.context_length = context_length
        
        self.token_embedding=Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        self.transformer_blocks=nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )
        
        self.final_norm=RMSNorm(d_model, device=device, dtype=dtype)
        
        self.pred=Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self,token_ids:Tensor)->Tensor:
        batch_size,seq_len=token_ids.shape
        device=token_ids.device
        
        token_positions=torch.arange(seq_len,device=device)
        
        x=self.token_embedding(token_ids)
        
        for block in self.transformer_blocks:
            x=block(x,token_positions)
        
        x=self.final_norm(x)
        
        return self.pred(x)