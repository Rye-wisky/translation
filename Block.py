import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math
from Linear import Linear
from RMSNorm import RMSNorm
from CausalMultiHeadAttention import CausalMultiHeadAttention
from Embedding import Embedding
from SwiGLU import SwiGLU
from utils import softmax, scaled_dot_product_attention

class TransformerBlock(nn.Module):
    def __init__(self,
        d_model:int, 
        num_heads:int,
        d_ff:int, #FFN隐藏层维度
        max_seq_len:int,
        theta:float=10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.max_seq_len=max_seq_len
        self.theta=theta
        #预归一化
        self.norm1=RMSNorm(d_model)
        #注意力层
        self.attentionlayer=CausalMultiHeadAttention(d_model,num_heads,max_seq_len,theta,device,dtype)
        #归一化
        self.norm2=RMSNorm(d_model)
        #前馈层
        self.ffn=SwiGLU(d_model,d_ff,device=device, dtype=dtype)
    
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        x_norm1=self.norm1(x)
        att_output=self.attentionlayer(x_norm1,token_positions)
        #残差连接
        x=x+att_output
        
        ffn_output=self.ffn(self.norm2(x))
        #残差连接
        x=x+ffn_output
        return x