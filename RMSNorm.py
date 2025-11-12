import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.g=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #类型转换防止计算均方根时发生溢出
        intype=x.dtype
        x=x.to(torch.float32)
        #计算均方根的倒数
        rrms=torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        normalized_x=x*rrms*self.g
        
        return normalized_x.to(intype) 
        