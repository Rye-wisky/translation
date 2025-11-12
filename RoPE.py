import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        
        if d_k%2!=0:
            raise ValueError(f"d_k 必须是偶数，但传入的是 {d_k}")
        #i是token在序列中的位置，k是pair的索引，根据RoPE，对位置为i的token中的词嵌入向量的每个对旋转不同频率的角度
        k_indicies=torch.arange(1,d_k//2+1,device=device)
        #指数
        exponent=(2*k_indicies-1)/d_k
        #计算inv_freq，也就是旋转的频率
        inv_freq=1.0/(theta**exponent)
        #i的索引
        pos_indices=torch.arange(max_seq_len,device=device)
        #计算所有可能的角度
        angles=einsum(pos_indices,inv_freq,"i,freq->i freq")
        #k扩展两倍方便做内积
        angles_expanded=angles.repeat_interleave(2,dim=-1)
        
        #将cos、sin值存起来
        self.register_buffer('sin_cached', angles_expanded.sin(), persistent=False)
        self.register_buffer('cos_cached', angles_expanded.cos(), persistent=False)
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        #x是张量，token_pos和d_k
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        return x_rotated