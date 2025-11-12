import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math
from Linear import Linear

class SwiGLU(nn.Module):
    # (*** 变化点: 添加 device 和 dtype ***)
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        
        # (*** 变化点: 处理 d_ff=None 的情况 ***)
        if d_ff is None:
            self.d_ff = int(round(((8 / 3 * d_model) / 64)) * 64)
        else:
            self.d_ff = d_ff
            
        # (*** 变化点: 将 device 和 dtype 传递给 Linear ***)
        self.W1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.V = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        
        
    def forward(self,x:torch.Tensor):
        x1=self.W1(x)
        x2=self.V(x)
        x_gate=x1*torch.sigmoid(x1)
        #逐元素相乘
        x3=x_gate*x2
        output=self.W2(x3)
        
        return output