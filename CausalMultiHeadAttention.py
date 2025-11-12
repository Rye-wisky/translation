import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math
from Linear import Linear
from RoPE import RoPE
from utils import softmax, scaled_dot_product_attention

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model:int,num_heads:int,max_seq_ken:int,theta:float=10000.0,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.max_seq_len=max_seq_ken
        self.theta=theta
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除。")
        self.d_k=d_model//num_heads
        self.w_qkv=Linear(d_model,3*d_model,device,dtype)
        self.w_o=Linear(d_model,d_model,device,dtype)
        self.rope=RoPE(theta,self.d_k,max_seq_ken)
        
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        batch_size,seq_len,_=x.shape
        
        qkv=self.w_qkv(x)
        q,k,v=qkv.chunk(3,dim=-1)
        q=rearrange(q,"batch sequence (head d_k)->batch head sequence d_k",head=self.num_heads)
        k=rearrange(k,"batch sequence (head d_k)->batch head sequence d_k",head=self.num_heads)
        v=rearrange(v,"batch sequence (head d_k)->batch head sequence d_k",head=self.num_heads)
        
        q=self.rope(q,token_positions)
        k=self.rope(k,token_positions)
        
        causal_mask= ~torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device), diagonal=1)
        
        output_heads=scaled_dot_product_attention(q,k,v,mask=causal_mask)
        output=rearrange(output_heads,"batch head sequence d_k->batch sequence (head d_k)")
        
        return self.w_o(output)