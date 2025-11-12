import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math


class Linear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.W=nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self):
        std=math.sqrt(2/(self.in_features+self.out_features))
        nn.init.trunc_normal_(self.W,mean=0.0,std=std,a=-30.0,b=30.0)
        
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x_W=einsum(x,self.W,"... in_features, out_features in_features->... out_features")
        return x_W