import torch 
from torch import nn
import einops
from einops import rearrange,einsum
import math

class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        #初始化嵌入矩阵 vocab_size d_model
        self.embedding_matrix=nn.Parameter(torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype))
        self.reset_parameters()
        
    def reset_parameters(self,):
        nn.init.trunc_normal_(self.embedding_matrix,std=1,a=-3.0,b=3.0)
            
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        
        return self.embedding_matrix[token_ids]
            
    