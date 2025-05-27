import torch 
from torch import nn
import math

class MySelfAttention(nn.Module):
    def __init__(self, dim_embedding, dim_qk, dim_v):
        super(MySelfAttention, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self._norm_fact = 1 / math.sqrt(dim_embedding)
        

        self.linear_q = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_k = nn.Linear(dim_embedding, dim_qk, bias=False)
        self.linear_v = nn.Linear(dim_embedding, dim_v, bias=False)
    def forward(self, x):
        # x: batch, sequence_length, dim_embedding
        # 根据文本获得相应的维度
        
        batch, n, dim_embedding = x.shape
        assert dim_embedding == self.dim_embedding
        
        # nn.Linear 自动生成 W，b
        # 1. Q = X * W_Q + b
        # 2. K = X * W_K + b
        # 3. V = X * W_V + b 
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Softmax( Q * K^T )/ sqrt(dim_embedding)
        score = nn.functional.softmax(torch.bmm(q, k.T)) * self._norm_fact 

        # Softmax( Q * K^T )/ sqrt(dim_embedding)
        att = torch.bmm(score, v)
        return att
        