import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, model_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

        self.qkv_weights_list = nn.ModuleList()
        for _ in range(num_heads):
            q_proj = nn.Linear(embed_dim, model_dim)
            k_proj = nn.Linear(embed_dim, model_dim)
            v_proj = nn.Linear(embed_dim, model_dim)
            self.qkv_weights_list.append(nn.ModuleList([q_proj, k_proj, v_proj]))

        self.out_proj = nn.Linear(model_dim * num_heads, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        # Pair-wise Dot Product Similarity
        S = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(self.head_dim)

        # Mask values with -inf so softmax is negligible
        if mask is not None:
            S.masked_fill_(mask == torch.tensor(False), float("-inf"))
        # Softmax into Probability / Score Distribution
        S = F.softmax(S, dim=0)
        # Apply weighted scores to value matrix
        return torch.matmul(S, value)
    

    def forward(self, query, key, value, mask=None):
        # QKV weight matrices -> (seq_len, embed_dim) x (embed_dim, model_dim) = (seq_len, model_dim)
        out = torch.cat([self.attention(Q(query), K(key), V(value), mask) for Q, K ,V in self.qkv_weights_list], dim=-1)
        # Shape(out) -> (seq_len, (model_dim * num_heads))
        out = self.out_proj(out)
        return out


