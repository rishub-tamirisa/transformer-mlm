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
        # If MultiHeadAttention, QKV projections are split across 2nd dim -> embed_dim x model_dim -> embed_dim x (model_dim / num_heads)
        for _ in range(num_heads):
            q_proj = nn.Linear(embed_dim, self.head_dim)
            k_proj = nn.Linear(embed_dim, self.head_dim)
            v_proj = nn.Linear(embed_dim, self.head_dim)
            self.qkv_weights_list.append(nn.ModuleList([q_proj, k_proj, v_proj]))
        
        # Output Projection, project to embed_dim from model_dim (usually model_dim = embed_dim)
        self.out_proj = nn.Linear(model_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        # Pair-wise Dot Product Similarity
        S = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(self.head_dim)

        # Mask values with -inf so softmax is negligible
        if mask is not None:
            S.masked_fill_(mask == torch.tensor(False), float("-inf"))
        # Softmax into Probability / Score Distribution
        S = F.softmax(S, dim=-1)
        # Apply weighted scores to value matrix
        return torch.matmul(S, value)
    

    def forward(self, query, key, value, mask=None):
        '''
        Typically, the concatenation operation over a list is not used because rather than creating q_proj, k_proj, v_proj
        with dim (embed_dim, self.head_dim), we create them as larger linear layers with dim (embed_dim, model_dim)
        and instead reshape the output with reshape() or view() to (num_heads, seq_len, self.head_dim). This works because 
        num_heads * self.head_dim = model_dim. This is more efficient than concatenating the output of the QKV linear layers
        because attention can be computed in parallel. Here, we use the concatenation operation to directly replicate the MultiHeadAttention
        definition in Vaswani et al. (2017).
        '''
        # QKV weight matrices -> (seq_len, embed_dim) x (embed_dim, model_dim) = (seq_len, model_dim)
        out = torch.cat([self.attention(Q(query), K(key), V(value), mask) for Q, K ,V in self.qkv_weights_list], dim=-1)
        # Shape(out) -> (seq_len, (head_dim * num_heads))
        out = self.out_proj(out)
        return self.dropout(out)


if __name__ == "__main__":
    # Test MultiHeadAttention
    torch.manual_seed(0)
    embed_dim = 512
    model_dim = 512
    num_heads = 4
    seq_len = 10
    test = torch.rand((1, seq_len, embed_dim))
    # mask = torch.rand((seq_len, seq_len)) > 0.5

    mha = MultiHeadAttention(embed_dim, model_dim, num_heads)
    mha_torch = nn.MultiheadAttention(embed_dim, num_heads)
    out = mha(test, test, test)
    out_torch = mha_torch(test, test, test)
    assert out.shape == out_torch[0].shape
