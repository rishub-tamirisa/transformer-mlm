import torch.nn as nn
from .attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding

class EncoderModel(nn.Module):
    '''
    `EncoderModel` wraps the Encoder with the embedding input layer and output projection layer, for use in downstream tasks.
    Importantly, positional encoding is added to the embedding before sending to the encoder.
    '''
    def __init__(self, vocab_size, embed_dim, model_dim, n_layers, num_heads, dropout=0.1):
        super(EncoderModel, self).__init__()
        # embeddings are a table where each row corresponds to an input_id is a vector of size embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_en = PositionalEncoding(embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)

        # Uncomment these 2 lines to use PyTorch's TransformerEncoder
        # encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.encoder = Encoder(embed_dim=embed_dim, model_dim=model_dim, n_layers=n_layers, num_heads=num_heads, dropout=dropout)
        # output projection layer to map to vocab_size 
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, input_mask=None):
        # add positional encoding of iputs to the embeddings before sending to encoder
        embedding = self.embedding(input_ids)
        ''' 
        Embedding permutes are to make the embedding dimension the first dimension for positional encoding (just for compatability with PyTorch code in positional_encoding.py). This is still just embedding + positional encoding
        '''
        input_embedding = self.dropout(embedding + self.pos_en(embedding.permute(1, 0, 2)).permute(1, 0, 2))
        X = self.encoder(input_embedding, input_mask)
        return self.out_proj(X)
        

class Encoder(nn.Module):
    def __init__(self, embed_dim, model_dim, n_layers, num_heads, dropout=0.1):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(
            embed_dim=embed_dim, model_dim=model_dim, num_heads=num_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(self, input_embedding, input_mask=None):
        X = input_embedding
        for layer in self.encoder_layers:
            X = layer(X, input_mask)
        return X


class EncoderBlock(nn.Module):
    '''
    Encoder block layer as described in the paper.
    '''
    def __init__(self, embed_dim, model_dim, num_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()    

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = MultiHeadAttention(embed_dim=embed_dim,
                                                       model_dim=model_dim,
                                                       num_heads=num_heads,
                                                       dropout=dropout)
        self.feed_forward = FeedForward(embed_dim=embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input_embeddings, input_mask=None):
        # compute self attention
        X = input_embeddings
        X = self.layer_norm1(self.multi_head_attention(X, X, X, input_mask) + X) # add residual connection + layer_norm
        # compute feed-forward
        X = self.layer_norm2(self.feed_forward(X) + X) # add residual connection + layer_norm
        return X


class FeedForward(nn.Module):
    def __init__(self, embed_dim, width_fac=4, dropout=0.1):
        super(FeedForward, self).__init__()

        self.W_ff1 = nn.Linear(embed_dim, width_fac * embed_dim)
        self.W_ff2 = nn.Linear(embed_dim * width_fac, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # Simple Feedforward network that projects into a higher space (by width_fac) and back to embed_dim
        X = self.W_ff1(X)
        X = self.dropout(self.relu(X))
        return self.W_ff2(X)
