import torch
from modules.encoder import EncoderModel



if __name__ == "__main__":
    # Test Encoder
    
    encoder = EncoderModel(vocab_size=100, embed_dim=512, model_dim=512, n_layers=6, num_heads=8)
    X = torch.randint(0, 100, (1, 10))
    out = encoder(X)
    # to see predicted ids:
    # torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=2)
   

    print(out.shape)
