To run training in a separate script, the following format can be used.

```python 
from modules.encoder import EncoderModel
from preprocess.mlm_preprocess import get_dataset_example, mask_dataset_for_mlm
from torch.utils.data import TensorDataset, DataLoader
from train import train_mlm
import torch

# Load Data
dataset, tokenizer = get_dataset_example() # this is not needed as long as you know your vocab_size and your data adheres to the format
dataset = mask_dataset_for_mlm(data=dataset, vocab_size=tokenizer.vocab_size)

# Set Params
embed_dim = 512
model_dim = 512
n_layers = 6
num_heads = 8
encoder = EncoderModel(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, model_dim=model_dim, n_layers=n_layers, num_heads=num_heads)

# Prepare DataLoader
dataset = TensorDataset( dataset['input_ids'], dataset['labels'], dataset['attention_mask'] )
loader = DataLoader(dataset, batch_size = 32, shuffle=True)

# Start Training
train_mlm(epochs=4, 
    tokenizer=tokenizer, 
    model=encoder, 
    loader=loader, 
    optimizer=torch.optim.Adam(encoder.parameters(), lr=1e-4),
    wandb_log=False)
```
#### Dependencies
`transformers` : used for tokenizing input data <br>
`datasets` : used for retrieving datasets <br>
`torch` : used for model implementation <br>

