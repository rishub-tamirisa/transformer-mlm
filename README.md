# BERT-Style Masked Language Modeling with Transformer Encoders

The purpose of this repository is to provide a easily-readable / beginner-friendly implementation of a Transformer Encoder and demonstrate Masked Language Modeling as described in the BERT paper in PyTorch.

For newcomers to NLP, it can be daunting to sift through the HuggingFace / PyTorch boilerplate for an understanding of different NN implementations. Often times, important model functions are hidden behind clever abstractions used by the library API for flexibility, which comes at the cost of readability. Additionally, a lot of YouTube videos covering this implementation tend to simply re-read the paper without providing implementation intuition, or when it comes to downstream tasks like MLM, simply applying HuggingFace functions without showing the inner workings.

Users can check the number of parameters of this implementation and `TransformerEncoder` from PyTorch with the same config and verify that they are the same.

### Repository Structure

- [modules](https://github.com/rishub-tamirisa/language-model-impl/tree/main/modules) contains the implementation of important Transformer encoder submodules, i.e. `MultiHeadAttention`.

- [preprocess](https://github.com/rishub-tamirisa/language-model-impl/tree/main/preprocess) contains all functions for preparing masked language model training. This code also includes a function for retrieving data from HuggingFace. Implementing dataset preparation tasks for NLP like tokenization would add a ton of extra code. For this reason, I refer to HuggingFace for dataset retrieval and tokenization. See notes below for using your own data. 

### Reproducibility


- The main important function is [`mask_dataset_for_mlm`](https://github.com/rishub-tamirisa/transformer-mlm/blob/main/preprocess/mlm_preprocess.py) 
- If you want to use your own data, the repository uses 101 and 102 for `[CLS]` and `[SEP]` tokens, and 103 and 0 for `[MASK]` and `[PAD]` tokens. Tokenizing your dataset following that schema should be all that is necessary for processing. 
- It's assumed that your data follows the format before passing to `mask_dataset_for_mlm`:
```python
{ 
  'input_ids': torch.LongTensor,
  'attention_mask': torch.LongTensor,
}
```

- `train.ipynb` provides a complete notebook for loading a dataset and training the model.

#### Example

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

