# [WIP] Masked Language Modeling with Transformer Encoders

For newcomers to NLP, it can be daunting to sift through the HuggingFace / PyTorch boilerplate for an understanding of different NN implementations. Often times, important model functions are hidden behind clever abstractions used by the library API for flexibility, which comes at the cost of readability. Additionally, a lot of YouTube videos covering this implementation tend to simply re-read the paper without providing implementation intuition, or when it comes to downstream tasks like MLM, simply applying HuggingFace functions without showing the inner workings.

**Disclaimer**: The code in this repository is **not** intended for production use, although some elements/computation may serve as an inspiration. The implementation emphasizes readability w.r.t the original Transformer Paper, ["Attention is All You Need" (Vaswani, et al. 2017)](https://arxiv.org/abs/1706.03762), rather than efficiency. Primarily, this code is meant to demonstrate Masked Language Modeling for encoders, whereas commonly-used implementations in PyTorch or HuggingFace contain substantially more configurable parameters for API flexibility for use in other downstream tasks.

This repo is not an attempt to replicate SOTA, but to experiment/play with the model. Feel free to adjust the hyperparameters/model config to anything.

### Repository Structure

- [modules](https://github.com/rishub-tamirisa/language-model-impl/tree/main/modules) contains the implementation of important Transformer encoder submodules, i.e. `MultiHeadAttention`.

- [preprocess](https://github.com/rishub-tamirisa/language-model-impl/tree/main/preprocess) contains functions for preparing masked language model training. For a full understanding of the implementation, it would be important to read through [`mask_dataset_for_mlm`](https://github.com/rishub-tamirisa/language-model-impl/blob/da81c342021b53e8589bc60945bc40bc326e3b7d/preprocess/mlm_preprocess.py#L8).

### Reproducibility

`train.ipynb` provides a complete notebook for loading a dataset and training the model.

#### Example

To run the training in a separate script, the following format can be used.

```python 
from modules.encoder import EncoderModel
from preprocess.mlm_preprocess import get_dataset_example, mask_dataset_for_mlm
from torch.utils.data import TensorDataset, DataLoader
from train import train_mlm
import torch


input_ids, tokenizer = get_dataset_example()
mlm_input_ids, mlm_labels = mask_dataset_for_mlm(input_ids)

embed_dim = 512
model_dim = 512
n_layers = 6
num_heads = 8
encoder = EncoderModel(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, model_dim=model_dim, n_layers=n_layers, num_heads=num_heads)

dataset = TensorDataset( mlm_input_ids, mlm_labels )
loader = DataLoader(dataset, batch_size = 32, shuffle=True)

train_mlm(epochs=4, 
          tokenizer=tokenizer, 
          model=encoder, 
          loader=loader, 
          optimizer=torch.optim.Adam(encoder.parameters(), lr=1e-4))
```


