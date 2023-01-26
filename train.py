from modules.encoder import EncoderModel
from preprocess.mlm_preprocess import get_dataset_example, mask_dataset_for_mlm
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm

'''
Same content as train.ipynb
'''

def train_mlm(epochs, model, tokenizer, loader, optimizer=torch.optim.Adam, device=torch.device('cpu')):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    model.to(device)
    with tqdm(total=epochs) as pbar:
        for _ in range(epochs):
            cur_batch = 0
            total_batches = len(loader) 
            for batch in loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device, dtype=torch.int64)
                labels = labels.to(device, dtype=torch.int64)
                optimizer.zero_grad()
                output = model(input_ids)
                loss = criterion(output.view(-1, tokenizer.vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()
                cur_batch += 1
                pbar.set_postfix(**{"batch: ": f"{cur_batch} / {total_batches}", "loss:": loss.item()})
        checkpoint = {'vocab_size': tokenizer.vocab_size,
                      'embed_dim': embed_dim,
                      'model_dim': model_dim,
                      'n_layers': n_layers,
                      'num_heads': num_heads,
                      'state_dict': model.state_dict()}
        torch.save(checkpoint, 'model_checkpoints/checkpoint.pth')

def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = EncoderModel(vocab_size=checkpoint['vocab_size'], 
                         embed_dim=checkpoint['embed_dim'], 
                         model_dim=checkpoint['model_dim'], 
                         n_layers=checkpoint['n_layers'], 
                         num_heads=checkpoint['num_heads'])
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == "__main__":
    
    # Look in preprocess/mlm_preprocess.py for the code that retrieves the dataset
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