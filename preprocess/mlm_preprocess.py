from datasets import load_dataset
import torch
from transformers import AutoTokenizer

'''
Masks the dataset for MLM (important)
'''
def mask_dataset_for_mlm(dataset, mlm_probability=0.15):
    rand = torch.rand(dataset.shape)
    # create mask array with true values appearing `mlm_probability`` of the time
    mask_arr = (rand < mlm_probability) * (dataset != 101) * \
            (dataset != 102) * (dataset != 0)
    
    # create mask indices for input and label selection for each row 
    input_selection = []
    label_selection = []
    for i in range(dataset.shape[0]):
        input_selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
        label_selection.append(torch.flatten((mask_arr[i] == 0).nonzero()).tolist())

    # set mask positions in input to 103 for [MASK] token
    # set all positions EXCEPT mask to -100 for ignored loss in torch.nn.CrossEntropyLoss
    input_ids = dataset.clone()
    labels = dataset.clone()
    for i in range(input_ids.shape[0]):
        input_ids[i, input_selection[i]] = 103
        labels[i, label_selection[i]] = -100
    return input_ids, labels

'''
Retrieves data from HuggingFace (not important)
'''
def get_dataset_example():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    chunk_size = 128
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    return torch.Tensor(lm_datasets['train']['input_ids']), tokenizer
