from datasets import load_dataset
import torch
from transformers import AutoTokenizer

'''
Masks the dataset for MLM (important)
'''
def mask_dataset_for_mlm(dataset, tokenizer, mlm_probability=0.15):
    # create mask array with true values appearing `mlm_probability`` of the time
    # mask_arr = (rand < mlm_probability) * (dataset != 101) * (dataset != 102) * (dataset != 0)
    input_ids = dataset.clone()
    labels = dataset.clone()

    '''
    80% of the time: Replace the word with the
    [MASK] token, e.g., my dog is hairy →
    my dog is [MASK]
    • 10% of the time: Replace the word with a
    random word, e.g., my dog is hairy → my
    dog is apple
    • 10% of the time: Keep the word unchanged, e.g., my dog is hairy → my dog
    is hairy. The purpose of this is to bias the
    representation towards the actual observed
    word.

    BERT-style masking: (currently decides masking at sequence level but should be at token level)
    '''
    for i in range(input_ids.shape[0]):
        mask = torch.rand(input_ids[i].shape) < mlm_probability
        mask = mask * (input_ids[i] != 101) * (input_ids[i] != 102) * (input_ids[i] != 0)

        decision = torch.rand(input_ids[i].shape)
        for j in range (mask.shape[0]):
            if mask[j]:
                if decision[j] < 0.8:
                    input_ids[i][j] = tokenizer.mask_token_id
                elif decision[j] < 0.9:
                    input_ids[i][j] = torch.randint(0, tokenizer.vocab_size, (1,))[0]
                else:
                    pass
            else:
                input_ids[i][j] = input_ids[i][j]
                labels[i][j] = -100

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

    return torch.LongTensor(lm_datasets['train']['input_ids']), tokenizer
