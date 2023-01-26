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

    # 80% of the time: Replace the word with the
    # [MASK] token, e.g., my dog is hairy →
    # my dog is [MASK]
    # • 10% of the time: Replace the word with a
    # random word, e.g., my dog is hairy → my
    # dog is apple
    # • 10% of the time: Keep the word unchanged, e.g., my dog is hairy → my dog
    # is hairy. The purpose of this is to bias the
    # representation towards the actual observed
    # word.

    rand = torch.rand(input_ids.shape[0])
    # BERT-style masking:
    # create decision array with shape [input_ids.shape[0], 1] containing values 0, 1, 2 where 0 appears 80% of the time, 1 appears 10% of the time, and 2 appears 10% of the time
    decision_arr = torch.zeros(input_ids.shape[0], 1)
    decision_arr[rand < 0.8] = 0
    decision_arr[(rand >= 0.8) & (rand < 0.9)] = 1
    decision_arr[rand >= 0.9] = 2


    for i in range(input_ids.shape[0]):
        mask = torch.rand(input_ids[i].shape) < mlm_probability
        mask = mask * (input_ids[i] != 101) * (input_ids[i] != 102) * (input_ids[i] != 0)
        # Follow the 80-10-10 rule
        if decision_arr[i] == 0:
            # 80% of the time: Replace the word with the [MASK] token
            input_ids[i][mask] = tokenizer.mask_token_id
            labels[i][~mask] = -100
        elif decision_arr[i] == 1:
            # 10% of the time: Replace the word with a random word
            random_words = torch.randint(len(tokenizer), input_ids[i][mask].shape)
            input_ids[i][mask] = random_words
            labels[i][~mask] = -100
        else:
            # 10% of the time: Keep the word unchanged
            pass
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
