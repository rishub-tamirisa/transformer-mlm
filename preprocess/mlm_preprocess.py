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
                    input_ids[i][j] = input_ids[i][j]
            else:
                input_ids[i][j] = input_ids[i][j]
                labels[i][j] = -100
        # assert that every row of labels has at least one -100 in it
        assert torch.sum(labels[i] == -100) > 0

    return input_ids, labels



'''
Retrieves data from HuggingFace (not important)
'''
def get_dataset_example(dataset_name="wikitext", dataset_config_name="wikitext-2-raw-v1", max_seq_length=128):
    dataset = load_dataset(dataset_name, dataset_config_name)
    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    text_column_name = "text"
    
    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
        )

    lm_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column_name],
    )

    return torch.LongTensor(lm_datasets['train']['input_ids']), tokenizer
