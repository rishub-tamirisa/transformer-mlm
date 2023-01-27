from datasets import load_dataset
import torch
from transformers import AutoTokenizer


def _generate_bertstyle_mask(input_ids, mlm_probability=0.15):
    '''
    Helper function for generating a mask that contains true values 15% of the time within the range of non-zero values in input_ids
    '''
    zero_idx = torch.nonzero(input_ids == 0) # get index of first 0 in input_ids if it exists, if not set to length of input_ids
    if zero_idx.shape[0] == 0:
        zero_idx = input_ids.shape[0]
    else:
        zero_idx = zero_idx[0][0]

    num_values = int(zero_idx * mlm_probability)
    # generate num_values number of integers between 0 and zero_idx
    mask_indices = torch.randint(0, zero_idx, (num_values,))
    nonpad_ids = input_ids[:zero_idx] 
    # create bool array called mask with true values appearing at indices in mask_indices
    mask = torch.zeros(nonpad_ids.shape, dtype=torch.bool)
    mask[mask_indices] = True
    # retry if mask is all 0s, this is unlikely to happen but just in case
    mask = mask * (nonpad_ids != 101) * (nonpad_ids != 102) * (nonpad_ids != 0)
    return mask, nonpad_ids

'''
Masks the dataset for MLM (important)
'''
def mask_dataset_for_mlm(data, tokenizer, mlm_probability=0.15):
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
    # create mask array with true values appearing `mlm_probability`` of the time
    # mask_arr = (rand < mlm_probability) * (dataset != 101) * (dataset != 102) * (dataset != 0)
    dataset = data['input_ids']
    input_ids = dataset.clone()
    labels = dataset.clone()

    for i in range(input_ids.shape[0]):
        #  generate random values only where input_ids is not 0 
        # get range of non-zero values in input_ids
        # get index of first 0 in input_ids if it exists, if not set to length of input_ids

        mask, nonpad_ids = _generate_bertstyle_mask(input_ids[i], mlm_probability)

        decision = torch.rand(nonpad_ids.shape)
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
        # make rest of labels -100
        labels[i][nonpad_ids.shape[0]:] = -100
        # assert that every row of labels has at least one -100 in it
        assert torch.sum(labels[i] == -100) > 0

    data['input_ids'] = input_ids
    data['labels'] = labels
    return data



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

    return {'input_ids': torch.LongTensor(lm_datasets['train']['input_ids']), 
            'attention_mask': torch.LongTensor(lm_datasets['train']['attention_mask'])}, tokenizer
