import data.config as config
from datasets import Dataset

def tokenize(split, tokenizer):
    texts = split['text']
    labels = split['label']
    encodings = tokenizer(
        texts,
        truncation = config.Tokenizer.TRUNCATION, 
        padding = config.Tokenizer.PADDING, 
        max_length = config.Tokenizer.MAX_LENGTH
    )
    return Dataset.from_dict({
        'text': texts,
        'label': labels,
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
    })