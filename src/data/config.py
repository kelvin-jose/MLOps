class Dataset:
    DATASET_LOCATION = "dair-ai/emotion"
    CACHE_DIR = "../data/raw"

class Tokenizer:
    TRUNCATION = True
    PADDING = "max_length"
    MAX_LENGTH = 128
