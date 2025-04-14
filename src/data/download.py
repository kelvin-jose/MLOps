import data.config as config
from datasets import load_dataset

def download():
    dataset = load_dataset(config.Dataset.DATASET_LOCATION, cache_dir = config.Dataset.CACHE_DIR)
    print(f"[INFO] {config.Dataset.DATASET_LOCATION} is now available at {config.Dataset.CACHE_DIR}.")
    return dataset
