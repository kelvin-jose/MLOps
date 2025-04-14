import models.config as config
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification

def download():
    tokenizer = RobertaTokenizer.from_pretrained(config.Model.NAME)
    model = RobertaForSequenceClassification.from_pretrained(config.Model.NAME, 
                                                            num_labels = config.Model.NUM_CLASSES)
    print(f"[INFO] Tokenizer and model files are now downloaded.")
    return tokenizer, model