import os
import torch
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification

import models.config as config

MODEL_ID = os.environ.get("MODEL_ID", "checkpoint-1")
MODEL_PATH = f"results/{MODEL_ID}"

tokenizer = RobertaTokenizer.from_pretrained(config.Model.NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)


def predict(text):
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return {"label": predicted_class, "logits": logits.tolist()}
