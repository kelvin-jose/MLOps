import models.config as config
from transformers import RobertaTokenizer


def test_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained(config.Model.NAME)
    output = tokenizer("Hello world!", return_tensors="pt")
    assert "input_ids" in output
