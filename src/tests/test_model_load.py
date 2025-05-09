import models.config as config
from transformers import RobertaForSequenceClassification


def test_model_load():
    model = RobertaForSequenceClassification.from_pretrained(
        config.Model.NAME, num_labels=config.Model.NUM_CLASSES
    )
    assert model is not None
