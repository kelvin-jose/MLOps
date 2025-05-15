from data import download as data_download
from data import preprocess
from models import download as model_download
from models import utils
from models.config import Train
from models.config import Track

import mlflow
from transformers import Trainer
from transformers import TrainingArguments
from datasets import DatasetDict

mlflow.set_tracking_uri(Track.URI)
mlflow.set_experiment(Track.EXPERIMENT_ID)

dataset = data_download.download()
tokenizer, model = model_download.download()

training_args = TrainingArguments(
    output_dir=Train.OUTPUT_DIR,
    eval_strategy=Train.EVAL_STRATEGY,
    save_strategy=Train.SAVE_STRATEGY,
    learning_rate=Train.LEARNING_RATE,
    per_device_train_batch_size=Train.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=Train.EVAL_BATCH_SIZE,
    num_train_epochs=Train.TRAIN_EPOCHS,
    weight_decay=Train.WEIGHT_DECAY,
    logging_steps=Train.LOGGING_STEPS,
    logging_strategy=Train.LOGGING_STRATEGY,
    logging_first_step=True,
    report_to="tensorboard" if Train.TENSORBOARD else "none",
    disable_tqdm=False,
    bf16=True if Train.USE_BF16 else False,
    logging_dir=Train.LOGGING_DIR,
    no_cuda=False if Train.CUDA else True,
)

tokenized_train = preprocess.tokenize(dataset["train"], tokenizer)
tokenized_val = preprocess.tokenize(dataset["validation"], tokenizer)
tokenized_test = preprocess.tokenize(dataset["test"], tokenizer)

tokenized_dataset = DatasetDict({"train": tokenized_train,
                                 "validation": tokenized_val,
                                 "test": tokenized_test})

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(160)),
    eval_dataset=tokenized_dataset["test"].select(range(4)),
    compute_metrics=utils.compute_metrics,
)

with mlflow.start_run():
    mlflow.log_param("model_checkpoint", Train.OUTPUT_DIR)
    mlflow.log_param("learning_rate", Train.LEARNING_RATE)
    mlflow.log_param("train_batch_size", Train.TRAIN_BATCH_SIZE)
    mlflow.log_param("evaluation_batch_size", Train.EVAL_BATCH_SIZE)
    mlflow.log_param("epochs", Train.TRAIN_EPOCHS)
    mlflow.log_param("weight_decay", Train.WEIGHT_DECAY)
    trainer.train()

    eval_metrics = trainer.evaluate()
    mlflow.log_metrics(eval_metrics)

    mlflow.log_artifacts(Train.OUTPUT_DIR, artifact_path="model")

    if Track.REGISTER:
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, Track.MODEL_NAME)


