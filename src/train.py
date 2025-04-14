from data import download as data_download
from data import preprocess
from models import download as model_download
from models import utils
from models.config import Train

from transformers import Trainer
from transformers import TrainingArguments
from datasets import DatasetDict


dataset = data_download.download()
tokenizer, model = model_download.download()

training_args = TrainingArguments(
    output_dir = Train.OUTPUT_DIR,
    eval_strategy = Train.EVAL_STRATEGY,
    save_strategy = Train.SAVE_STRATEGY,
    learning_rate = Train.LEARNING_RATE,
    per_device_train_batch_size = Train.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size = Train.EVAL_BATCH_SIZE,
    num_train_epochs = Train.TRAIN_EPOCHS,
    weight_decay = Train.WEIGHT_DECAY,
    logging_steps = Train.LOGGING_STEPS,
    logging_strategy = Train.LOGGING_STRATEGY,
    logging_first_step = True,
    report_to = "tensorboard" if Train.TENSORBOARD else "none",       
    disable_tqdm = False, 
    bf16 = True if Train.USE_BF16 else False, 
    logging_dir = Train.LOGGING_DIR,
    no_cuda = False if Train.CUDA else True
)

tokenized_train = preprocess.tokenize(dataset['train'], tokenizer)
tokenized_val = preprocess.tokenize(dataset['validation'], tokenizer)
tokenized_test = preprocess.tokenize(dataset['test'], tokenizer)

tokenized_dataset = DatasetDict({
    'train': tokenized_train,
    'validation': tokenized_val,
    'test': tokenized_test
})

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset["train"].select(range(4)),
    eval_dataset = tokenized_dataset["test"].select(range(4)),
    compute_metrics = utils.compute_metrics,
)

trainer.train()

trainer.evaluate()
