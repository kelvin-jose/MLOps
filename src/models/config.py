class Train:
    OUTPUT_DIR = "./results"
    EVAL_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LEARNING_RATE = 2e-5
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    TRAIN_EPOCHS = 1
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 5
    LOGGING_STRATEGY = "steps"
    LOGGING_DIR = "logs"
    TENSORBOARD = True
    USE_BF16 = True
    CUDA = True


class Model:
    NAME = "distilroberta-base"
    NUM_CLASSES = 6


class Track:
    EXPERIMENT_ID = "roberta-emotion-classifier"
    URI = "http://host.docker.internal:5000"
    REGISTER = True
    MODEL_NAME = "roberta-emotion-model"