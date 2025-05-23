# MLOps Experiment

This repo is going to be used to keep track of all the MLOps experiments I did and I'm planning to do.

My goal is to train and serve a RoBERTa-Base model on a classification problem. The model and the dataset used in this experiment are available [here](https://huggingface.co/distilbert/distilroberta-base) and [here](https://huggingface.co/datasets/dair-ai/emotion).

## Project Directory Strcture

```
project/
├── README.md                # High-level project overview
├── .gitignore               # Git ignore rules
├── Makefile                 # Makefile
├── requirements.txt         # Python package dependencies
├── data/                    # Data folders for different stages
│   ├── raw/                 # Original unmodified data from external sources
├── notebooks/               # Exploratory data analysis and experiment notebooks
│   ├── eda.ipynb
|   ├── prototype.ipynb  
├── src/                     # Source code for the project
│   ├── __init__.py
│   ├── data/                # Data handling code
│   │   ├── download.py      # Scripts to fetch data
│   │   ├── config.py        # Configuration file
│   │   └── preprocess.py    # Text tokenization
│   ├── models/              # Model code and training scripts
│   │   ├── download.py      # Download the RoBERTa-Base model
│   │   ├── config.py        # Model configuration
│   │   └── utils.py         # Contains code for evaluation
│   ├── serving/             # Deployment layer code for inference
│   │   ├── test_model_load.py # Test case for model loading 
│   │   └── test_tokenizer.py  # Test case for tokenization
│   ├── tests/               # Test cases
│   │   ├── app.py           # FastAPI app exposing the prediction endpoints
│   │   └── predict.py       # Prediction logic (loading and serving the model)
│   ├── logs/                # TensorFlow logs directory
│   ├── results/             # Trained model checkpoints will be saved here
├── train.py                 # Traing script
├── docker/                  # Docker-related files for containerizing the serving
│   ├── Dockerfile.serve     # Dockerfile for the serving image
│   ├── Dockerfile.train     # Dockerfile for the training image
│   ├── start-serve.sh     
│   └── start-train.sh
└── scripts/                 # TODO: CI/CD related scripts
```
## Features

### Branch: exp-track
- **Experiment** and **model tracking** using ```mlflow```

### Branch: mvp-upgrade
- New ```docker/Dockerfile.train``` to **containerize** training
- A ```Makefile``` to **standardize** local dev commands
- ```black``` for auto code-formatting
- ```flake8``` for lint testing
- ```pytest``` to run minimal test cases

### Branch: mvp
- Data **download** and **preprocessing** steps
- A clear **training script** (```train.py```)
- A basic **serving layer** with **FastAPI** + **Docker**
- **Model artifacts** stored in ```results/```

## Run Locally

### Model Training
Clone the project

```bash
  git clone https://github.com/kelvin-jose/MLOps.git
```

Go to the project directory

```bash
  cd MLOps
```

Install dependencies

```bash
  make install
```

Code auto-formatting

```bash
  make format
```

Code lint test

```bash
  make lint
```

Start mlflow local server

```bash
  mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Model training

```bash
  make docker-train
```

To see the training logs

```bash
  tensorboard --logdir=./src/logs
```

### Serving
Building and serving the docker image

```bash
  make serve
```

Test the endpoint

```bash
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "Your sample input"}'
```

## Roadmap

- Add more integrations to make the pipeline robust

- Become successful


## License

[MIT](https://choosealicense.com/licenses/mit/)
