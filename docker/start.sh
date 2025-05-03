#!bin/bash

export MODEL_ID=${MODEL_ID:-checkpoint-1}
uvicorn serving.app:app --host 0.0.0.0 --port 8000