#!bin/bash

export MODEL_ID=${MODEL_ID:-$(ls -td results/checkpoint-* | head -1 | xargs -n 1 basename)}
uvicorn serving.app:app --host 0.0.0.0 --port 8000