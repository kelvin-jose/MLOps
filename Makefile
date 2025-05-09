.PHONY: install lint test train docker-train

install:
	pip install -r requirements.txt

format:
	black src

lint:
	flake8 src

test:
	PYTHONPATH=src pytest

train:
	python src/train.py

docker-train:
	docker build -t roberta-train -f docker/Dockerfile.train .
	docker run --rm -v ./src/results:/app/results -v ./src/logs:/app/logs roberta-train

serve:
	docker build -t roberta-api -f docker/Dockerfile.serve .
	docker run -p 8000:8000 roberta-api