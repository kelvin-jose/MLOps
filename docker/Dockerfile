FROM python:3.10-slim

WORKDIR /app

COPY ./src/serving ./serving
COPY ./src/results ./results
COPY ./docker/start.sh ./start.sh
COPY ./src/models/config.py ./models/config.py
COPY ./requirements.txt .

RUN pip3 install -r requirements.txt

CMD ["bash", "./start.sh"]

