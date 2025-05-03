from fastapi import FastAPI
from pydantic import BaseModel

from serving.predict import predict

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def classify(input: TextInput):
    return predict(input.text)