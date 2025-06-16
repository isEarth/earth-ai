from fastapi import FastAPI
from predict import predict
import pandas as pd

app = FastAPI()

@app.get("/")
def health():
    return {"message": "RVGAE 학습 및 예측 API"}

@app.get("/predict")
def run_prediction():
    results = predict()
    return results