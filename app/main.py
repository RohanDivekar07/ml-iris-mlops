from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
model = joblib.load(MODEL_PATH)
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Prediction API")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"message": "Welcome to the Iris Prediction API"}


@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width,
                 data.petal_length, data.petal_width]]
    pred = int(model.predict(features)[0])
    return {"prediction": pred, "class_name": CLASS_NAMES[pred]}
