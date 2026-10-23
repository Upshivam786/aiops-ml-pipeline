import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_URI = "models:/jev-model@production"

model = mlflow.sklearn.load_model(MODEL_URI)

PREDICTIONS = Counter(
    "jev_predictions_total", "Predictions by class", ["predicted_class"]
)

app = FastAPI(title="JEV model API")
Instrumentator().instrument(app).expose(app)  # adds /metrics


class Features(BaseModel):
    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict")
def predict(features: Features):
    df = pd.DataFrame([features.model_dump()])
    pred = int(model.predict(df)[0])
    PREDICTIONS.labels(predicted_class=str(pred)).inc()
    return {"prediction": pred}
