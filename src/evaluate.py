import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

THRESHOLD = 0.80
NAME = "jev-model"

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
client = MlflowClient()

versions = client.search_model_versions(f"name='{NAME}'")
if not versions:
    print(f"No registered versions of '{NAME}'. Run src/train.py first.")
    sys.exit(1)

latest = max(versions, key=lambda v: int(v.version))
model = mlflow.sklearn.load_model(f"models:/{NAME}/{latest.version}")

test = pd.read_csv("data/test.csv")
f1 = f1_score(test["target"], model.predict(test.drop(columns="target")), average="weighted")
print(f"version={latest.version} f1={f1:.4f} threshold={THRESHOLD}")

if f1 < THRESHOLD:
    print("FAILED quality gate")
    sys.exit(1)

client.set_registered_model_alias(NAME, "production", latest.version)
print(f"Promoted version {latest.version} to @production")
