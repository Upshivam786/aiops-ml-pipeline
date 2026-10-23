import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

THRESHOLD = 0.80
NAME = "jev-model"


def score(version, X, y) -> float:
    model = mlflow.sklearn.load_model(f"models:/{NAME}/{version}")
    return f1_score(y, model.predict(X), average="weighted")


def main(test_path: str) -> int:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{NAME}'")
    if not versions:
        print(f"No registered versions of '{NAME}'. Run src/train.py first.")
        return 1
    cand = max(versions, key=lambda v: int(v.version))

    test = pd.read_csv(test_path)
    X, y = test.drop(columns="target"), test["target"]

    cand_f1 = score(cand.version, X, y)
    print(f"candidate v{cand.version}: f1={cand_f1:.4f} (threshold {THRESHOLD})")
    if cand_f1 < THRESHOLD:
        print("FAILED quality gate")
        return 1

    try:
        prod = client.get_model_version_by_alias(NAME, "production")
    except Exception:
        prod = None

    if prod is None:
        client.set_registered_model_alias(NAME, "production", cand.version)
        print(f"No production model yet. Promoted v{cand.version}")
        return 0
    if prod.version == cand.version:
        print(f"v{cand.version} is already production")
        return 0

    prod_f1 = score(prod.version, X, y)
    print(f"production v{prod.version}: f1={prod_f1:.4f} on the same test set")
    if cand_f1 > prod_f1:
        client.set_registered_model_alias(NAME, "production", cand.version)
        print(f"PROMOTED v{cand.version} over v{prod.version}")
    else:
        print(f"KEPT v{prod.version}: candidate did not beat it")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="data/test.csv")
    sys.exit(main(p.parse_args().test))
