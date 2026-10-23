import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("demo-classifier")


def load_data() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=5, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df


def apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate the world shifting (same shifts as the drift check)."""
    df = df.copy()
    df["f0"] += 2.0
    df["f1"] -= 1.5
    df["f2"] += 1.0
    df["f3"] *= 2.0
    return df


def main(n_estimators: int, max_depth, drift: bool):
    os.makedirs("data", exist_ok=True)
    df = load_data()
    if drift:
        df = apply_drift(df)
    prefix = "new_" if drift else ""

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(f"data/{prefix}train.csv", index=False)
    test_df.to_csv(f"data/{prefix}test.csv", index=False)

    X_train, y_train = train_df.drop(columns="target"), train_df["target"]
    X_test, y_test = test_df.drop(columns="target"), test_df["target"]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        ).fit(X_train, y_train)
        f1 = f1_score(y_test, model.predict(X_test), average="weighted")

        mlflow.log_params(
            {"n_estimators": n_estimators, "max_depth": max_depth, "drifted_world": drift}
        )
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="demo-classifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"f1={f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--drift", action="store_true", help="train on the shifted world")
    a = p.parse_args()
    main(a.n_estimators, a.max_depth, a.drift)
