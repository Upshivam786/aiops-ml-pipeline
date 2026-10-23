import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("jev-model")


def load_data() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=5, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    return df


def main(n_estimators: int, max_depth):
    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    X_train, y_train = train_df.drop(columns="target"), train_df["target"]
    X_test, y_test = test_df.drop(columns="target"), test_df["target"]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        ).fit(X_train, y_train)
        f1 = f1_score(y_test, model.predict(X_test), average="weighted")

        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, "model", registered_model_name="jev-model")
        print(f"f1={f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    a = p.parse_args()
    main(a.n_estimators, a.max_depth)
