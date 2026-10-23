"""Compare training data against a production batch and flag data drift."""
import argparse
import os
import json
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

FEATURES = [f"f{i}" for i in range(8)]


def make_batch(mode: str, n: int = 300) -> pd.DataFrame:
    pool = pd.read_csv("data/test.csv")[FEATURES]
    batch = pool.sample(min(n, len(pool)), random_state=1).copy()
    if mode == "drift":
        batch["f0"] += 2.0   # mean shift
        batch["f1"] -= 1.5   # mean shift
        batch["f2"] += 1.0   # mean shift
        batch["f3"] *= 2.0   # variance change
    return batch.reset_index(drop=True)


def main(mode: str, threshold: float) -> int:
    os.makedirs("reports", exist_ok=True)
    reference = pd.read_csv("data/train.csv")[FEATURES]
    current = make_batch(mode)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(f"reports/drift_{mode}.html")

    result = next(
        m["result"]
        for m in report.as_dict()["metrics"]
        if "share_of_drifted_columns" in m["result"]
    )
    share = result["share_of_drifted_columns"]
    summary = {
        "mode": mode,
        "drifted_columns": result["number_of_drifted_columns"],
        "total_columns": result["number_of_columns"],
        "share_drifted": round(share, 3),
        "threshold": threshold,
        "drift_detected": share >= threshold,
    }
    json.dump(summary, open(f"reports/drift_{mode}.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 1 if summary["drift_detected"] else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["normal", "drift"], default="normal")
    p.add_argument("--threshold", type=float, default=0.3)
    a = p.parse_args()
    sys.exit(main(a.mode, a.threshold))
