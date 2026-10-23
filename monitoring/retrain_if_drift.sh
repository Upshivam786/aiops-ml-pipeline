#!/usr/bin/env bash
# usage: retrain_if_drift.sh [normal|drift]
MODE="${1:-drift}"

python monitoring/drift_check.py --mode "$MODE"
rc=$?

if [ "$rc" -eq 0 ]; then
  echo "No drift detected. Nothing to do."
  exit 0
elif [ "$rc" -ne 1 ]; then
  echo "Drift check failed with an error (exit $rc)."
  exit "$rc"
fi

echo "Drift detected. Retraining on the new data..."
python src/train.py --n-estimators 300 --drift
python src/evaluate.py --test data/new_test.csv
