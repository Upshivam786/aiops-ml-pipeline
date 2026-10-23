import argparse, time
import pandas as pd
import requests

p = argparse.ArgumentParser()
p.add_argument("--drift", action="store_true")
p.add_argument("--n", type=int, default=600)
a = p.parse_args()

df = pd.read_csv("data/test.csv").drop(columns="target")
df = df.sample(a.n, replace=True, random_state=2).reset_index(drop=True)
if a.drift:
    df["f0"] += 2.0; df["f1"] -= 1.5; df["f2"] += 1.0; df["f3"] *= 2.0

for _, row in df.iterrows():
    requests.post("http://localhost:8000/predict", json=row.to_dict(), timeout=5)
    time.sleep(0.05)
print("sent", len(df), "drifted" if a.drift else "normal")
