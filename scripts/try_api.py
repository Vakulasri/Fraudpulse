"""Send one legit and one fraud transaction from data/stream.csv to the running API.
Run:  python scripts/try_api.py"""
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

df = pd.read_csv(config.data_dir() / "stream.csv")
for label in (0, 1):
    tx = df[df.Class == label].iloc[0].drop("Class").to_dict()
    r = httpx.post(f"{config.API_URL}/score", json=tx, timeout=10)
    print(f"\n--- actual {'FRAUD' if label else 'legit'} ---  HTTP {r.status_code}")
    print(json.dumps(r.json(), indent=2))

bad = {**tx, "Amount": -5}
r = httpx.post(f"{config.API_URL}/score", json=bad, timeout=10)
print(f"\n--- invalid input (negative Amount) ---  HTTP {r.status_code}")
print(r.json()["detail"][0]["msg"])
