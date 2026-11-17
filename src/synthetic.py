"""Generates FAKE data with the same columns as the Kaggle dataset.
Used by the tests and for a quick smoke test before you download the real data.
Never report metrics from synthetic data on your resume."""
import numpy as np
import pandas as pd

from src.features import V_COLS


def make_synthetic(n: int = 20_000, fraud_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    V = rng.normal(0, 1, size=(n, 28))
    # Make fraud separable on a few components (loosely mimics the real data, where V14/V12/V10 matter)
    shifts = {13: -3.0, 11: -2.5, 9: -2.0, 3: 2.0, 16: -2.0}  # 0-based index -> shift
    for idx, s in shifts.items():
        V[y == 1, idx] += s + rng.normal(0, 1, size=(y == 1).sum())
    amount = np.round(rng.lognormal(3.5, 1.2, n), 2)
    amount[y == 1] = np.round(rng.lognormal(4.2, 1.5, (y == 1).sum()), 2)
    df = pd.DataFrame(V, columns=V_COLS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)).round(0))
    df["Amount"] = amount
    df["Class"] = y
    return df
