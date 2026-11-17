"""Feature engineering shared by training AND serving.
Using one function in both places prevents 'training/serving skew' (a classic production ML bug)."""
import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time", *V_COLS, "Amount"]          # columns a transaction arrives with
LABEL_COL = "Class"                              # 1 = fraud, 0 = legit
FEATURE_COLS = [*V_COLS, "Amount", "log_amount", "hour"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction columns into model features."""
    out = df[V_COLS].astype(float).copy()
    out["Amount"] = df["Amount"].astype(float)
    out["log_amount"] = np.log1p(out["Amount"])                 # tames the long tail of amounts
    out["hour"] = ((df["Time"].astype(float) // 3600) % 24)     # Time = seconds since first tx
    return out[FEATURE_COLS]


def describe_feature(name: str) -> str:
    """Human-readable label used in explanations. V1..V28 are anonymized PCA components."""
    if name.startswith("V"):
        return f"anonymized behaviour signal {name}"
    return {
        "Amount": "transaction amount",
        "log_amount": "transaction amount (log scale)",
        "hour": "hour of day",
    }.get(name, name)
