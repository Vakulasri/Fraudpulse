"""Loads the trained XGBoost model once and scores transactions.

Explainability: XGBoost can compute exact TreeSHAP values natively
(predict(..., pred_contribs=True)), so we get per-transaction feature contributions
without the heavy `shap` package. These contributions are what the LLM explanation is
grounded on.

Performance: TreeSHAP is the most expensive part of a request, so it runs only when it
is needed - for FLAGGED transactions (~0.2% of traffic), or when the caller asks for it
with explain=true. Legit transactions take the fast path (probability only)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import build_features, describe_feature


class FraudModel:
    def __init__(self, model_dir: Path):
        model_file, meta_file = model_dir / "xgb_model.json", model_dir / "model_meta.json"
        if not model_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"No trained model in {model_dir}. Run `python -m src.train` first.")
        self.meta = json.loads(meta_file.read_text())
        self.features: list[str] = self.meta["features"]
        # FRAUD_THRESHOLD in .env lets ops tune sensitivity without retraining
        self.threshold = float(os.getenv("FRAUD_THRESHOLD") or self.meta["threshold"])
        self.version: str = self.meta["model_version"]
        self.booster = xgb.Booster()
        self.booster.load_model(model_file)
        self.booster.set_param({"nthread": 1})  # one request = small work; avoid thread thrash

    def _top_features(self, X: pd.DataFrame, rows: list[int], top_k: int) -> dict[int, list]:
        """TreeSHAP contributions for the given rows only."""
        if not rows:
            return {}
        sub = X.iloc[rows]
        contribs = self.booster.predict(xgb.DMatrix(sub, feature_names=self.features),
                                        pred_contribs=True)[:, :-1]  # last col = bias
        out = {}
        for k, i in enumerate(rows):
            order = np.argsort(-contribs[k])[:top_k]   # features pushing most toward fraud
            out[i] = [{
                "feature": self.features[j],
                "description": describe_feature(self.features[j]),
                "value": round(float(sub.iat[k, j]), 4),
                "contribution": round(float(contribs[k, j]), 4),
            } for j in order]
        return out

    def score(self, records: list[dict], top_k: int = 3, explain: bool = False) -> list[dict]:
        X = build_features(pd.DataFrame(records))
        # inplace_predict skips building a DMatrix -> noticeably faster for single rows
        probs = self.booster.inplace_predict(X.to_numpy(dtype=np.float32))
        flags = probs >= self.threshold
        need = [i for i in range(len(probs)) if explain or flags[i]]
        tops = self._top_features(X, need, top_k)
        return [{
            "fraud_probability": round(float(p), 6),
            "is_fraud": bool(f),
            "top_features": tops.get(i, []),
        } for i, (p, f) in enumerate(zip(probs, flags))]
