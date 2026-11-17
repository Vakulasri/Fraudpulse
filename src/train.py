"""Step 2 - Train & evaluate the fraud models.

  * Baseline: Logistic Regression with class_weight='balanced'
  * Main model: XGBoost with scale_pos_weight (+ early stopping on validation PR-AUC)
  * Split: stratified train / validation / test (60/20/20) so every split keeps the ~0.17% fraud rate
  * Threshold: chosen on the VALIDATION set (max F1), then reported once on the untouched TEST set
  * Saves: models/xgb_model.json, models/model_meta.json, models/metrics.json, data/stream.csv

Run:
    python -m src.train --data data/creditcard.csv
    python -m src.train --synthetic          # quick smoke test with fake data
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import FEATURE_COLS, LABEL_COL, RAW_COLS, build_features
from src.synthetic import make_synthetic

SEED = 42


def split(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.4, stratify=df[LABEL_COL], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[LABEL_COL], random_state=SEED)
    return train, val, test


def pick_threshold(y_true, y_prob) -> float:
    """Threshold that maximises F1 on the validation set.
    Default 0.5 is meaningless on imbalanced data (especially with class weighting, which
    inflates probabilities). Max-F1 balances catching fraud (recall) vs. not annoying
    legit customers (precision). A bank could instead fix a minimum precision."""
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 6),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def pr_curve_points(y_true, y_prob, n: int = 200) -> dict:
    """Sample the curve on an even RECALL grid (interpolated precision), so the
    high-precision region isn't lost when the curve has tens of thousands of points."""
    p, r, _ = precision_recall_curve(y_true, y_prob)
    grid = np.linspace(0, 1, n)
    prec = [float(p[r >= g].max()) if (r >= g).any() else 0.0 for g in grid]
    return {"precision": np.round(prec, 4).tolist(), "recall": grid.round(4).tolist()}


def train(df: pd.DataFrame, model_dir: Path, stream_path: Path | None = None,
          dataset_name: str = "unknown", n_estimators: int = 2000) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = split(df)
    X_tr, X_va, X_te = (build_features(d) for d in (train_df, val_df, test_df))
    y_tr, y_va, y_te = (d[LABEL_COL].values for d in (train_df, val_df, test_df))
    print(f"Rows  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")
    print(f"Fraud rate  train={y_tr.mean():.4%}  test={y_te.mean():.4%}")

    # ---- Baseline: logistic regression -------------------------------------------------
    logreg = make_pipeline(StandardScaler(),
                           LogisticRegression(class_weight="balanced", max_iter=2000))
    logreg.fit(X_tr, y_tr)
    lr_thr = pick_threshold(y_va, logreg.predict_proba(X_va)[:, 1])
    lr_test_prob = logreg.predict_proba(X_te)[:, 1]
    lr_metrics = evaluate(y_te, lr_test_prob, lr_thr)

    # ---- Main model: XGBoost ----------------------------------------------------------
    # Full re-weighting (neg/pos, ~578 on Kaggle) is often TOO aggressive: the model stops
    # improving after a few trees and probabilities get squashed. So we try three strengths
    # and keep whichever gives the best PR-AUC on the VALIDATION set (test stays untouched).
    ratio = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    candidates = {"none (1)": 1.0, f"sqrt ({ratio ** 0.5:.0f})": ratio ** 0.5,
                  f"full ({ratio:.0f})": ratio}
    best = None
    print("\nXGBoost scale_pos_weight search (validation PR-AUC):")
    for label, spw in candidates.items():
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=100,
            tree_method="hist", random_state=SEED, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        bi = int(clf.best_iteration)
        bst = clf.get_booster()[: bi + 1]                  # keep only the best trees
        va_prob = bst.predict(xgb.DMatrix(X_va))
        ap = average_precision_score(y_va, va_prob)
        print(f"  {label:<14} PR-AUC={ap:.4f}  trees={bi + 1}")
        if best is None or ap > best[0]:
            best = (ap, spw, bi, bst, va_prob, label)
    _, spw, best_iter, booster, xgb_va_prob, chosen = best
    print(f"  -> chose scale_pos_weight {chosen}")
    xgb_thr = pick_threshold(y_va, xgb_va_prob)
    xgb_test_prob = booster.predict(xgb.DMatrix(X_te))
    xgb_metrics = evaluate(y_te, xgb_test_prob, xgb_thr)

    # "Always predict legit" - shows why accuracy is useless here
    naive_acc = round(1 - y_te.mean(), 4)

    # ---- Save artifacts ----------------------------------------------------------------
    version = datetime.now(timezone.utc).strftime("xgb-%Y%m%d-%H%M%S")
    booster.save_model(model_dir / "xgb_model.json")
    meta = {
        "model_version": version, "features": FEATURE_COLS, "threshold": xgb_thr,
        "threshold_rule": "max F1 on validation set", "best_iteration": best_iter,
        "scale_pos_weight": round(spw, 2), "dataset": dataset_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    metrics = {
        "model_version": version, "dataset": dataset_name,
        "test_rows": int(len(y_te)), "test_frauds": int(y_te.sum()),
        "naive_always_legit_accuracy": naive_acc,
        "models": {"logistic_regression": lr_metrics, "xgboost": xgb_metrics},
        "pr_curves": {"logistic_regression": pr_curve_points(y_te, lr_test_prob),
                      "xgboost": pr_curve_points(y_te, xgb_test_prob)},
        "feature_importance": dict(sorted(
            ((k, round(v, 4)) for k, v in booster.get_score(importance_type="gain").items()),
            key=lambda kv: -kv[1])[:15]),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Held-out test transactions become the "live stream" (the model never saw them)
    if stream_path is not None:
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        test_df[RAW_COLS + [LABEL_COL]].to_csv(stream_path, index=False)

    _print_report(metrics)
    return metrics


def _print_report(m: dict) -> None:
    print("\n=== Test-set results (threshold chosen on validation) ===")
    print(f"{'model':<22}{'thr':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'PR-AUC':>8}{'acc':>8}")
    for name, r in m["models"].items():
        print(f"{name:<22}{r['threshold']:>8.4f}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['f1']:>8.3f}{r['pr_auc']:>8.3f}{r['accuracy']:>8.4f}")
    print(f"Naive 'always legit' accuracy: {m['naive_always_legit_accuracy']:.4f}  "
          f"<- why accuracy is misleading")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train fraud models")
    ap.add_argument("--data", default=str(config.data_dir() / "creditcard.csv"))
    ap.add_argument("--synthetic", action="store_true", help="use generated fake data")
    args = ap.parse_args()

    if args.synthetic:
        df, name = make_synthetic(n=30_000, fraud_rate=0.01), "synthetic"
    else:
        path = Path(args.data)
        if not path.exists():
            raise SystemExit(f"{path} not found. See scripts/download_data.py, "
                             f"or run with --synthetic for a smoke test.")
        df, name = pd.read_csv(path), path.name
    train(df, config.model_dir(), stream_path=config.data_dir() / "stream.csv", dataset_name=name)
    print(f"\nSaved model + metrics to {config.model_dir()}")


if __name__ == "__main__":
    main()
