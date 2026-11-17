"""Prints a Markdown results table from models/metrics.json and results/loadtest_latest.json,
ready to paste into the README.   Run:  python -m src.report"""
import json

from src import config


def main() -> None:
    m_path = config.model_dir() / "metrics.json"
    if not m_path.exists():
        raise SystemExit("No metrics yet - run `python -m src.train` first.")
    m = json.loads(m_path.read_text())
    print(f"**Dataset:** {m['dataset']} - test set {m['test_rows']:,} transactions, "
          f"{m['test_frauds']} frauds\n")
    print("| Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |")
    print("|---|---|---|---|---|---|---|")
    names = {"logistic_regression": "Logistic Regression (baseline)", "xgboost": "**XGBoost**"}
    for k, r in m["models"].items():
        print(f"| {names.get(k, k)} | {r['threshold']:.4f} | {r['precision']:.3f} | "
              f"{r['recall']:.3f} | {r['f1']:.3f} | {r['pr_auc']:.3f} | {r['roc_auc']:.3f} |")
    print(f"\n'Always legit' accuracy would be {m['naive_always_legit_accuracy']:.2%} - "
          f"which is why accuracy is not reported as the headline metric.\n")

    lt_path = config.results_dir() / "loadtest_latest.json"
    if lt_path.exists():
        lt = json.loads(lt_path.read_text())
        L = lt["latency_ms"]
        print("| Load test | Requests | Concurrency | Throughput | p50 | p95 | p99 | Errors |")
        print("|---|---|---|---|---|---|---|---|")
        print(f"| POST /score | {lt['requests']} | {lt['concurrency']} | "
              f"{lt['throughput_rps']} req/s | {L['p50']} ms | {L['p95']} ms | {L['p99']} ms | "
              f"{lt['errors']} |")
        print(f"\n_{lt['machine']}; {lt['note']}._")


if __name__ == "__main__":
    main()
