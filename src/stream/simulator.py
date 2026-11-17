"""Step 4 - Real-time stream simulator.

Replays held-out transactions to the API at a fixed rate, like a live card stream:
    source (CSV/Kafka) -> POST /score -> SQLite -> (if flagged) explanation worker -> Groq

Run (API must be running in another terminal):
    python -m src.stream.simulator --rate 5 --limit 500 --fraud-ratio 0.02 --reset-db
"""
from __future__ import annotations

import argparse
import time
import uuid
from datetime import datetime, timezone

import httpx

from src import config, db
from src.explain.worker import ExplanationWorker
from src.features import LABEL_COL
from src.stream.sources import CsvReplaySource


def wait_for_api(client: httpx.Client, url: str, tries: int = 15) -> dict:
    for _ in range(tries):
        try:
            r = client.get(f"{url}/health")
            if r.status_code == 200 and r.json().get("model_loaded"):
                return r.json()
        except httpx.HTTPError:
            pass
        print(f"Waiting for API at {url} ...")
        time.sleep(2)
    raise SystemExit(f"API not reachable at {url}. Start it with: uvicorn src.api.main:app --port 8000")


def run(rate: float, limit: int, fraud_ratio: float, loop: bool, reset_db: bool) -> None:
    db_path = config.db_path()
    db.init_db(db_path, reset=reset_db)
    source = CsvReplaySource(config.data_dir() / "stream.csv", fraud_ratio=fraud_ratio, loop=loop)
    worker = ExplanationWorker(db_path)
    worker.start()
    print(f"LLM explanations: {'Groq ' + config.GROQ_MODEL if worker.explainer.enabled else 'OFF (template fallback; set GROQ_API_KEY in .env)'}")

    sent = flagged = errors = 0
    interval = 1.0 / rate
    next_tick = time.perf_counter()
    with httpx.Client(timeout=10) as client:
        health = wait_for_api(client, config.API_URL)
        print(f"Connected to API - model {health['model_version']}, threshold {health['threshold']:.3f}")
        try:
            for rec in source:
                if limit and sent >= limit:
                    break
                label = rec.pop(LABEL_COL, None)
                rec["transaction_id"] = f"tx-{uuid.uuid4().hex[:12]}"
                t0 = time.perf_counter()
                try:
                    r = client.post(f"{config.API_URL}/score", json=rec)
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    errors += 1
                    print(f"[error] {e}")
                    continue
                res = r.json()
                roundtrip = (time.perf_counter() - t0) * 1000
                db.insert_scored(db_path, {
                    "transaction_id": res["transaction_id"],
                    "scored_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "amount": rec["Amount"], "tx_time": rec["Time"],
                    "fraud_probability": res["fraud_probability"],
                    "is_flagged": int(res["is_fraud"]),
                    "actual_label": None if label is None else int(label),
                    "api_latency_ms": res["latency_ms"], "roundtrip_ms": round(roundtrip, 2),
                    "model_version": res["model_version"], "top_features": res["top_features"],
                    "explanation": None,
                    "explanation_source": "pending" if res["is_fraud"] else None,
                })
                sent += 1
                if res["is_fraud"]:
                    flagged += 1
                    worker.submit({
                        "transaction_id": res["transaction_id"], "prob": res["fraud_probability"],
                        "threshold": res["threshold"], "amount": rec["Amount"],
                        "hour": (rec["Time"] // 3600) % 24, "top_features": res["top_features"],
                    })
                    print(f"  ALERT {res['transaction_id']}  p={res['fraud_probability']:.3f}  "
                          f"${rec['Amount']:,.2f}  (actual={'FRAUD' if label == 1 else 'legit'})")
                if sent % 50 == 0:
                    print(f"sent={sent} flagged={flagged} errors={errors} queue={worker.q.qsize()}")
                # fixed-rate pacing that doesn't drift
                next_tick += interval
                time.sleep(max(0.0, next_tick - time.perf_counter()))
        except KeyboardInterrupt:
            print("\nStopping...")
    print(f"Done. sent={sent} flagged={flagged} errors={errors}. Finishing explanations...")
    worker.stop(timeout=60)
    print(f"Explanations: {worker.stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay transactions to the scoring API")
    ap.add_argument("--rate", type=float, default=5, help="transactions per second")
    ap.add_argument("--limit", type=int, default=500, help="stop after N transactions (0 = no limit)")
    ap.add_argument("--fraud-ratio", type=float, default=0.0,
                    help="demo mode: enrich the stream to this fraud share, e.g. 0.02")
    ap.add_argument("--loop", action="store_true", help="restart from the beginning when done")
    ap.add_argument("--reset-db", action="store_true", help="clear previous results first")
    a = ap.parse_args()
    run(a.rate, a.limit, a.fraud_ratio, a.loop, a.reset_db)


if __name__ == "__main__":
    main()
