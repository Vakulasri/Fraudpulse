"""Step 7 - Load test the scoring API.

Fires N requests with C concurrent clients (asyncio + httpx) and measures
throughput (requests/sec) and latency percentiles (p50/p95/p99).
Results -> results/loadtest_latest.json (+ a timestamped copy) and are shown on the dashboard.

Run (API running WITHOUT --reload in another terminal):
    python -m src.loadtest --requests 2000 --concurrency 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

from src import config
from src.features import LABEL_COL
from src.synthetic import make_synthetic


def load_payloads(n: int = 1000) -> list[dict]:
    path = config.data_dir() / "stream.csv"
    df = pd.read_csv(path, nrows=n) if path.exists() else make_synthetic(n)
    return df.drop(columns=[LABEL_COL], errors="ignore").to_dict(orient="records")


async def _worker(client, url, payloads, n_requests, counter, latencies, errors):
    while True:
        i = counter[0]
        if i >= n_requests:
            return
        counter[0] += 1
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=random.choice(payloads))
            if r.status_code == 200:
                latencies.append((time.perf_counter() - t0) * 1000)
            else:
                errors.append(r.status_code)
        except httpx.HTTPError as e:
            errors.append(type(e).__name__)


async def run(n_requests: int, concurrency: int, base_url: str, workers: int = 1) -> dict:
    payloads = load_payloads()
    url = f"{base_url}/score"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        (await client.get(f"{base_url}/health")).raise_for_status()
        for p in payloads[:20]:                       # warm-up (not measured)
            await client.post(url, json=p)
        latencies: list[float] = []
        errors: list = []
        counter = [0]
        t0 = time.perf_counter()
        await asyncio.gather(*(_worker(client, url, payloads, n_requests, counter, latencies, errors)
                               for _ in range(concurrency)))
        elapsed = time.perf_counter() - t0

    lat = np.array(latencies) if latencies else np.array([np.nan])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requests": n_requests, "concurrency": concurrency,
        "successful": len(latencies), "errors": len(errors),
        "error_types": {str(k): errors.count(k) for k in set(errors)},
        "workers": workers,
        "duration_s": round(elapsed, 2),
        "throughput_rps": round(len(latencies) / elapsed, 1),
        "latency_ms": {k: round(float(v), 2) for k, v in {
            "mean": np.nanmean(lat), "p50": np.nanpercentile(lat, 50),
            "p95": np.nanpercentile(lat, 95), "p99": np.nanpercentile(lat, 99),
            "max": np.nanmax(lat)}.items()},
        "machine": f"{platform.system()} {platform.release()} / {platform.processor() or platform.machine()}",
        "note": f"{workers} uvicorn worker(s), client and server on the same machine",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Load test POST /score")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--url", default=config.API_URL)
    ap.add_argument("--workers", type=int, default=1,
                    help="how many uvicorn workers the API runs (recorded in the results)")
    a = ap.parse_args()

    res = asyncio.run(run(a.requests, a.concurrency, a.url, a.workers))
    out = config.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in ("loadtest_latest.json", f"loadtest_{stamp}.json"):
        (out / name).write_text(json.dumps(res, indent=2))
    with open(out / "loadtest_history.jsonl", "a") as f:   # every run, for the dashboard table
        f.write(json.dumps(res) + "\n")

    L = res["latency_ms"]
    print(f"\nRequests: {res['successful']}/{res['requests']} ok, {res['errors']} errors, "
          f"concurrency {res['concurrency']}, workers {res['workers']}")
    if res["errors"]:
        print(f"Error types: {res['error_types']}")
    print(f"Throughput: {res['throughput_rps']} req/s")
    print(f"Latency ms: p50={L['p50']}  p95={L['p95']}  p99={L['p99']}  max={L['max']}")
    print(f"Saved to {out / 'loadtest_latest.json'}")


if __name__ == "__main__":
    main()
