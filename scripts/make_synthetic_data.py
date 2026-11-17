"""Write a synthetic dataset to data/synthetic.csv for a quick smoke test.
Run:  python scripts/make_synthetic_data.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import data_dir  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402

if __name__ == "__main__":
    out = data_dir() / "synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic(n=30_000, fraud_rate=0.01).to_csv(out, index=False)
    print(f"Wrote {out}")
