"""Download the Kaggle 'Credit Card Fraud Detection' dataset (mlg-ulb/creditcardfraud).

Option A (needs a free Kaggle account + API token):
    pip install kaggle
    # put kaggle.json in C:\\Users\\<you>\\.kaggle\\kaggle.json
    python scripts/download_data.py

Option B (no code): download the zip in your browser from
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and unzip creditcard.csv into the data/ folder.
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import data_dir  # noqa: E402


def main() -> None:
    out_dir = data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "creditcard.csv"
    if target.exists():
        print(f"Already present: {target}")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Install the Kaggle client first:  pip install kaggle   (or use Option B in this file)")

    api = KaggleApi()
    api.authenticate()
    print("Downloading mlg-ulb/creditcardfraud (~66 MB zipped)...")
    api.dataset_download_files("mlg-ulb/creditcardfraud", path=str(out_dir), unzip=False)
    zip_path = out_dir / "creditcardfraud.zip"
    with zipfile.ZipFile(zip_path) as z:
        z.extract("creditcard.csv", out_dir)
    zip_path.unlink()
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
