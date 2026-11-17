"""Test setup: trains a tiny model on synthetic data in a temp folder, so the tests
never depend on the Kaggle download or on your real trained model."""
import os
import tempfile
from pathlib import Path

# Must be set BEFORE importing src modules that read config
_TMP = Path(tempfile.mkdtemp(prefix="fraudpulse_test_"))
os.environ["MODEL_DIR"] = str(_TMP / "models")
os.environ["FRAUD_THRESHOLD"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.features import LABEL_COL  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402
from src.train import train  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic(n=6000, fraud_rate=0.03, seed=1)


@pytest.fixture(scope="session")
def client(synthetic_df):
    train(synthetic_df, Path(os.environ["MODEL_DIR"]), stream_path=None,
          dataset_name="synthetic-test", n_estimators=80)
    from src.api.main import app
    with TestClient(app) as c:   # `with` runs the startup (model loading)
        yield c


@pytest.fixture
def legit_tx(synthetic_df):
    return synthetic_df[synthetic_df[LABEL_COL] == 0].iloc[0].drop(LABEL_COL).to_dict()


@pytest.fixture
def fraud_tx(synthetic_df):
    return synthetic_df[synthetic_df[LABEL_COL] == 1].iloc[0].drop(LABEL_COL).to_dict()
