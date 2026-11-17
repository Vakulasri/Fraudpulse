"""Central configuration. Everything is read from environment variables / the .env file,
so secrets (GROQ_API_KEY) and things that change (model names, rate limits) never live in code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    """Read a path from env; relative paths are resolved from the project root."""
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


# Read lazily (functions) so tests can override paths via env vars.
def data_dir() -> Path:
    return _path("DATA_DIR", "data")


def model_dir() -> Path:
    return _path("MODEL_DIR", "models")


def results_dir() -> Path:
    return _path("RESULTS_DIR", "results")


def db_path() -> Path:
    return _path("DB_PATH", "data/transactions.db")


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- Groq LLM settings (model name is configurable because Groq retires models) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MAX_RPM = int(os.getenv("GROQ_MAX_RPM", "20"))  # stay under the free-tier requests/minute
