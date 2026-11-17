"""SQLite storage for scored transactions.
WAL mode lets the dashboard READ while the simulator WRITES at the same time.
Each call opens its own short-lived connection, which is thread-safe and simple."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id     TEXT UNIQUE NOT NULL,
    scored_at          TEXT NOT NULL,          -- ISO timestamp (UTC)
    amount             REAL,
    tx_time            REAL,
    fraud_probability  REAL NOT NULL,
    is_flagged         INTEGER NOT NULL,       -- model decision (1/0)
    actual_label       INTEGER,                -- ground truth from the replayed dataset
    api_latency_ms     REAL,                   -- server-side model time
    roundtrip_ms       REAL,                   -- client-observed time
    model_version      TEXT,
    top_features       TEXT,                   -- JSON list
    explanation        TEXT,
    explanation_source TEXT                    -- 'llm' | 'template' | 'pending'
);
CREATE INDEX IF NOT EXISTS idx_tx_scored_at ON transactions(scored_at);
CREATE INDEX IF NOT EXISTS idx_tx_flagged  ON transactions(is_flagged);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Path, reset: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS transactions")
        conn.executescript(SCHEMA)


def insert_scored(db_path: Path, row: dict) -> None:
    row = {**row, "top_features": json.dumps(row.get("top_features", []))}
    cols = ", ".join(row)
    marks = ", ".join(f":{k}" for k in row)
    with _connect(db_path) as conn:
        conn.execute(f"INSERT OR REPLACE INTO transactions ({cols}) VALUES ({marks})", row)


def update_explanation(db_path: Path, transaction_id: str, text: str, source: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE transactions SET explanation=?, explanation_source=? "
                     "WHERE transaction_id=?", (text, source, transaction_id))


def read_df(db_path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with _connect(db_path) as conn:
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except Exception:  # table not created yet
            return pd.DataFrame()
