"""Where transactions come from. The simulator only depends on the abstract
`TransactionSource`, so swapping the CSV replay for Kafka later is a one-class change
(the 'adapter' / dependency-inversion pattern)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.features import LABEL_COL


class TransactionSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[dict]:
        """Yield one transaction dict at a time (raw columns + optional 'Class' label)."""


class CsvReplaySource(TransactionSource):
    """Replays the held-out test set (data/stream.csv).

    fraud_ratio: real data is ~0.17% fraud, i.e. ~1 alert every 10 min at 1 tx/s.
    For a demo you can enrich the stream (e.g. 0.02 = 2%) so alerts appear quickly.
    """

    def __init__(self, path: Path, fraud_ratio: float = 0.0, loop: bool = False, seed: int = 7):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python -m src.train` first.")
        df = pd.read_csv(path)
        if fraud_ratio > 0:
            fraud, legit = df[df[LABEL_COL] == 1], df[df[LABEL_COL] == 0]
            n_fraud = int(len(legit) * fraud_ratio / (1 - fraud_ratio))
            fraud = fraud.sample(n_fraud, replace=n_fraud > len(fraud), random_state=seed)
            df = pd.concat([legit, fraud])
        self.df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
        self.loop = loop

    def __iter__(self) -> Iterator[dict]:
        while True:
            for rec in self.df.to_dict(orient="records"):
                yield rec
            if not self.loop:
                return


class KafkaSource(TransactionSource):
    """Placeholder showing how Kafka would plug in (not used, to keep Windows setup simple).

        from confluent_kafka import Consumer
        c = Consumer({"bootstrap.servers": "localhost:9092", "group.id": "fraud-scorer",
                      "auto.offset.reset": "latest"})
        c.subscribe(["transactions"])
        while True:
            msg = c.poll(1.0)
            if msg and not msg.error():
                yield json.loads(msg.value())
    """

    def __init__(self, *_, **__):
        raise NotImplementedError("Kafka source not enabled - see docstring.")

    def __iter__(self) -> Iterator[dict]:  # pragma: no cover
        raise NotImplementedError
