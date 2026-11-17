"""Background worker that turns flagged transactions into explanations.

  scoring loop --(flagged tx)--> bounded queue --> worker thread --(paced)--> Groq --> SQLite

  * Only FLAGGED transactions are queued -> very few LLM calls.
  * A rate limiter spaces calls to GROQ_MAX_RPM (requests per minute) for the free tier.
  * The queue is bounded: if alerts arrive faster than we can explain them, the overflow
    gets an instant template explanation instead of piling up (backpressure)."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

from src import config, db
from src.explain.llm import GroqExplainer, template_explanation


class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.min_interval = 60.0 / max(max_per_minute, 1)
        self._last = 0.0

    def wait(self) -> None:
        sleep_for = self._last + self.min_interval - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.monotonic()


class ExplanationWorker(threading.Thread):
    _STOP = object()

    def __init__(self, db_path: Path, explainer: GroqExplainer | None = None,
                 max_rpm: int = config.GROQ_MAX_RPM, max_queue: int = 50):
        super().__init__(daemon=True, name="explanation-worker")
        self.db_path = db_path
        self.explainer = explainer or GroqExplainer()
        self.limiter = RateLimiter(max_rpm)
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stats = {"llm": 0, "template": 0, "overflow": 0}

    def submit(self, job: dict) -> None:
        """job keys: transaction_id, prob, threshold, amount, hour, top_features"""
        try:
            self.q.put_nowait(job)
        except queue.Full:
            text = template_explanation(job["prob"], job["amount"], job["top_features"])
            db.update_explanation(self.db_path, job["transaction_id"], text, "template")
            self.stats["overflow"] += 1

    def run(self) -> None:
        while True:
            job = self.q.get()
            if job is self._STOP:
                break
            if self.explainer.enabled:
                self.limiter.wait()   # only pace real API calls
            text, source = self.explainer.explain(
                job["prob"], job["threshold"], job["amount"], job["hour"], job["top_features"])
            db.update_explanation(self.db_path, job["transaction_id"], text, source)
            self.stats[source] += 1

    def stop(self, timeout: float = 30.0) -> None:
        """Let queued jobs finish (up to timeout), then stop."""
        self.q.put(self._STOP)
        self.join(timeout)
