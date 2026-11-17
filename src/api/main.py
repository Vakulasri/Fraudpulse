"""Step 3 - FastAPI scoring service.

Run:   uvicorn src.api.main:app --port 8000
Docs:  http://127.0.0.1:8000/docs   (interactive Swagger UI, generated from the Pydantic schemas)

The API only scores. It does NOT call the LLM - that happens asynchronously downstream,
so a slow/rate-limited LLM can never slow down fraud scoring."""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src import config
from src.api.model_service import FraudModel
from src.api.schemas import HealthResponse, ScoreResponse, Transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not per request
    try:
        app.state.model = FraudModel(config.model_dir())
    except FileNotFoundError as e:
        print(f"[startup] {e}")
        app.state.model = None
    yield


app = FastAPI(title="FraudPulse Scoring API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    m: FraudModel | None = request.app.state.model
    if m is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(status="ok", model_loaded=True,
                          model_version=m.version, threshold=m.threshold)


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction, request: Request, explain: bool = False):
    # Plain `def` (not async): FastAPI runs it in a thread pool, right for CPU-bound model calls.
    # top_features (SHAP) are returned for flagged transactions, or for any tx with ?explain=true
    m: FraudModel | None = request.app.state.model
    if m is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")
    start = time.perf_counter()
    result = m.score([tx.model_dump(exclude={"transaction_id"})], explain=explain)[0]
    return ScoreResponse(
        transaction_id=tx.transaction_id or str(uuid.uuid4()),
        threshold=m.threshold,
        model_version=m.version,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
        **result,
    )
