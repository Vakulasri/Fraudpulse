"""Pydantic models = the API's contract. FastAPI uses them to validate requests
(bad input -> automatic 422 error) and to generate the interactive docs at /docs."""
from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    # extra="forbid": unknown fields are rejected; allow_inf_nan=False: no NaN/inf sneaking in
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: str | None = Field(None, max_length=64, description="Optional client ID")
    Time: float = Field(..., ge=0, description="Seconds since the first transaction in the dataset")
    Amount: float = Field(..., ge=0, le=1_000_000, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class FeatureContribution(BaseModel):
    feature: str
    description: str
    value: float
    contribution: float  # SHAP value in log-odds; positive = pushes toward fraud


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float
    top_features: list[FeatureContribution]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    threshold: float | None = None
