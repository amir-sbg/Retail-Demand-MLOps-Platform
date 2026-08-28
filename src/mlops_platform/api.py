from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mlops_platform.inference import predict_records

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # API dependencies are optional for local batch runs and tests.
    FastAPI = None
    HTTPException = RuntimeError
    BaseModel = object
    Field = None


REGISTRY_DIR = Path(os.environ.get("MODEL_REGISTRY_DIR", "models/registry"))


if FastAPI is not None:
    app = FastAPI(title="Retail Demand Forecasting API")

    class PredictionRequest(BaseModel):
        records: list[dict[str, Any]] = Field(..., min_length=1)

    @app.get("/health")
    def health() -> dict:
        champion = REGISTRY_DIR / "champion" / "metadata.json"
        return {
            "status": "ok" if champion.exists() else "no_model",
            "registry_dir": str(REGISTRY_DIR),
        }

    @app.get("/model")
    def model_info() -> dict:
        metadata_path = REGISTRY_DIR / "champion" / "metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="no champion model registered")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @app.post("/predict")
    def predict(request: PredictionRequest) -> dict:
        try:
            predictions = predict_records(request.records, REGISTRY_DIR)
        except Exception as exc:  # keep API errors readable instead of exposing stack traces
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"predictions": predictions}
else:
    app = None
