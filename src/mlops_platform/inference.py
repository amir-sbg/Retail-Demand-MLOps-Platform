from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_platform.features import FEATURE_COLUMNS, model_matrix
from mlops_platform.metrics import regression_metrics, save_json
from mlops_platform.registry import load_champion


def batch_predict(
    feature_path: Path,
    registry_dir: Path,
    output_path: Path,
    metrics_path: Path | None = None,
) -> pd.DataFrame:
    bundle, metadata = load_champion(registry_dir)
    frame = pd.read_csv(feature_path, parse_dates=["date"])
    matrix, _ = model_matrix(frame)
    matrix = _align_columns(matrix, bundle["feature_columns"])
    predictions = bundle["model"].predict(matrix)

    output = frame[["date", "store_id", "sku_id", "category"]].copy()
    for column in FEATURE_COLUMNS:
        output[column] = frame[column].to_numpy()
    output["model_version"] = metadata["version"]
    output["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()
    output["predicted_units"] = predictions.clip(min=0)
    if bundle["target_column"] in frame.columns:
        output["actual_units"] = frame[bundle["target_column"]].to_numpy(dtype=float)
        output["absolute_error"] = (output["predicted_units"] - output["actual_units"]).abs()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    if metrics_path is not None and "actual_units" in output.columns:
        save_json(
            {
                "model_version": metadata["version"],
                "rows": len(output),
                "prediction_summary": prediction_output_summary(output),
                "metrics": regression_metrics(output["actual_units"], output["predicted_units"]),
            },
            metrics_path,
        )
    return output


def prediction_output_summary(predictions: pd.DataFrame) -> dict[str, float | int | bool]:
    if "predicted_units" not in predictions.columns:
        raise ValueError("predictions must include predicted_units")
    values = predictions["predicted_units"].to_numpy(dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("predicted_units must be non-empty and finite")

    return {
        "rows": int(len(predictions)),
        "unique_stores": int(predictions["store_id"].nunique()) if "store_id" in predictions else 0,
        "unique_skus": int(predictions["sku_id"].nunique()) if "sku_id" in predictions else 0,
        "has_actuals": "actual_units" in predictions.columns,
        "min_predicted_units": float(np.min(values)),
        "p50_predicted_units": float(np.percentile(values, 50)),
        "p90_predicted_units": float(np.percentile(values, 90)),
        "max_predicted_units": float(np.max(values)),
    }


def predict_records(records: list[dict], registry_dir: Path) -> list[dict]:
    if not records:
        return []
    bundle, metadata = load_champion(registry_dir)
    frame = pd.DataFrame(records)
    matrix, _ = model_matrix(frame)
    matrix = _align_columns(matrix, bundle["feature_columns"])
    predictions = bundle["model"].predict(matrix).clip(min=0)
    return [
        {
            "model_version": metadata["version"],
            "predicted_units": float(prediction),
        }
        for prediction in predictions
    ]


def _align_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    return aligned.loc[:, columns]
