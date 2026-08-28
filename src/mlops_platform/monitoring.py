from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_platform.config import MonitoringConfig
from mlops_platform.metrics import regression_metrics, save_json


def build_prediction_profile(predictions: pd.DataFrame) -> dict:
    if "predicted_units" not in predictions.columns:
        raise ValueError("predictions must include predicted_units")
    values = predictions["predicted_units"].to_numpy(dtype=float)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("predicted_units must be non-empty and finite")
    profile = {
        "rows": int(len(predictions)),
        "predicted_units": _distribution_profile(values),
    }
    if {"actual_units", "predicted_units"}.issubset(predictions.columns):
        profile["quality"] = regression_metrics(predictions["actual_units"], predictions["predicted_units"])
    return profile


def monitor_predictions(
    baseline_profile_path: Path,
    prediction_log_path: Path,
    output_path: Path,
    config: MonitoringConfig,
) -> dict:
    config.validate()
    baseline = json.loads(baseline_profile_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_log_path)
    current = build_prediction_profile(predictions)

    baseline_target = baseline["target"]
    drift = {
        "predicted_units_psi": psi_from_stats(
            mean=float(baseline_target["mean"]),
            std=float(baseline_target["std"]),
            observed=predictions["predicted_units"].to_numpy(dtype=float),
        )
    }
    quality = current.get("quality", {})
    reference_mae = float(baseline.get("reference_metrics", {}).get("mae", 0.0))
    current_mae = float(quality.get("mae", 0.0)) if quality else 0.0
    retrain = drift["predicted_units_psi"] >= config.psi_threshold
    if reference_mae > 0 and current_mae > reference_mae * config.mae_degradation_ratio:
        retrain = True

    report = {
        "rows": current["rows"],
        "drift": drift,
        "quality": quality,
        "reference_mae": reference_mae,
        "retrain_recommended": retrain,
        "reason": _reason(drift["predicted_units_psi"], current_mae, reference_mae, config),
    }
    save_json(report, output_path)
    return report


def psi_from_stats(mean: float, std: float, observed: np.ndarray, bins: int = 10) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = np.asarray(observed, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("observed values must be non-empty and finite")
    std = max(float(std), 1e-6)
    expected_sample = np.linspace(mean - 3 * std, mean + 3 * std, num=max(200, bins * 40))
    edges = np.quantile(expected_sample, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = min(edges[0], values.min()) - 1e-6
    edges[-1] = max(edges[-1], values.max()) + 1e-6
    expected_counts, _ = np.histogram(expected_sample, bins=edges)
    observed_counts, _ = np.histogram(values, bins=edges)
    expected_share = np.clip(expected_counts / expected_counts.sum(), 1e-6, 1.0)
    observed_share = np.clip(observed_counts / observed_counts.sum(), 1e-6, 1.0)
    return float(np.sum((observed_share - expected_share) * np.log(observed_share / expected_share)))


def _distribution_profile(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _reason(psi: float, current_mae: float, reference_mae: float, config: MonitoringConfig) -> str:
    if psi >= config.psi_threshold:
        return "prediction distribution drift is above the PSI threshold"
    if reference_mae > 0 and current_mae > reference_mae * config.mae_degradation_ratio:
        return "prediction error degraded beyond the configured gate"
    return "no retraining trigger crossed"
