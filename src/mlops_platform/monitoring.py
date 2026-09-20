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

    reference = baseline.get("prediction_reference", baseline["target"])
    drift = {
        "predicted_units_psi": psi_from_profile(
            reference,
            predictions["predicted_units"].to_numpy(dtype=float),
        )
    }
    quality = current.get("quality", {})
    segments = segment_quality_report(
        predictions,
        segment_columns=("store_id", "category"),
        min_rows=5,
    )
    reference_mae = float(baseline.get("reference_metrics", {}).get("mae", 0.0))
    current_mae = float(quality.get("mae", 0.0)) if quality else 0.0
    alerts = monitoring_alerts(
        psi=drift["predicted_units_psi"],
        current_mae=current_mae,
        reference_mae=reference_mae,
        segments=segments,
        config=config,
    )
    retrain = bool(alerts)
    reason = str(alerts[0]["message"]) if alerts else "no retraining trigger crossed"

    report = {
        "rows": current["rows"],
        "drift": drift,
        "quality": quality,
        "segments": segments,
        "alerts": alerts,
        "reference_mae": reference_mae,
        "retrain_recommended": retrain,
        "reason": reason,
    }
    save_json(report, output_path)
    return report


def monitoring_alerts(
    psi: float,
    current_mae: float,
    reference_mae: float,
    segments: list[dict[str, float | int | str]],
    config: MonitoringConfig,
) -> list[dict[str, float | str]]:
    alerts: list[dict[str, float | str]] = []
    if psi >= config.psi_threshold:
        alerts.append(
            {
                "type": "prediction_drift",
                "severity": "high",
                "value": round(float(psi), 6),
                "threshold": config.psi_threshold,
                "message": "prediction PSI exceeded the monitoring threshold",
            }
        )

    if reference_mae > 0:
        mae_threshold = reference_mae * config.mae_degradation_ratio
        if current_mae > mae_threshold:
            alerts.append(
                {
                    "type": "error_degradation",
                    "severity": "high",
                    "value": round(float(current_mae), 6),
                    "threshold": round(float(mae_threshold), 6),
                    "message": "current MAE degraded against the training reference",
                }
            )

        if segments:
            worst_segment = segments[0]
            segment_mae = float(worst_segment["mae"])
            segment_threshold = mae_threshold * 1.25
            if segment_mae > segment_threshold:
                alerts.append(
                    {
                        "type": "segment_error",
                        "severity": "medium",
                        "value": round(segment_mae, 6),
                        "threshold": round(segment_threshold, 6),
                        "segment": str(worst_segment["segment"]),
                        "message": "one monitored segment is materially worse than baseline",
                    }
                )
    return alerts


def segment_quality_report(
    predictions: pd.DataFrame,
    segment_columns: tuple[str, ...] = ("store_id", "category"),
    min_rows: int = 10,
) -> list[dict[str, float | int | str]]:
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    required = {"actual_units", "predicted_units"}
    if not required.issubset(predictions.columns):
        return []

    available_segments = [column for column in segment_columns if column in predictions.columns]
    if not available_segments:
        return []

    rows = []
    for keys, group in predictions.groupby(available_segments, dropna=False):
        if len(group) < min_rows:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = group["actual_units"].to_numpy(dtype=float)
        predicted = group["predicted_units"].to_numpy(dtype=float)
        if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
            continue
        metrics = regression_metrics(actual, predicted)
        row = {
            "segment": " / ".join(f"{name}={value}" for name, value in zip(available_segments, keys)),
            "rows": int(len(group)),
            "mean_actual": float(np.mean(actual)),
            "mean_predicted": float(np.mean(predicted)),
            "mae": metrics["mae"],
            "wape": metrics["wape"],
            "bias": metrics["bias"],
        }
        rows.append(row)

    rows.sort(key=lambda row: (float(row["mae"]), float(row["rows"])), reverse=True)
    return rows[:10]


def psi_from_profile(reference: dict, observed: np.ndarray) -> float:
    if "bin_edges" in reference and "expected_share" in reference:
        return psi_from_bins(
            reference["bin_edges"],
            reference["expected_share"],
            observed,
        )
    return psi_from_stats(
        mean=float(reference["mean"]),
        std=float(reference["std"]),
        observed=observed,
    )


def psi_from_bins(bin_edges: list[float], expected_share: list[float], observed: np.ndarray) -> float:
    values = np.asarray(observed, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("observed values must be non-empty and finite")
    edges = np.asarray(bin_edges, dtype=float).copy()
    expected = np.asarray(expected_share, dtype=float)
    if edges.size < 3:
        raise ValueError("at least two bins are required for PSI")
    if expected.shape[0] != edges.shape[0] - 1:
        raise ValueError("expected_share must have one value per bin")
    if not np.all(np.isfinite(edges)) or not np.all(np.isfinite(expected)):
        raise ValueError("PSI reference values must be finite")
    if np.any(expected < 0) or expected.sum() <= 0:
        raise ValueError("expected_share must be non-negative with a positive sum")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing")

    edges[0] = min(edges[0], values.min()) - 1e-6
    edges[-1] = max(edges[-1], values.max()) + 1e-6
    observed_counts, _ = np.histogram(values, bins=edges)
    expected = np.clip(expected / expected.sum(), 1e-6, 1.0)
    observed_share = np.clip(observed_counts / observed_counts.sum(), 1e-6, 1.0)
    return float(np.sum((observed_share - expected) * np.log(observed_share / expected)))


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
