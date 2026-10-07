from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def time_splits(
    frame: pd.DataFrame,
    validation_days: int,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "date" not in frame.columns:
        raise ValueError("feature table must include a date column")
    if validation_days < 1 or test_days < 1:
        raise ValueError("validation_days and test_days must be positive")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    max_date = data["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)

    train = data[data["date"] < validation_start]
    validation = data[(data["date"] >= validation_start) & (data["date"] < test_start)]
    test = data[data["date"] >= test_start]
    if train.empty or validation.empty or test.empty:
        raise ValueError("time split produced an empty train, validation, or test slice")
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def rolling_origin_splits(
    frame: pd.DataFrame,
    horizon_days: int,
    folds: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Build expanding-window folds ordered from oldest to newest."""

    if horizon_days < 1 or folds < 1:
        raise ValueError("horizon_days and folds must be positive")
    if "date" not in frame.columns:
        raise ValueError("feature table must include a date column")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    dates = np.sort(data["date"].dt.normalize().unique())
    if len(dates) <= horizon_days * folds:
        raise ValueError("not enough dates for the requested rolling-origin folds")

    windows = []
    for fold in range(folds):
        validation_end = len(dates) - horizon_days * (folds - fold - 1)
        validation_start = validation_end - horizon_days
        train_dates = dates[:validation_start]
        validation_dates = dates[validation_start:validation_end]
        train = data[data["date"].dt.normalize().isin(train_dates)].reset_index(drop=True)
        validation = data[data["date"].dt.normalize().isin(validation_dates)].reset_index(drop=True)
        windows.append((train, validation))
    return windows


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual, predicted = _matching_arrays(y_true, y_pred)
    residuals = predicted - actual
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    denominator = np.maximum(np.abs(actual), 1.0)
    demand_total = float(np.sum(np.abs(actual)))
    symmetric_denominator = np.maximum((np.abs(actual) + np.abs(predicted)) / 2.0, 1.0)
    ss_total = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 if ss_total == 0.0 and mse == 0.0 else 0.0
    if ss_total > 0:
        r2 = 1.0 - float(np.sum(residuals**2) / ss_total)
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "mape": float(np.mean(np.abs(residuals) / denominator)),
        "wape": float(np.sum(np.abs(residuals)) / demand_total) if demand_total > 0 else 0.0,
        "smape": float(np.mean(np.abs(residuals) / symmetric_denominator)),
        "r2": r2,
        "bias": float(np.mean(residuals)),
        "under_forecast_rate": float(np.mean(predicted < actual)),
        "over_forecast_rate": float(np.mean(predicted > actual)),
    }


def residual_summary(y_true, y_pred) -> dict[str, float]:
    actual, predicted = _matching_arrays(y_true, y_pred)
    residuals = predicted - actual
    return {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "p10": float(np.percentile(residuals, 10)),
        "p90": float(np.percentile(residuals, 90)),
        "max_abs": float(np.max(np.abs(residuals))),
    }


def split_conformal_summary(
    validation_actual,
    validation_predicted,
    test_actual,
    test_predicted,
    coverage: float = 0.90,
) -> dict[str, float | int]:
    """Calibrate a symmetric forecast interval on validation residuals."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    calibration_y, calibration_prediction = _matching_arrays(
        validation_actual,
        validation_predicted,
    )
    evaluation_y, evaluation_prediction = _matching_arrays(test_actual, test_predicted)
    calibration_errors = np.abs(calibration_prediction - calibration_y)
    level = min(
        1.0,
        np.ceil((len(calibration_errors) + 1) * coverage) / len(calibration_errors),
    )
    try:
        radius = float(np.quantile(calibration_errors, level, method="higher"))
    except TypeError:
        radius = float(np.quantile(calibration_errors, level, interpolation="higher"))
    covered = np.abs(evaluation_prediction - evaluation_y) <= radius
    return {
        "target_coverage": coverage,
        "observed_coverage": float(np.mean(covered)),
        "interval_radius": radius,
        "mean_interval_width": 2.0 * radius,
        "calibration_rows": int(len(calibration_errors)),
        "test_rows": int(len(evaluation_y)),
    }


def naive_forecast(frame: pd.DataFrame) -> np.ndarray:
    if "rolling_7_mean_units" not in frame.columns:
        raise ValueError("feature table must include rolling_7_mean_units")
    return frame["rolling_7_mean_units"].to_numpy(dtype=float)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")


def _matching_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("arrays must have the same shape")
    if actual.size == 0:
        raise ValueError("arrays must not be empty")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("arrays must contain only finite values")
    return actual, predicted
