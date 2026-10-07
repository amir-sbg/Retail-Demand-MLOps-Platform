from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mlops_platform.config import PipelineConfig
from mlops_platform.features import model_matrix
from mlops_platform.metrics import (
    naive_forecast,
    regression_metrics,
    residual_summary,
    save_json,
    split_conformal_summary,
    time_splits,
)
from mlops_platform.tracking import ExperimentTracker


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    run_dir: Path
    model_path: Path
    metrics: dict
    feature_columns: list[str]
    promoted: bool = False


def train_forecaster(config: PipelineConfig) -> TrainingResult:
    config.validate()
    feature_frame = pd.read_csv(config.data.feature_path, parse_dates=["date"])
    train, validation, test = time_splits(
        feature_frame,
        validation_days=config.training.validation_days,
        test_days=config.training.test_days,
    )
    matrices = _split_model_matrices(train, validation, test)
    y_train = train[config.training.target_column].to_numpy(dtype=float)
    y_validation = validation[config.training.target_column].to_numpy(dtype=float)
    y_test = test[config.training.target_column].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.055,
        l2_regularization=0.04,
        max_leaf_nodes=31,
        random_state=config.training.random_state,
    )

    tracker = ExperimentTracker(config.tracking)
    run = tracker.start_run(config.to_dict())
    tracker.log_params(
        run,
        {
            "model_type": "HistGradientBoostingRegressor",
            "feature_count": len(matrices.feature_columns),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
    )

    model.fit(matrices.x_train, y_train)
    validation_predictions = model.predict(matrices.x_validation)
    test_predictions = model.predict(matrices.x_test)
    baseline_validation = naive_forecast(validation)
    baseline_test = naive_forecast(test)

    metrics = {
        "validation": regression_metrics(y_validation, validation_predictions),
        "test": regression_metrics(y_test, test_predictions),
        "baseline_validation": regression_metrics(y_validation, baseline_validation),
        "baseline_test": regression_metrics(y_test, baseline_test),
        "residuals": residual_summary(y_test, test_predictions),
        "prediction_interval": split_conformal_summary(
            y_validation,
            validation_predictions,
            y_test,
            test_predictions,
        ),
    }
    metrics["validation"]["mae_vs_baseline"] = (
        metrics["validation"]["mae"] / metrics["baseline_validation"]["mae"]
    )

    model_path = run.run_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": matrices.feature_columns,
            "target_column": config.training.target_column,
        },
        model_path,
    )
    save_json(metrics, run.run_dir / "metrics.json")
    _write_predictions(
        test,
        test_predictions,
        run.run_dir / "test_predictions.csv",
        interval_radius=metrics["prediction_interval"]["interval_radius"],
    )
    model_card_path = _write_model_card(
        run.run_dir / "model_card.md",
        train,
        validation,
        test,
        matrices.feature_columns,
        metrics,
        config,
    )
    _write_training_profile(
        train,
        matrices.x_train,
        y_train,
        test_predictions,
        metrics,
        config,
    )
    tracker.log_metrics(run, _flat_metric_dict(metrics))
    tracker.log_artifact(run, model_path)
    tracker.log_artifact(run, model_card_path)

    return TrainingResult(
        run_id=run.run_id,
        run_dir=run.run_dir,
        model_path=model_path,
        metrics=metrics,
        feature_columns=matrices.feature_columns,
    )


@dataclass(frozen=True)
class SplitMatrices:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    feature_columns: list[str]


def _split_model_matrices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> SplitMatrices:
    combined = pd.concat(
        [
            train.assign(_split="train"),
            validation.assign(_split="validation"),
            test.assign(_split="test"),
        ],
        ignore_index=True,
    )
    matrix, feature_columns = model_matrix(combined)
    split = combined["_split"]
    return SplitMatrices(
        x_train=matrix[split == "train"].reset_index(drop=True),
        x_validation=matrix[split == "validation"].reset_index(drop=True),
        x_test=matrix[split == "test"].reset_index(drop=True),
        feature_columns=feature_columns,
    )


def _write_predictions(
    test: pd.DataFrame,
    predictions,
    path: Path,
    interval_radius: float | None = None,
) -> None:
    output = test[["date", "store_id", "sku_id", "units_sold"]].copy()
    output["prediction"] = predictions
    output["absolute_error"] = (output["prediction"] - output["units_sold"]).abs()
    if interval_radius is not None:
        output["prediction_lower"] = (output["prediction"] - interval_radius).clip(lower=0)
        output["prediction_upper"] = output["prediction"] + interval_radius
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def _write_model_card(
    path: Path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    metrics: dict,
    config: PipelineConfig,
) -> Path:
    validation_metrics = metrics["validation"]
    baseline_metrics = metrics["baseline_validation"]
    test_metrics = metrics["test"]
    train_start, train_end = train["date"].min().date(), train["date"].max().date()
    validation_start = validation["date"].min().date()
    validation_end = validation["date"].max().date()
    test_start, test_end = test["date"].min().date(), test["date"].max().date()
    card = f"""# Retail Demand Forecast Model

This run trains a `HistGradientBoostingRegressor` for store-SKU demand forecasting. The
pipeline builds calendar, lag, rolling-window, promotion, price, and categorical features,
then compares the model against a rolling naive forecast before registration.

## Data window

- Train: {len(train)} rows, {train_start} to {train_end}
- Validation: {len(validation)} rows, {validation_start} to {validation_end}
- Test: {len(test)} rows, {test_start} to {test_end}
- Features: {len(feature_columns)}

## Quality gates

- Minimum validation R2: {config.training.min_validation_r2:.3f}
- Max candidate/baseline MAE ratio: {config.training.max_champion_mae_ratio:.3f}
- Validation MAE: {validation_metrics["mae"]:.3f}
- Baseline validation MAE: {baseline_metrics["mae"]:.3f}
- Validation MAE ratio: {validation_metrics["mae_vs_baseline"]:.3f}
- Test WAPE: {test_metrics["wape"]:.3f}

## Operational notes

The training profile saved with this run becomes the reference distribution for production
monitoring. Prediction drift, segment error, and MAE degradation are checked before the
next retraining decision.
"""
    path.write_text(card, encoding="utf-8")
    return path


def _write_training_profile(
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train,
    reference_predictions,
    metrics: dict,
    config: PipelineConfig,
) -> None:
    profile = {
        "row_count": len(train),
        "min_date": train["date"].min().date().isoformat(),
        "max_date": train["date"].max().date().isoformat(),
        "target": {
            "mean": float(pd.Series(y_train).mean()),
            "std": float(pd.Series(y_train).std()),
        },
        "prediction_reference": _reference_distribution(reference_predictions),
        "features": {
            column: {
                "mean": float(x_train[column].mean()),
                "std": float(x_train[column].std()),
            }
            for column in x_train.columns
        },
        "reference_metrics": metrics["test"],
    }
    save_json(profile, config.monitoring.baseline_profile_path)


def _reference_distribution(values, bins: int = 10) -> dict:
    data = np.asarray(values, dtype=float)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("reference predictions must be non-empty and finite")
    edges = np.quantile(data, np.linspace(0.0, 1.0, bins + 1))
    edges = _strictly_increasing_edges(edges)
    counts, _ = np.histogram(data, bins=edges)
    expected_share = counts / counts.sum()
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "p10": float(np.percentile(data, 10)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "bin_edges": [float(edge) for edge in edges],
        "expected_share": [float(value) for value in expected_share],
    }


def _strictly_increasing_edges(edges: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(edges, dtype=float).copy()
    for index in range(1, len(adjusted)):
        if adjusted[index] <= adjusted[index - 1]:
            adjusted[index] = adjusted[index - 1] + 1e-6
    adjusted[0] -= 1e-6
    adjusted[-1] += 1e-6
    return adjusted


def _flat_metric_dict(metrics: dict) -> dict[str, float]:
    flat = {}
    for group, values in metrics.items():
        for name, value in values.items():
            if isinstance(value, (int, float)):
                flat[f"{group}_{name}"] = float(value)
    return flat
