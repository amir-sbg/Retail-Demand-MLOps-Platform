from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_platform.config import DataConfig, MonitoringConfig, PipelineConfig, TrackingConfig, TrainingConfig
from mlops_platform.data import write_raw_demand
from mlops_platform.features import build_feature_table
from mlops_platform.inference import batch_predict
from mlops_platform.lake import build_bronze_table, build_silver_table
from mlops_platform.metrics import regression_metrics, time_splits
from mlops_platform.monitoring import monitor_predictions, psi_from_bins, psi_from_stats, segment_quality_report
from mlops_platform.registry import promotion_decision, register_candidate
from mlops_platform.retraining import build_retraining_plan
from mlops_platform.training import train_forecaster


def _tmp_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            raw_path=tmp_path / "raw" / "demand.csv",
            bronze_path=tmp_path / "bronze" / "demand.csv",
            silver_path=tmp_path / "silver" / "demand.csv",
            feature_path=tmp_path / "gold" / "features.csv",
            n_days=70,
            n_stores=2,
            n_skus=3,
            seed=11,
        ),
        training=TrainingConfig(
            validation_days=10,
            test_days=10,
            min_validation_r2=-1.0,
            max_champion_mae_ratio=10.0,
        ),
        tracking=TrackingConfig(
            tracking_dir=tmp_path / "runs",
            registry_dir=tmp_path / "registry",
        ),
        monitoring=MonitoringConfig(
            baseline_profile_path=tmp_path / "reports" / "training_profile.json",
            prediction_log_path=tmp_path / "predictions" / "batch.csv",
            drift_report_path=tmp_path / "reports" / "drift.json",
        ),
    )


def test_end_to_end_lifecycle_runs_on_temp_paths(tmp_path: Path) -> None:
    config = _tmp_config(tmp_path)

    write_raw_demand(config.data)
    build_bronze_table(config.data)
    build_silver_table(config.data)
    build_feature_table(config.data)
    result = train_forecaster(config)
    record = register_candidate(result, config)
    predictions = batch_predict(
        config.data.feature_path,
        config.tracking.registry_dir,
        config.monitoring.prediction_log_path,
        metrics_path=tmp_path / "reports" / "batch_metrics.json",
    )
    drift = monitor_predictions(
        config.monitoring.baseline_profile_path,
        config.monitoring.prediction_log_path,
        config.monitoring.drift_report_path,
        config.monitoring,
    )
    plan = build_retraining_plan(
        config.monitoring.drift_report_path,
        tmp_path / "reports" / "retraining_plan.json",
    )

    assert record.promoted
    assert result.model_path.exists()
    model_card = result.run_dir / "model_card.md"
    assert model_card.exists()
    assert "HistGradientBoostingRegressor" in model_card.read_text(encoding="utf-8")
    assert (config.tracking.registry_dir / "champion" / "model.joblib").exists()
    assert len(predictions) > 0
    assert "retrain_recommended" in drift
    assert plan["status"] in {"ready_to_retrain", "no_action"}


def test_time_splits_keep_recent_rows_for_test() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=30),
            "value": np.arange(30),
        }
    )

    train, validation, test = time_splits(frame, validation_days=5, test_days=5)

    assert train["date"].max() < validation["date"].min()
    assert validation["date"].max() < test["date"].min()
    assert len(test) == 5


def test_regression_metrics_reject_bad_arrays() -> None:
    with pytest.raises(ValueError, match="same shape"):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_regression_metrics_include_demand_forecast_diagnostics() -> None:
    metrics = regression_metrics(
        np.array([10.0, 20.0, 30.0]),
        np.array([12.0, 16.0, 33.0]),
    )

    assert metrics["wape"] == pytest.approx(9.0 / 60.0)
    assert metrics["smape"] > 0.0
    assert metrics["under_forecast_rate"] == pytest.approx(1 / 3)
    assert metrics["over_forecast_rate"] == pytest.approx(2 / 3)


def test_promotion_decision_applies_quality_gates() -> None:
    config = _tmp_config(Path("/tmp/mlops-test"))
    metrics = {
        "validation": {"r2": 0.8, "mae": 2.0},
        "baseline_validation": {"mae": 4.0},
    }

    promoted, reason = promotion_decision(metrics, config)

    assert promoted
    assert "passed" in reason


def test_psi_increases_for_shifted_distribution() -> None:
    near = psi_from_stats(10.0, 1.0, np.linspace(8.5, 11.5, 100))
    shifted = psi_from_stats(10.0, 1.0, np.linspace(15.0, 18.0, 100))

    assert shifted > near


def test_psi_from_bins_uses_stored_reference_shares() -> None:
    near = psi_from_bins([0.0, 5.0, 10.0], [0.5, 0.5], np.array([1.0, 2.0, 8.0, 9.0]))
    shifted = psi_from_bins([0.0, 5.0, 10.0], [0.5, 0.5], np.array([8.0, 9.0, 9.5, 9.8]))

    assert shifted > near


def test_retraining_plan_requires_existing_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_retraining_plan(tmp_path / "missing.json", tmp_path / "plan.json")


def test_monitoring_recommends_retraining_on_error_degradation(tmp_path: Path) -> None:
    baseline = {
        "target": {"mean": 10.0, "std": 2.0},
        "reference_metrics": {"mae": 1.0},
    }
    baseline_path = tmp_path / "baseline.json"
    predictions_path = tmp_path / "predictions.csv"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    pd.DataFrame(
        {
            "predicted_units": [10.0, 11.0, 12.0],
            "actual_units": [18.0, 19.0, 20.0],
        }
    ).to_csv(predictions_path, index=False)

    report = monitor_predictions(
        baseline_path,
        predictions_path,
        tmp_path / "drift.json",
        MonitoringConfig(drift_report_path=tmp_path / "drift.json"),
    )

    assert report["retrain_recommended"]


def test_segment_quality_report_ranks_segments_by_error() -> None:
    predictions = pd.DataFrame(
        {
            "store_id": ["s1", "s1", "s2", "s2"],
            "category": ["grocery", "grocery", "grocery", "grocery"],
            "predicted_units": [10.0, 11.0, 25.0, 30.0],
            "actual_units": [10.0, 12.0, 10.0, 10.0],
        }
    )

    rows = segment_quality_report(predictions, segment_columns=("store_id",), min_rows=2)

    assert rows[0]["segment"] == "store_id=s2"
    assert rows[0]["mae"] > rows[1]["mae"]
