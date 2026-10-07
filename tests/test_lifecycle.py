from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_platform.config import DataConfig, MonitoringConfig, PipelineConfig, TrackingConfig, TrainingConfig
from mlops_platform.data import write_raw_demand
from mlops_platform.features import build_feature_table
from mlops_platform.inference import batch_predict, prediction_output_summary
from mlops_platform.lake import build_bronze_table, build_silver_table
from mlops_platform.metrics import (
    regression_metrics,
    rolling_origin_splits,
    split_conformal_summary,
    time_splits,
)
from mlops_platform.monitoring import (
    feature_mean_shift_report,
    monitor_predictions,
    monitoring_alerts,
    psi_from_bins,
    psi_from_stats,
    segment_quality_report,
)
from mlops_platform.registry import promotion_decision, register_candidate
from mlops_platform.retraining import build_retraining_plan, retraining_next_steps, retraining_priority
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
    batch_metrics = json.loads((tmp_path / "reports" / "batch_metrics.json").read_text(encoding="utf-8"))
    assert batch_metrics["prediction_summary"]["rows"] == len(predictions)
    assert batch_metrics["prediction_summary"]["has_actuals"]


def test_prediction_output_summary_reports_serving_shape() -> None:
    predictions = pd.DataFrame(
        {
            "store_id": ["s1", "s1", "s2"],
            "sku_id": ["a", "b", "a"],
            "predicted_units": [10.0, 20.0, 40.0],
            "actual_units": [12.0, 18.0, 39.0],
        }
    )

    summary = prediction_output_summary(predictions)

    assert summary["rows"] == 3
    assert summary["unique_stores"] == 2
    assert summary["unique_skus"] == 2
    assert summary["has_actuals"]
    assert summary["p90_predicted_units"] == pytest.approx(36.0)


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


def test_rolling_origin_splits_expand_training_window() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40),
            "value": np.arange(40),
        }
    )

    windows = rolling_origin_splits(frame, horizon_days=5, folds=3)

    assert [len(train) for train, _ in windows] == [25, 30, 35]
    assert all(len(validation) == 5 for _, validation in windows)
    assert all(train["date"].max() < validation["date"].min() for train, validation in windows)


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


def test_split_conformal_interval_uses_validation_errors() -> None:
    report = split_conformal_summary(
        [10.0, 20.0, 30.0, 40.0],
        [10.0, 21.0, 32.0, 44.0],
        [15.0, 25.0, 35.0],
        [16.0, 28.0, 40.0],
        coverage=0.75,
    )

    assert report["interval_radius"] == 4.0
    assert report["observed_coverage"] == pytest.approx(2 / 3)


def test_promotion_decision_applies_quality_gates() -> None:
    config = _tmp_config(Path("/tmp/mlops-test"))
    metrics = {
        "validation": {"r2": 0.8, "mae": 2.0},
        "baseline_validation": {"mae": 4.0},
    }

    promoted, reason = promotion_decision(metrics, config)

    assert promoted
    assert "passed" in reason


def test_promotion_decision_blocks_champion_regression() -> None:
    config = _tmp_config(Path("/tmp/mlops-test"))
    metrics = {
        "validation": {"r2": 0.8, "mae": 2.0},
        "baseline_validation": {"mae": 4.0},
    }

    promoted, reason = promotion_decision(
        metrics,
        config,
        champion_metrics={"mae": 1.5},
    )

    assert not promoted
    assert "champion" in reason


def test_psi_increases_for_shifted_distribution() -> None:
    near = psi_from_stats(10.0, 1.0, np.linspace(8.5, 11.5, 100))
    shifted = psi_from_stats(10.0, 1.0, np.linspace(15.0, 18.0, 100))

    assert shifted > near


def test_psi_from_bins_uses_stored_reference_shares() -> None:
    near = psi_from_bins([0.0, 5.0, 10.0], [0.5, 0.5], np.array([1.0, 2.0, 8.0, 9.0]))
    shifted = psi_from_bins([0.0, 5.0, 10.0], [0.5, 0.5], np.array([8.0, 9.0, 9.5, 9.8]))

    assert shifted > near


def test_psi_from_bins_rejects_invalid_reference_shares() -> None:
    with pytest.raises(ValueError, match="positive sum"):
        psi_from_bins([0.0, 1.0, 2.0], [0.0, 0.0], np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="finite"):
        psi_from_bins([0.0, 1.0, 2.0], [np.nan, 1.0], np.array([0.5, 1.5]))


def test_retraining_plan_requires_existing_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_retraining_plan(tmp_path / "missing.json", tmp_path / "plan.json")


def test_retraining_plan_includes_priority_and_next_steps(tmp_path: Path) -> None:
    drift_report = tmp_path / "drift.json"
    drift_report.write_text(
        json.dumps(
            {
                "retrain_recommended": True,
                "reason": "current MAE degraded against the training reference",
                "alerts": [
                    {"type": "error_degradation", "severity": "high"},
                    {"type": "forecast_bias", "severity": "medium"},
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = build_retraining_plan(drift_report, tmp_path / "plan.json")

    assert plan["priority"] == "high"
    assert plan["alert_types"] == ["error_degradation", "forecast_bias"]
    assert any("retrain" in step for step in plan["next_steps"])


def test_retraining_helpers_map_alerts_to_actions() -> None:
    alerts = [
        {"type": "segment_error", "severity": "medium"},
        {"type": "prediction_drift", "severity": "high"},
    ]

    assert retraining_priority(alerts) == "high"
    steps = retraining_next_steps(alerts)
    assert any("distribution" in step for step in steps)
    assert any("segment" in step for step in steps)


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
    assert "error_degradation" in {alert["type"] for alert in report["alerts"]}


def test_monitoring_alerts_include_drift_and_segment_details() -> None:
    alerts = monitoring_alerts(
        psi=0.35,
        current_mae=2.5,
        reference_mae=1.0,
        segments=[
            {
                "segment": "category=frozen",
                "rows": 12,
                "mae": 2.3,
                "wape": 0.2,
                "bias": -0.3,
            }
        ],
        config=MonitoringConfig(psi_threshold=0.2, mae_degradation_ratio=1.5),
        current_bias=-0.8,
    )

    assert [alert["type"] for alert in alerts] == [
        "prediction_drift",
        "error_degradation",
        "forecast_bias",
        "segment_error",
    ]
    assert alerts[-1]["segment"] == "category=frozen"


def test_monitoring_alerts_catches_directional_bias() -> None:
    alerts = monitoring_alerts(
        psi=0.01,
        current_mae=1.1,
        reference_mae=1.0,
        segments=[],
        config=MonitoringConfig(bias_alert_ratio=0.4),
        current_bias=0.6,
    )

    assert [alert["type"] for alert in alerts] == ["forecast_bias"]


def test_feature_mean_shift_report_ranks_changed_inputs() -> None:
    reference = {
        "price": {"mean": 10.0, "std": 2.0},
        "promotion": {"mean": 0.2, "std": 0.4},
    }
    observed = pd.DataFrame(
        {
            "price": [20.0, 22.0],
            "promotion": [0.0, 1.0],
        }
    )

    rows = feature_mean_shift_report(reference, observed)

    assert rows[0]["feature"] == "price"
    assert rows[0]["standardized_shift"] == pytest.approx(5.5)


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
