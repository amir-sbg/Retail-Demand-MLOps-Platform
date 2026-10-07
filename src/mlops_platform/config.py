from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path = Path("data/raw/demand.csv")
    bronze_path: Path = Path("data/bronze/demand_bronze.csv")
    silver_path: Path = Path("data/silver/demand_silver.csv")
    feature_path: Path = Path("data/gold/demand_features.csv")
    n_days: int = 180
    n_stores: int = 5
    n_skus: int = 8
    seed: int = 42

    def validate(self) -> None:
        if self.n_days < 45:
            raise ValueError("n_days must be at least 45")
        if self.n_stores < 1 or self.n_skus < 1:
            raise ValueError("n_stores and n_skus must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    target_column: str = "units_sold"
    validation_days: int = 21
    test_days: int = 21
    random_state: int = 42
    min_validation_r2: float = 0.35
    max_champion_mae_ratio: float = 1.05
    max_champion_regression_ratio: float = 1.02

    def validate(self) -> None:
        if self.validation_days < 7 or self.test_days < 7:
            raise ValueError("validation_days and test_days must be at least 7")
        if self.min_validation_r2 < -1.0:
            raise ValueError("min_validation_r2 is too low to be useful")
        if self.max_champion_mae_ratio <= 0:
            raise ValueError("max_champion_mae_ratio must be positive")
        if self.max_champion_regression_ratio < 1.0:
            raise ValueError("max_champion_regression_ratio must be at least 1")


@dataclass(frozen=True)
class TrackingConfig:
    tracking_dir: Path = Path("artifacts/runs")
    registry_dir: Path = Path("models/registry")
    experiment_name: str = "retail-demand-forecasting"

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")


@dataclass(frozen=True)
class MonitoringConfig:
    baseline_profile_path: Path = Path("reports/training_profile.json")
    prediction_log_path: Path = Path("predictions/batch_predictions.csv")
    drift_report_path: Path = Path("reports/drift_report.json")
    psi_threshold: float = 0.20
    mae_degradation_ratio: float = 1.20
    bias_alert_ratio: float = 0.50
    feature_mean_shift_threshold: float = 2.0

    def validate(self) -> None:
        if self.psi_threshold <= 0:
            raise ValueError("psi_threshold must be positive")
        if self.mae_degradation_ratio <= 1.0:
            raise ValueError("mae_degradation_ratio must be greater than 1")
        if self.bias_alert_ratio <= 0:
            raise ValueError("bias_alert_ratio must be positive")
        if self.feature_mean_shift_threshold <= 0:
            raise ValueError("feature_mean_shift_threshold must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def validate(self) -> None:
        self.data.validate()
        self.training.validate()
        self.tracking.validate()
        self.monitoring.validate()

    def to_dict(self) -> dict:
        payload = asdict(self)
        return _stringify_paths(payload)


def _stringify_paths(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value
