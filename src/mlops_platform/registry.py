from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from mlops_platform.config import PipelineConfig
from mlops_platform.training import TrainingResult


@dataclass(frozen=True)
class RegistryRecord:
    version: str
    stage: str
    model_path: Path
    metadata_path: Path
    promoted: bool


def register_candidate(result: TrainingResult, config: PipelineConfig) -> RegistryRecord:
    registry_dir = config.tracking.registry_dir
    version_dir = registry_dir / "versions" / result.run_id
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.joblib"
    shutil.copy2(result.model_path, model_path)

    champion_metrics = _load_champion_validation_metrics(registry_dir)
    should_promote, reason = promotion_decision(
        result.metrics,
        config,
        champion_metrics=champion_metrics,
    )
    metadata = {
        "version": result.run_id,
        "stage": "champion" if should_promote else "candidate",
        "source_run": str(result.run_dir),
        "model_path": str(model_path),
        "metrics": result.metrics,
        "feature_columns": result.feature_columns,
        "promotion_reason": reason,
    }
    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    if should_promote:
        champion_dir = registry_dir / "champion"
        champion_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, champion_dir / "model.joblib")
        shutil.copy2(metadata_path, champion_dir / "metadata.json")
        _write_registry_index(registry_dir, result.run_id, metadata)

    return RegistryRecord(
        version=result.run_id,
        stage=metadata["stage"],
        model_path=model_path,
        metadata_path=metadata_path,
        promoted=should_promote,
    )


def load_champion(registry_dir: Path) -> tuple[object, dict]:
    import joblib

    model_path = registry_dir / "champion" / "model.joblib"
    metadata_path = registry_dir / "champion" / "metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("no champion model is registered yet")
    return joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8"))


def promotion_decision(
    metrics: dict,
    config: PipelineConfig,
    champion_metrics: dict | None = None,
) -> tuple[bool, str]:
    validation = metrics["validation"]
    baseline = metrics["baseline_validation"]
    if validation["r2"] < config.training.min_validation_r2:
        return False, "validation r2 is below the minimum gate"
    if validation["mae"] > baseline["mae"] * config.training.max_champion_mae_ratio:
        return False, "candidate does not improve enough over the rolling baseline"
    if champion_metrics is not None:
        champion_mae = float(champion_metrics.get("mae", float("inf")))
        allowed_mae = champion_mae * config.training.max_champion_regression_ratio
        if validation["mae"] > allowed_mae:
            return False, "candidate regresses beyond the allowed champion MAE tolerance"
    return True, "candidate passed validation quality gates"


def _load_champion_validation_metrics(registry_dir: Path) -> dict | None:
    metadata_path = registry_dir / "champion" / "metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {}).get("validation")
    return metrics if isinstance(metrics, dict) else None


def _write_registry_index(registry_dir: Path, champion_version: str, metadata: dict) -> None:
    index = {
        "champion_version": champion_version,
        "champion_stage": "production",
        "champion_metrics": metadata["metrics"]["validation"],
        "model_path": metadata["model_path"],
    }
    (registry_dir / "registry.json").write_text(json.dumps(index, indent=2) + "\n")
