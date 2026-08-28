from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mlops_platform.config import TrackingConfig


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    mlflow_run_id: str | None = None


class ExperimentTracker:
    def __init__(self, config: TrackingConfig, use_mlflow: bool = True) -> None:
        config.validate()
        self.config = config
        self.mlflow = _load_mlflow() if use_mlflow else None
        if self.mlflow is not None:
            self.mlflow.set_tracking_uri(str(config.tracking_dir / "mlruns"))
            self.mlflow.set_experiment(config.experiment_name)

    def start_run(self, params: dict) -> RunContext:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
        run_dir = self.config.tracking_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": run_id,
            "experiment": self.config.experiment_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "params": params,
        }
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        return RunContext(run_id=run_id, run_dir=run_dir)

    def log_params(self, context: RunContext, params: dict) -> None:
        _merge_json(context.run_dir / "params.json", params)
        with self._mlflow_run(context):
            if self.mlflow is not None:
                self.mlflow.log_params(_flat_params(params))

    def log_metrics(self, context: RunContext, metrics: dict, prefix: str = "") -> None:
        path = context.run_dir / f"{prefix or 'metrics'}.json"
        path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        with self._mlflow_run(context):
            if self.mlflow is not None:
                self.mlflow.log_metrics(
                    {f"{prefix}_{key}" if prefix else key: float(value) for key, value in metrics.items()}
                )

    def log_artifact(self, context: RunContext, path: Path) -> None:
        with self._mlflow_run(context):
            if self.mlflow is not None and path.exists():
                self.mlflow.log_artifact(str(path))

    def _mlflow_run(self, context: RunContext):
        if self.mlflow is None:
            return nullcontext()
        return self.mlflow.start_run(run_name=context.run_id, nested=True)


def _load_mlflow():
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _flat_params(params: dict, prefix: str = "") -> dict[str, str | int | float | bool]:
    flat = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flat_params(value, name))
        elif isinstance(value, (str, int, float, bool)):
            flat[name] = value
        else:
            flat[name] = str(value)
    return flat


def _merge_json(path: Path, values: dict) -> None:
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
