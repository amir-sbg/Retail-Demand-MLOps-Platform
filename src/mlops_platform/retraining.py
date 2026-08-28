from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mlops_platform.metrics import save_json


def build_retraining_plan(
    drift_report_path: Path,
    output_path: Path,
    retrain_command: str = "python -m mlops_platform.cli run-all",
) -> dict:
    if not drift_report_path.exists():
        raise FileNotFoundError(f"drift report not found: {drift_report_path}")
    report = json.loads(drift_report_path.read_text(encoding="utf-8"))
    triggered = bool(report.get("retrain_recommended", False))
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "triggered": triggered,
        "status": "ready_to_retrain" if triggered else "no_action",
        "reason": report.get("reason", "no reason recorded"),
        "retrain_command": retrain_command if triggered else "",
        "source_report": str(drift_report_path),
    }
    save_json(plan, output_path)
    return plan
