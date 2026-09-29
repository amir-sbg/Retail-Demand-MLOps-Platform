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
    alerts = list(report.get("alerts", []))
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "triggered": triggered,
        "status": "ready_to_retrain" if triggered else "no_action",
        "reason": report.get("reason", "no reason recorded"),
        "priority": retraining_priority(alerts),
        "alert_types": [str(alert.get("type", "unknown")) for alert in alerts],
        "next_steps": retraining_next_steps(alerts) if triggered else [],
        "retrain_command": retrain_command if triggered else "",
        "source_report": str(drift_report_path),
    }
    save_json(plan, output_path)
    return plan


def retraining_priority(alerts: list[dict]) -> str:
    severities = {str(alert.get("severity", "")).lower() for alert in alerts}
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    return "low"


def retraining_next_steps(alerts: list[dict]) -> list[str]:
    alert_types = {str(alert.get("type", "")) for alert in alerts}
    steps = []
    if "prediction_drift" in alert_types:
        steps.append("refresh the recent scoring window and compare demand distribution against training")
    if "error_degradation" in alert_types:
        steps.append("retrain the candidate model and compare it against the current champion gate")
    if "forecast_bias" in alert_types:
        steps.append("inspect systematic over/under-forecasting before promotion")
    if "segment_error" in alert_types:
        steps.append("review the worst store/category segments and add segment-specific diagnostics")
    return steps or ["review the drift report before starting a retraining run"]
