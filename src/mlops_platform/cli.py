from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlops_platform.config import DataConfig, PipelineConfig
from mlops_platform.data import write_raw_demand
from mlops_platform.features import build_feature_table
from mlops_platform.inference import batch_predict
from mlops_platform.lake import build_bronze_table, build_silver_table
from mlops_platform.monitoring import monitor_predictions
from mlops_platform.registry import register_candidate
from mlops_platform.retraining import build_retraining_plan
from mlops_platform.training import train_forecaster


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _config_from_args(args)

    if args.command == "version":
        from mlops_platform import __version__

        _print({"version": __version__})
    elif args.command == "make-data":
        _print({"raw_path": str(write_raw_demand(config.data))})
    elif args.command == "build-lake":
        bronze = build_bronze_table(config.data)
        silver = build_silver_table(config.data)
        _print({"bronze_path": str(bronze), "silver_path": str(silver)})
    elif args.command == "build-features":
        _print({"feature_path": str(build_feature_table(config.data))})
    elif args.command == "train":
        result = train_forecaster(config)
        record = register_candidate(result, config)
        _print({"run_id": result.run_id, "registry_stage": record.stage})
    elif args.command == "batch-predict":
        predictions = batch_predict(
            config.data.feature_path,
            config.tracking.registry_dir,
            config.monitoring.prediction_log_path,
            metrics_path=Path("reports/batch_metrics.json"),
        )
        _print({"prediction_path": str(config.monitoring.prediction_log_path), "rows": len(predictions)})
    elif args.command == "monitor":
        report = monitor_predictions(
            config.monitoring.baseline_profile_path,
            config.monitoring.prediction_log_path,
            config.monitoring.drift_report_path,
            config.monitoring,
        )
        _print(report)
    elif args.command == "retrain-plan":
        plan = build_retraining_plan(
            config.monitoring.drift_report_path,
            Path("reports/retraining_plan.json"),
        )
        _print(plan)
    elif args.command == "run-all":
        _run_all(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlops-demand")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "version",
        "make-data",
        "build-lake",
        "build-features",
        "train",
        "batch-predict",
        "monitor",
        "retrain-plan",
        "run-all",
    ):
        add = subcommands.add_parser(command)
        _add_common_args(add)
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-path", type=Path, default=DataConfig.raw_path)
    parser.add_argument("--bronze-path", type=Path, default=DataConfig.bronze_path)
    parser.add_argument("--silver-path", type=Path, default=DataConfig.silver_path)
    parser.add_argument("--feature-path", type=Path, default=DataConfig.feature_path)
    parser.add_argument("--n-days", type=int, default=DataConfig.n_days)
    parser.add_argument("--n-stores", type=int, default=DataConfig.n_stores)
    parser.add_argument("--n-skus", type=int, default=DataConfig.n_skus)
    parser.add_argument("--seed", type=int, default=DataConfig.seed)


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        data=DataConfig(
            raw_path=args.raw_path,
            bronze_path=args.bronze_path,
            silver_path=args.silver_path,
            feature_path=args.feature_path,
            n_days=args.n_days,
            n_stores=args.n_stores,
            n_skus=args.n_skus,
            seed=args.seed,
        )
    )


def _run_all(config: PipelineConfig) -> None:
    raw = write_raw_demand(config.data)
    bronze = build_bronze_table(config.data)
    silver = build_silver_table(config.data)
    feature_path = build_feature_table(config.data)
    result = train_forecaster(config)
    record = register_candidate(result, config)
    predictions = batch_predict(
        feature_path,
        config.tracking.registry_dir,
        config.monitoring.prediction_log_path,
        metrics_path=Path("reports/batch_metrics.json"),
    )
    drift = monitor_predictions(
        config.monitoring.baseline_profile_path,
        config.monitoring.prediction_log_path,
        config.monitoring.drift_report_path,
        config.monitoring,
    )
    _print(
        {
            "raw_path": str(raw),
            "bronze_path": str(bronze),
            "silver_path": str(silver),
            "feature_path": str(feature_path),
            "run_id": result.run_id,
            "registry_stage": record.stage,
            "predictions": len(predictions),
            "retrain_recommended": drift["retrain_recommended"],
        }
    )


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
