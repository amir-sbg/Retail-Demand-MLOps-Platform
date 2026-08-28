from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlops_platform.config import DataConfig
from mlops_platform.data import RAW_COLUMNS, read_raw_demand


def build_bronze_table(config: DataConfig) -> Path:
    raw = read_raw_demand(config.raw_path)
    bronze = raw.drop_duplicates(subset=["date", "store_id", "sku_id"]).copy()
    bronze = bronze.sort_values(["date", "store_id", "sku_id"]).reset_index(drop=True)
    _write_csv_with_manifest(
        bronze,
        config.bronze_path,
        layer="bronze",
        source=str(config.raw_path),
        checks={
            "rows": len(bronze),
            "columns": list(bronze.columns),
            "duplicates_removed": int(len(raw) - len(bronze)),
        },
    )
    return config.bronze_path


def build_silver_table(config: DataConfig) -> Path:
    bronze = pd.read_csv(config.bronze_path, parse_dates=["date"])
    missing = sorted(set(RAW_COLUMNS) - set(bronze.columns))
    if missing:
        raise ValueError(f"bronze table is missing columns: {missing}")

    silver = bronze.copy()
    silver["units_sold"] = silver["units_sold"].clip(lower=0)
    silver["price"] = silver["price"].clip(lower=0.01)
    silver["revenue"] = silver["price"] * silver["units_sold"]
    silver["is_weekend"] = silver["date"].dt.dayofweek.isin([5, 6]).astype(int)
    silver["day_of_week"] = silver["date"].dt.dayofweek
    silver["month"] = silver["date"].dt.month
    silver["week_of_year"] = silver["date"].dt.isocalendar().week.astype(int)
    silver = silver.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)

    _write_csv_with_manifest(
        silver,
        config.silver_path,
        layer="silver",
        source=str(config.bronze_path),
        checks={
            "rows": len(silver),
            "min_date": silver["date"].min().date().isoformat(),
            "max_date": silver["date"].max().date().isoformat(),
            "negative_units": int((silver["units_sold"] < 0).sum()),
        },
    )
    return config.silver_path


def _write_csv_with_manifest(
    frame: pd.DataFrame,
    path: Path,
    layer: str,
    source: str,
    checks: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    manifest = {
        "layer": layer,
        "source": source,
        "path": str(path),
        "checks": checks,
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
