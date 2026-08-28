from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_platform.config import DataConfig, PipelineConfig
from mlops_platform.data import generate_raw_demand, read_raw_demand, write_raw_demand
from mlops_platform.features import FEATURE_COLUMNS, make_features, model_matrix
from mlops_platform.lake import build_bronze_table, build_silver_table


def test_pipeline_config_serializes_paths() -> None:
    payload = PipelineConfig().to_dict()

    assert payload["data"]["raw_path"] == "data/raw/demand.csv"
    assert payload["tracking"]["experiment_name"] == "retail-demand-forecasting"


def test_raw_demand_generation_is_deterministic() -> None:
    config = DataConfig(n_days=45, n_stores=2, n_skus=3, seed=7)

    first = generate_raw_demand(config)
    second = generate_raw_demand(config)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 45 * 2 * 3
    assert {"promotion", "stockout", "temperature", "units_sold"}.issubset(first.columns)


def test_lake_layers_write_validated_tables(tmp_path) -> None:
    config = DataConfig(
        raw_path=tmp_path / "raw.csv",
        bronze_path=tmp_path / "bronze" / "demand.csv",
        silver_path=tmp_path / "silver" / "demand.csv",
        feature_path=tmp_path / "gold" / "features.csv",
        n_days=50,
        n_stores=2,
        n_skus=2,
    )

    write_raw_demand(config)
    build_bronze_table(config)
    build_silver_table(config)

    raw = read_raw_demand(config.raw_path)
    silver = pd.read_csv(config.silver_path)

    assert len(raw) == len(silver)
    assert "revenue" in silver.columns
    assert config.bronze_path.with_suffix(".manifest.json").exists()


def test_feature_engineering_keeps_windows_inside_each_series() -> None:
    dates = pd.date_range("2025-01-01", periods=10)
    rows = []
    for sku, base in [("a", 10), ("b", 100)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "store_id": "s1",
                    "sku_id": sku,
                    "category": "grocery",
                    "price": 5.0,
                    "promotion": 0,
                    "stockout": 0,
                    "temperature": 60.0,
                    "is_weekend": int(date.dayofweek in [5, 6]),
                    "day_of_week": date.dayofweek,
                    "month": date.month,
                    "units_sold": base + index,
                }
            )
    features = make_features(pd.DataFrame(rows))
    b_day_8 = features[(features["sku_id"] == "b") & (features["date"] == dates[8])].iloc[0]

    assert b_day_8["rolling_7_mean_units"] == pytest.approx(np.mean([101, 102, 103, 104, 105, 106, 107]))


def test_model_matrix_adds_numeric_and_categorical_columns() -> None:
    raw = generate_raw_demand(DataConfig(n_days=50, n_stores=1, n_skus=2))
    silver = raw.assign(
        revenue=raw["price"] * raw["units_sold"],
        is_weekend=pd.to_datetime(raw["date"]).dt.dayofweek.isin([5, 6]).astype(int),
        day_of_week=pd.to_datetime(raw["date"]).dt.dayofweek,
        month=pd.to_datetime(raw["date"]).dt.month,
        week_of_year=pd.to_datetime(raw["date"]).dt.isocalendar().week.astype(int),
    )
    features = make_features(silver)
    matrix, columns = model_matrix(features)

    assert set(FEATURE_COLUMNS).issubset(columns)
    assert any(column.startswith("sku_id_") for column in columns)
    assert len(matrix) == len(features)
