from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlops_platform.config import DataConfig


FEATURE_COLUMNS = (
    "price",
    "promotion",
    "stockout",
    "temperature",
    "is_weekend",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "lag_1_units",
    "lag_7_units",
    "rolling_7_mean_units",
    "rolling_14_mean_units",
    "rolling_7_std_units",
    "price_vs_sku_avg",
    "sku_avg_units",
    "store_avg_units",
)

CATEGORICAL_COLUMNS = ("store_id", "sku_id", "category")


def build_feature_table(config: DataConfig) -> Path:
    silver = pd.read_csv(config.silver_path, parse_dates=["date"])
    features = make_features(silver)
    config.feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.feature_path, index=False)
    return config.feature_path


def make_features(silver: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "store_id",
        "sku_id",
        "category",
        "price",
        "promotion",
        "stockout",
        "temperature",
        "is_weekend",
        "day_of_week",
        "month",
        "units_sold",
    }
    missing = sorted(required - set(silver.columns))
    if missing:
        raise ValueError(f"silver table is missing columns: {missing}")

    frame = silver.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    group = frame.groupby(["store_id", "sku_id"], sort=False)["units_sold"]

    frame["lag_1_units"] = group.shift(1)
    frame["lag_7_units"] = group.shift(7)
    frame["rolling_7_mean_units"] = group.shift(1).rolling(7, min_periods=3).mean()
    frame["rolling_14_mean_units"] = group.shift(1).rolling(14, min_periods=5).mean()
    frame["rolling_7_std_units"] = group.shift(1).rolling(7, min_periods=3).std()
    frame["day_sin"] = np.sin(2.0 * np.pi * frame["day_of_week"] / 7.0)
    frame["day_cos"] = np.cos(2.0 * np.pi * frame["day_of_week"] / 7.0)
    frame["month_sin"] = np.sin(2.0 * np.pi * frame["month"] / 12.0)
    frame["month_cos"] = np.cos(2.0 * np.pi * frame["month"] / 12.0)

    sku_avg_price = frame.groupby("sku_id")["price"].transform("mean")
    frame["price_vs_sku_avg"] = frame["price"] - sku_avg_price
    frame["sku_avg_units"] = group.transform("mean")
    frame["store_avg_units"] = frame.groupby("store_id")["units_sold"].transform("mean")
    frame["entity_id"] = frame["store_id"] + "::" + frame["sku_id"]

    feature_frame = frame.dropna(subset=FEATURE_COLUMNS + ("units_sold",)).reset_index(drop=True)
    if feature_frame.empty:
        raise ValueError("feature table is empty after lag creation")
    return feature_frame


def model_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    numeric = frame.loc[:, FEATURE_COLUMNS].copy()
    encoded = pd.get_dummies(
        frame.loc[:, CATEGORICAL_COLUMNS],
        prefix=list(CATEGORICAL_COLUMNS),
        dtype=float,
    )
    matrix = pd.concat([numeric, encoded], axis=1)
    return matrix, list(matrix.columns)
