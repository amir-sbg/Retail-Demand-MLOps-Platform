from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlops_platform.config import DataConfig


RAW_COLUMNS = (
    "date",
    "store_id",
    "sku_id",
    "category",
    "price",
    "promotion",
    "stockout",
    "temperature",
    "units_sold",
)


def generate_raw_demand(config: DataConfig) -> pd.DataFrame:
    config.validate()
    rng = np.random.default_rng(config.seed)
    dates = pd.date_range("2025-01-01", periods=config.n_days, freq="D")
    categories = np.array(["grocery", "home", "personal_care", "seasonal"])
    rows = []

    store_effect = rng.normal(0.0, 4.0, size=config.n_stores)
    sku_base = rng.uniform(18.0, 85.0, size=config.n_skus)
    sku_price = rng.uniform(4.0, 25.0, size=config.n_skus)
    sku_category = rng.choice(categories, size=config.n_skus)

    for day_index, date in enumerate(dates):
        weekly = 6.0 * np.sin(2.0 * np.pi * day_index / 7.0)
        yearly = 8.0 * np.sin(2.0 * np.pi * day_index / 365.0)
        temperature = 55.0 + 20.0 * np.sin(2.0 * np.pi * day_index / 365.0) + rng.normal(0, 3)
        for store in range(config.n_stores):
            for sku in range(config.n_skus):
                promotion = int(rng.random() < 0.16)
                stockout = int(rng.random() < 0.035)
                price_noise = rng.normal(0.0, 0.35)
                price = max(0.99, sku_price[sku] - 0.75 * promotion + price_noise)
                demand = (
                    sku_base[sku]
                    + store_effect[store]
                    + weekly
                    + yearly
                    + 7.5 * promotion
                    - 1.35 * price
                    + 0.08 * temperature
                    + rng.normal(0.0, 4.5)
                )
                if sku_category[sku] == "seasonal":
                    demand += 5.0 * np.sin(2.0 * np.pi * day_index / 90.0)
                if stockout:
                    demand *= rng.uniform(0.20, 0.55)

                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "store_id": f"store_{store:02d}",
                        "sku_id": f"sku_{sku:03d}",
                        "category": str(sku_category[sku]),
                        "price": round(float(price), 2),
                        "promotion": promotion,
                        "stockout": stockout,
                        "temperature": round(float(temperature), 2),
                        "units_sold": max(0, int(round(demand))),
                    }
                )

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def write_raw_demand(config: DataConfig) -> Path:
    frame = generate_raw_demand(config)
    config.raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.raw_path, index=False)
    return config.raw_path


def read_raw_demand(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"raw demand data is missing columns: {missing}")
    frame = frame.loc[:, RAW_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty:
        raise ValueError("raw demand data is empty")
    return frame
