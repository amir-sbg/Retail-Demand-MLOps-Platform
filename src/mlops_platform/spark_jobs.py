from __future__ import annotations

from pathlib import Path


def build_features_with_spark(
    silver_path: Path,
    output_path: Path,
    output_format: str = "delta",
) -> Path:
    try:
        from pyspark.sql import SparkSession, Window
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise RuntimeError(
            "Spark feature jobs require the optional spark dependencies. "
            "Install with: pip install -r requirements-spark.txt"
        ) from exc

    if output_format not in {"delta", "parquet"}:
        raise ValueError("output_format must be delta or parquet")

    spark = (
        SparkSession.builder.appName("retail-demand-feature-job")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    try:
        frame = spark.read.option("header", True).option("inferSchema", True).csv(str(silver_path))
        frame = frame.withColumn("date", F.to_date("date"))
        by_entity = Window.partitionBy("store_id", "sku_id").orderBy("date")
        rolling_7 = by_entity.rowsBetween(-7, -1)
        rolling_14 = by_entity.rowsBetween(-14, -1)

        features = (
            frame.withColumn("lag_1_units", F.lag("units_sold", 1).over(by_entity))
            .withColumn("lag_7_units", F.lag("units_sold", 7).over(by_entity))
            .withColumn("rolling_7_mean_units", F.avg("units_sold").over(rolling_7))
            .withColumn("rolling_14_mean_units", F.avg("units_sold").over(rolling_14))
            .withColumn("rolling_7_std_units", F.stddev("units_sold").over(rolling_7))
            .withColumn("day_of_week", F.dayofweek("date") - F.lit(1))
            .withColumn("month", F.month("date"))
            .withColumn("day_sin", F.sin(2.0 * 3.141592653589793 * F.col("day_of_week") / 7.0))
            .withColumn("day_cos", F.cos(2.0 * 3.141592653589793 * F.col("day_of_week") / 7.0))
            .withColumn("month_sin", F.sin(2.0 * 3.141592653589793 * F.col("month") / 12.0))
            .withColumn("month_cos", F.cos(2.0 * 3.141592653589793 * F.col("month") / 12.0))
            .withColumn("entity_id", F.concat_ws("::", "store_id", "sku_id"))
            .dropna()
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.write.mode("overwrite").format(output_format).save(str(output_path))
        return output_path
    finally:
        spark.stop()
