"""Spark session factory and Parquet ZSTD write helper.

Centralizes SparkSession creation so every pipeline stage and test shares
loopback driver binding, settings-driven configuration, and ZSTD compression
without repeating boilerplate or setting `spark.jars.packages` at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from src.config import Settings, get_settings


def build_spark_session(
    settings: Settings | None = None,
    *,
    shuffle_partitions: int | None = None,
) -> SparkSession:
    """Create or return a configured SparkSession from ``Settings``."""
    cfg = settings or get_settings()
    builder = (
        SparkSession.builder.appName(cfg.spark_app_name)
        .master(cfg.spark_master)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
    )
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def stop(spark: SparkSession | None) -> None:
    """Stop a SparkSession cleanly if one is active."""
    if spark is not None:
        spark.stop()


def write_parquet_zstd(
    df: DataFrame,
    path: str | Path,
    *,
    mode: str = "overwrite",
    partition_by: Sequence[str] | str | None = None,
) -> None:
    """Write a DataFrame as ZSTD-compressed Parquet, optionally partitioned."""
    writer = df.write.mode(mode).option("compression", "zstd")
    if partition_by is not None:
        cols = [partition_by] if isinstance(partition_by, str) else list(partition_by)
        if cols:
            writer = writer.partitionBy(*cols)
    writer.parquet(str(path))
