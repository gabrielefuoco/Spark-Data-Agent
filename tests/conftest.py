"""Shared Pytest fixtures for Spark and synthetic eRISK lakehouse tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from scripts.generate_dev_sample import generate_synthetic_sample
from src.config import Settings
from src.lakehouse.paths import LakehousePaths
from src.spark import build_spark_session, stop


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Session-scoped local SparkSession with 2 cores and 2 shuffle partitions."""
    settings = Settings(
        _env_file=None,
        spark_app_name="spark-data-agent-tests",
        spark_master="local[2]",
    )
    session = build_spark_session(settings, shuffle_partitions=2)
    yield session
    stop(session)


@pytest.fixture
def lakehouse_env(tmp_path: Path) -> tuple[Settings, LakehousePaths, Path]:
    """Isolated Settings, LakehousePaths, and small synthetic corpus under tmp_path."""
    storage_root = tmp_path / "data"
    settings = Settings(
        _env_file=None,
        spark_storage_uri=str(storage_root),
        spark_app_name="spark-data-agent-tests",
        spark_master="local[2]",
    )
    paths = LakehousePaths.from_settings(settings)
    sample_dir = generate_synthetic_sample(
        paths.sample_root,
        seed=42,
        subjects=10,
        posts_per_subject=8,
        include_broken=True,
    )
    return settings, paths, sample_dir
