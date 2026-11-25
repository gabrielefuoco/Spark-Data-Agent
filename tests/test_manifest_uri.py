"""Unit and MinIO integration tests for URI-aware manifest storage."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pyarrow.fs as pafs
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType
from src.lakehouse.manifest import (
    DatasetManifest,
    build_dataset_manifest,
    load_manifest,
    resolve_manifest_filesystem,
)
from src.lakehouse.paths import LakehousePaths
from src.lakehouse.storage import StorageConfigurationError
from tests.conftest import make_s3_settings
from tests.test_storage import _is_endpoint_reachable, _minio_endpoint_from_env


def test_resolve_manifest_filesystem_maps_s3a_uris() -> None:
    s3_settings = make_s3_settings("s3a://lakehouse/prefix")
    manifest_uri = LakehousePaths.from_settings(s3_settings).gold_dataset_manifest
    s3_fs, s3_fs_path = resolve_manifest_filesystem(manifest_uri, settings=s3_settings)
    assert isinstance(s3_fs, pafs.S3FileSystem)
    assert s3_fs_path == "lakehouse/prefix/lakehouse/gold/gold_dataset_manifest.json"

    with pytest.raises(StorageConfigurationError, match="Invalid S3A manifest URI"):
        resolve_manifest_filesystem("s3a://lakehouse", settings=s3_settings)

    missing_creds = make_s3_settings(
        "s3a://lakehouse", endpoint=None, access_key=None, secret_key=None
    )
    with pytest.raises(StorageConfigurationError, match="AWS_ACCESS_KEY_ID"):
        resolve_manifest_filesystem(manifest_uri, settings=missing_creds)


def test_load_manifest_reads_local_json_without_spark(
    tmp_path: Path,
    forbid_active_spark: Callable[[], None],
) -> None:
    forbid_active_spark()

    manifest_file = tmp_path / "gold_dataset_manifest.json"
    sample = DatasetManifest(
        total_posts=4,
        total_subjects=2,
        subjects_per_class={"0": 1, "1": 1},
        posts_per_class={"0": 2, "1": 2},
        min_date="2025-01-01T00:00:00",
        max_date="2025-01-02T00:00:00",
        quarantine_count=1,
        quarantine_by_reason={"EMPTY_TEXT": 1},
        generated_at="2025-01-03T00:00:00+00:00",
    )
    manifest_file.write_text(sample.model_dump_json(indent=2) + "\n", encoding="utf-8")

    assert load_manifest(manifest_file) == sample
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")


@pytest.mark.spark
@pytest.mark.skipif(
    not _is_endpoint_reachable(_minio_endpoint_from_env()),
    reason="MinIO S3-compatible endpoint is not reachable.",
)
def test_manifest_s3a_roundtrip_on_minio(
    spark: SparkSession,
    forbid_active_spark: Callable[[], None],
) -> None:
    endpoint = _minio_endpoint_from_env()
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    prefix = f"s3a://lakehouse/pytest-manifest-{uuid.uuid4().hex[:10]}"

    settings = make_s3_settings(
        prefix, endpoint=endpoint, access_key=access_key, secret_key=secret_key
    )
    paths = LakehousePaths.from_settings(settings)

    silver_schema = StructType(
        [
            StructField("subject_id", StringType(), False),
            StructField("label", IntegerType(), False),
            StructField("date", TimestampType(), False),
        ]
    )
    quarantine_schema = StructType(
        [
            StructField("_rejection_reason", StringType(), False),
        ]
    )
    silver_df = spark.createDataFrame(
        [
            ("s1", 1, datetime(2025, 1, 1, 2, 0)),
            ("s2", 0, datetime(2025, 1, 2, 14, 0)),
        ],
        schema=silver_schema,
    )
    quarantine_df = spark.createDataFrame(
        [("EMPTY_TEXT",)],
        schema=quarantine_schema,
    )

    fs, _ = resolve_manifest_filesystem(paths.gold_dataset_manifest, settings=settings)
    bucket_prefix = prefix.removeprefix("s3a://")
    try:
        built = build_dataset_manifest(
            silver_df,
            quarantine_df,
            paths.gold_dataset_manifest,
            settings=settings,
        )

        jobs_before = spark._jsc.sc().dagScheduler().nextJobId()
        forbid_active_spark()
        loaded = load_manifest(settings=settings)
        jobs_after = spark._jsc.sc().dagScheduler().nextJobId()

        assert jobs_after == jobs_before
        assert loaded == built

        with pytest.raises(FileNotFoundError):
            load_manifest(f"{prefix}/missing_manifest.json", settings=settings)
    finally:
        fs.delete_dir(bucket_prefix)
