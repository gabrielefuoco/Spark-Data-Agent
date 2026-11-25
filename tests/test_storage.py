"""Unit and MinIO integration tests for S3A/MinIO storage configuration."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest
from pydantic import SecretStr
from pyspark.sql import SparkSession
from src.analytics.safe_sql import execute_safe_spark_sql, register_lakehouse_views
from src.config import Settings
from src.lakehouse.paths import LakehousePaths
from src.lakehouse.storage import (
    SIMPLE_CREDENTIALS_PROVIDER,
    StorageConfigurationError,
    apply_s3a_config,
    build_s3a_hadoop_options,
    is_s3a_uri,
)
from src.spark import write_parquet_zstd


def _minio_endpoint_from_env() -> str:
    return os.environ.get("S3_ENDPOINT_URL", "http://minio:9000").strip() or "http://minio:9000"


def _is_endpoint_reachable(endpoint_url: str, timeout: float = 1.0) -> bool:
    """Return True iff a TCP connection to the S3/MinIO endpoint succeeds."""
    parsed = urlparse(endpoint_url if "://" in endpoint_url else f"http://{endpoint_url}")
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_uri_detection_and_target_resolution(tmp_path: Path) -> None:
    assert is_s3a_uri("s3a://lakehouse/bronze") is True
    assert is_s3a_uri("S3A://lakehouse") is True
    assert is_s3a_uri(tmp_path) is False
    assert is_s3a_uri("/workspace/data") is False

    local_settings = Settings(_env_file=None, spark_storage_uri=str(tmp_path / "data"))
    assert build_s3a_hadoop_options(local_settings) == {}

    s3_settings = Settings(
        _env_file=None,
        spark_storage_uri="s3a://lakehouse/prefix/",
        aws_access_key_id=SecretStr("minioadmin"),
        aws_secret_access_key=SecretStr("minioadmin"),
        s3_endpoint_url="http://minio:9000",
    )
    lakehouse_paths = LakehousePaths.from_settings(s3_settings)
    assert lakehouse_paths.silver_posts == "s3a://lakehouse/prefix/lakehouse/silver/posts"
    assert lakehouse_paths.gold_dataset_manifest == (
        "s3a://lakehouse/prefix/lakehouse/gold/gold_dataset_manifest.json"
    )


def test_build_s3a_hadoop_options_for_minio_and_https() -> None:
    minio_settings = Settings(
        _env_file=None,
        spark_storage_uri="s3a://lakehouse/",
        aws_access_key_id=SecretStr("minioadmin"),
        aws_secret_access_key=SecretStr("minioadmin"),
        s3_endpoint_url="http://minio:9000/",
    )
    opts = build_s3a_hadoop_options(minio_settings)
    assert opts == {
        "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "fs.s3a.endpoint": "http://minio:9000",
        "fs.s3a.path.style.access": "true",
        "fs.s3a.access.key": "minioadmin",
        "fs.s3a.secret.key": "minioadmin",
        "fs.s3a.connection.ssl.enabled": "false",
        "fs.s3a.aws.credentials.provider": SIMPLE_CREDENTIALS_PROVIDER,
    }

    https_settings = Settings(
        _env_file=None,
        spark_storage_uri="s3a://lakehouse/",
        aws_access_key_id=SecretStr("minioadmin"),
        aws_secret_access_key=SecretStr("minioadmin"),
        s3_endpoint_url="https://s3.example.internal",
    )
    https_opts = build_s3a_hadoop_options(https_settings)
    assert https_opts["fs.s3a.connection.ssl.enabled"] == "true"
    assert https_opts["fs.s3a.endpoint"] == "https://s3.example.internal"


def test_s3a_configuration_fails_loudly_without_leaking_secrets() -> None:
    missing_all = Settings(
        _env_file=None,
        spark_storage_uri="s3a://lakehouse/",
        aws_access_key_id=None,
        aws_secret_access_key=None,
        s3_endpoint_url=None,
    )
    with pytest.raises(StorageConfigurationError) as exc_all:
        build_s3a_hadoop_options(missing_all)
    msg_all = str(exc_all.value)
    assert "AWS_ACCESS_KEY_ID" in msg_all
    assert "AWS_SECRET_ACCESS_KEY" in msg_all
    assert "S3_ENDPOINT_URL" in msg_all

    sentinel_secret = "super-secret-value-never-log"
    missing_endpoint = Settings(
        _env_file=None,
        spark_storage_uri="s3a://lakehouse/",
        aws_access_key_id=SecretStr("test-access-key-id"),
        aws_secret_access_key=SecretStr(sentinel_secret),
        s3_endpoint_url="   ",
    )
    with pytest.raises(StorageConfigurationError) as exc_endpoint:
        build_s3a_hadoop_options(missing_endpoint)
    msg_endpoint = str(exc_endpoint.value)
    assert "S3_ENDPOINT_URL" in msg_endpoint
    assert sentinel_secret not in msg_endpoint
    assert "test-access-key-id" not in msg_endpoint


@pytest.mark.spark
@pytest.mark.skipif(
    not _is_endpoint_reachable(_minio_endpoint_from_env()),
    reason="MinIO S3-compatible endpoint is not reachable.",
)
def test_s3a_parquet_roundtrip_and_safe_sql_on_minio(spark: SparkSession) -> None:
    endpoint = _minio_endpoint_from_env()
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    prefix = f"s3a://lakehouse/pytest-storage-{uuid.uuid4().hex[:10]}"

    settings = Settings(
        _env_file=None,
        spark_storage_uri=prefix,
        aws_access_key_id=SecretStr(access_key),
        aws_secret_access_key=SecretStr(secret_key),
        s3_endpoint_url=endpoint,
        sql_row_limit=10,
    )
    apply_s3a_config(spark, settings)
    paths = LakehousePaths.from_settings(settings)

    sample_df = spark.createDataFrame(
        [("subj_a", 1, "night post sample"), ("subj_b", 0, "day post sample")],
        schema="subject_id STRING, label INT, text STRING",
    )

    try:
        write_parquet_zstd(sample_df, paths.silver_posts, mode="overwrite")
        loaded_rows = spark.read.parquet(paths.silver_posts).orderBy("subject_id").collect()
        assert [(r["subject_id"], r["label"], r["text"]) for r in loaded_rows] == [
            ("subj_a", 1, "night post sample"),
            ("subj_b", 0, "day post sample"),
        ]

        registered = register_lakehouse_views(spark, settings=settings, paths=paths)
        assert registered == ["silver_posts"]

        sql_rows = execute_safe_spark_sql(
            "SELECT subject_id, label FROM silver_posts WHERE label = 1",
            spark,
            settings=settings,
            paths=paths,
        ).collect()
        assert [(r["subject_id"], r["label"]) for r in sql_rows] == [("subj_a", 1)]
    finally:
        jvm = spark.sparkContext._jvm
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        uri = jvm.java.net.URI(prefix)
        fs = jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
        fs.delete(jvm.org.apache.hadoop.fs.Path(prefix), True)


@pytest.mark.spark
def test_build_spark_session_wires_s3a_only_for_s3a_uri(tmp_path: Path) -> None:
    # Run in an isolated subprocess so stopping the dedicated SparkSession never
    # terminates the session-scoped `spark` fixture used by other tests.
    script = textwrap.dedent(
        f"""
        from pydantic import SecretStr
        from src.config import Settings
        from src.lakehouse.storage import SIMPLE_CREDENTIALS_PROVIDER
        from src.spark import build_spark_session

        local_settings = Settings(
            _env_file=None,
            spark_storage_uri={str(tmp_path / "local-data")!r},
            spark_app_name="local-session-test",
            spark_master="local[1]",
        )
        local_spark = build_spark_session(local_settings, shuffle_partitions=1)
        try:
            assert local_spark.conf.get("spark.hadoop.fs.s3a.endpoint", None) is None
            assert local_spark.conf.get("spark.hadoop.fs.s3a.impl", None) is None
            assert local_spark.conf.get("spark.hadoop.fs.s3a.path.style.access", None) is None
            assert (
                local_spark.conf.get("spark.hadoop.fs.s3a.aws.credentials.provider", None)
                is None
            )
        finally:
            local_spark.stop()

        s3_settings = Settings(
            _env_file=None,
            spark_storage_uri="s3a://lakehouse/unit-check",
            spark_app_name="s3a-session-test",
            spark_master="local[1]",
            aws_access_key_id=SecretStr("minioadmin"),
            aws_secret_access_key=SecretStr("minioadmin"),
            s3_endpoint_url="http://minio:9000",
        )
        s3_spark = build_spark_session(s3_settings, shuffle_partitions=1)
        try:
            assert s3_spark.conf.get("spark.hadoop.fs.s3a.endpoint") == "http://minio:9000"
            assert s3_spark.conf.get("spark.hadoop.fs.s3a.path.style.access") == "true"
            assert s3_spark.conf.get("spark.hadoop.fs.s3a.connection.ssl.enabled") == "false"
            assert (
                s3_spark.conf.get("spark.hadoop.fs.s3a.aws.credentials.provider")
                == SIMPLE_CREDENTIALS_PROVIDER
            )
            assert (
                s3_spark.sparkContext._jsc.hadoopConfiguration().get("fs.s3a.endpoint")
                == "http://minio:9000"
            )
        finally:
            s3_spark.stop()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
