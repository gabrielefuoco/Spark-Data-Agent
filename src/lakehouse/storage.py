"""S3A / MinIO storage configuration for PySpark lakehouse sessions.

Resolves storage targets from ``Settings.spark_storage_uri`` (supporting both
local filesystem paths and ``s3a://`` URIs) and builds deterministic Hadoop
S3A options without leaking credentials or hitting EC2 instance metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pyspark import SparkConf
from pyspark.sql import SparkSession

from src.config import Settings, get_settings

S3A_SCHEME = "s3a://"
S3A_FILESYSTEM_IMPL = "org.apache.hadoop.fs.s3a.S3AFileSystem"
SIMPLE_CREDENTIALS_PROVIDER = "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"

_SparkTarget = TypeVar("_SparkTarget", bound=SparkSession.Builder | SparkConf | SparkSession)


class StorageConfigurationError(ValueError):
    """Raised when ``s3a://`` storage is requested without required S3 settings."""


def is_s3a_uri(uri: str | Path) -> bool:
    """Return True when ``uri`` targets the Hadoop S3A filesystem."""
    return str(uri).strip().lower().startswith(S3A_SCHEME)


def resolve_s3_connection(
    settings: Settings | None = None,
    *,
    target_uri: str | Path | None = None,
) -> tuple[str, str, str]:
    """Validate and return ``(access_key, secret_key, endpoint_url)`` from ``Settings``."""
    cfg = settings or get_settings()
    access_key = (
        cfg.aws_access_key_id.get_secret_value().strip()
        if cfg.aws_access_key_id is not None
        else ""
    )
    secret_key = (
        cfg.aws_secret_access_key.get_secret_value().strip()
        if cfg.aws_secret_access_key is not None
        else ""
    )
    endpoint = cfg.s3_endpoint_url.strip().rstrip("/") if cfg.s3_endpoint_url is not None else ""

    missing: list[str] = []
    if not access_key:
        missing.append("AWS_ACCESS_KEY_ID")
    if not secret_key:
        missing.append("AWS_SECRET_ACCESS_KEY")
    if not endpoint:
        missing.append("S3_ENDPOINT_URL")

    if missing:
        uri_label = str(target_uri) if target_uri is not None else cfg.spark_storage_uri
        raise StorageConfigurationError(
            f"S3A storage ({uri_label!r}) requires {', '.join(missing)}. "
            "Set the missing environment variables or `.env` entries before starting Spark."
        )

    return access_key, secret_key, endpoint


def build_s3a_hadoop_options(settings: Settings | None = None) -> dict[str, str]:
    """Return Hadoop ``fs.s3a.*`` options for ``Settings``, or ``{}`` for local storage.

    Uses ``SimpleAWSCredentialsProvider`` so Hadoop reads static keys directly
    and never hangs or fails trying to reach the AWS EC2 metadata endpoint.
    SSL is disabled automatically for plain ``http://`` endpoints (such as local
    MinIO at ``http://minio:9000``) and enabled for ``https://`` endpoints.
    """
    cfg = settings or get_settings()
    if not is_s3a_uri(cfg.spark_storage_uri):
        return {}

    access_key, secret_key, endpoint = resolve_s3_connection(cfg)
    ssl_enabled = "true" if endpoint.lower().startswith("https://") else "false"
    return {
        "fs.s3a.impl": S3A_FILESYSTEM_IMPL,
        "fs.s3a.endpoint": endpoint,
        "fs.s3a.path.style.access": "true",
        "fs.s3a.access.key": access_key,
        "fs.s3a.secret.key": secret_key,
        "fs.s3a.connection.ssl.enabled": ssl_enabled,
        "fs.s3a.aws.credentials.provider": SIMPLE_CREDENTIALS_PROVIDER,
    }


def apply_s3a_config(
    target: _SparkTarget,
    settings: Settings | None = None,
) -> _SparkTarget:
    """Attach S3A Hadoop options from ``Settings`` to a Spark builder, conf, or session."""
    options = build_s3a_hadoop_options(settings)
    if not options:
        return target

    if isinstance(target, SparkSession.Builder):
        for key, value in options.items():
            target.config(f"spark.hadoop.{key}", value)
        return target

    if isinstance(target, SparkConf):
        for key, value in options.items():
            target.set(f"spark.hadoop.{key}", value)
        return target

    if isinstance(target, SparkSession):
        hadoop_conf = target.sparkContext._jsc.hadoopConfiguration()
        for key, value in options.items():
            hadoop_conf.set(key, value)
            target.conf.set(f"spark.hadoop.{key}", value)
        return target

    raise TypeError(f"Unsupported Spark target type: {type(target).__name__}")
