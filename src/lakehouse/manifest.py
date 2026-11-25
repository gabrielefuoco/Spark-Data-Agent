"""Static Gold dataset manifest built by ETL and read with zero Spark jobs.

Persists global corpus statistics at ETL time so startup and metadata tools
answer immediately without triggering blocking Spark ``.count()`` calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow.fs as pafs
from pydantic import BaseModel, Field
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import Settings, get_settings
from src.lakehouse.paths import LakehousePaths
from src.lakehouse.storage import (
    StorageConfigurationError,
    is_s3a_uri,
    resolve_s3_connection,
)


class DatasetManifest(BaseModel):
    """Precomputed lakehouse summary read directly from JSON at startup."""

    total_posts: int
    total_subjects: int
    subjects_per_class: dict[str, int]
    posts_per_class: dict[str, int]
    min_date: str | None
    max_date: str | None
    quarantine_count: int
    quarantine_by_reason: dict[str, int]
    # Bytes on disk per layer and per table, computed at ETL time so a "how big
    # is the lakehouse" question never wakes Spark. Optional: manifests written
    # before this field existed still load.
    storage_volumes: dict[str, Any] = Field(default_factory=dict)
    generated_at: str


def resolve_manifest_filesystem(
    manifest_path: str | Path,
    settings: Settings | None = None,
) -> tuple[pafs.FileSystem, str]:
    """Return the PyArrow filesystem and normalized path for a manifest path or ``s3a://`` URI."""
    raw = str(manifest_path).strip()
    if not is_s3a_uri(raw):
        return pafs.LocalFileSystem(), str(manifest_path)

    parsed = urlparse(raw)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise StorageConfigurationError(
            f"Invalid S3A manifest URI {raw!r}: expected 's3a://<bucket>/<key>'."
        )

    cfg = settings or get_settings()
    access_key, secret_key, endpoint = resolve_s3_connection(cfg, target_uri=raw)
    ep = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    scheme = ep.scheme or "http"
    endpoint_override = ep.netloc or ep.path
    fs = pafs.S3FileSystem(
        access_key=access_key,
        secret_key=secret_key,
        endpoint_override=endpoint_override,
        scheme=scheme,
    )
    return fs, f"{bucket}/{key}"


def build_dataset_manifest(
    silver_posts_df: DataFrame,
    quarantine_df: DataFrame,
    output_path: str | Path,
    *,
    settings: Settings | None = None,
) -> DatasetManifest:
    """Compute manifest statistics via Spark aggregations and write JSON."""
    class_rows = (
        silver_posts_df.groupBy("label")
        .agg(
            F.count("*").alias("post_count"),
            F.countDistinct("subject_id").alias("subject_count"),
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date"),
        )
        .collect()
    )

    subjects_per_class: dict[str, int] = {}
    posts_per_class: dict[str, int] = {}
    min_dates = []
    max_dates = []
    total_posts = 0
    total_subjects = 0

    for row in sorted(class_rows, key=lambda r: int(r["label"])):
        key = str(int(row["label"]))
        post_cnt = int(row["post_count"])
        subj_cnt = int(row["subject_count"])
        posts_per_class[key] = post_cnt
        subjects_per_class[key] = subj_cnt
        total_posts += post_cnt
        total_subjects += subj_cnt
        if row["min_date"] is not None:
            min_dates.append(row["min_date"])
        if row["max_date"] is not None:
            max_dates.append(row["max_date"])

    quarantine_rows = (
        quarantine_df.groupBy("_rejection_reason").agg(F.count("*").alias("row_count")).collect()
    )
    quarantine_by_reason = {
        str(row["_rejection_reason"]): int(row["row_count"])
        for row in sorted(quarantine_rows, key=lambda r: str(r["_rejection_reason"]))
    }
    quarantine_count = sum(quarantine_by_reason.values())

    manifest = DatasetManifest(
        total_posts=total_posts,
        total_subjects=total_subjects,
        subjects_per_class=subjects_per_class,
        posts_per_class=posts_per_class,
        min_date=min(min_dates).isoformat() if min_dates else None,
        max_date=max(max_dates).isoformat() if max_dates else None,
        quarantine_count=quarantine_count,
        quarantine_by_reason=quarantine_by_reason,
        generated_at=datetime.now(UTC).isoformat(),
    )

    fs, fs_path = resolve_manifest_filesystem(output_path, settings=settings)
    if isinstance(fs, pafs.LocalFileSystem):
        parent = str(Path(fs_path).parent)
        if parent not in {"", "."}:
            fs.create_dir(parent, recursive=True)
    payload = (manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
    with fs.open_output_stream(fs_path) as stream:
        stream.write(payload)
    return manifest


def load_manifest(
    manifest_path: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> DatasetManifest:
    """Load the precomputed manifest JSON without starting or touching Spark."""
    target = manifest_path or LakehousePaths.from_settings(settings).gold_dataset_manifest
    fs, fs_path = resolve_manifest_filesystem(target, settings=settings)
    with fs.open_input_stream(fs_path) as stream:
        raw_json = stream.readall().decode("utf-8")
    return DatasetManifest.model_validate_json(raw_json)
