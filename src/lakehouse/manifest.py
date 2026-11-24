"""Static Gold dataset manifest built by ETL and read with zero Spark jobs.

Persists global corpus statistics at ETL time so startup and metadata tools
answer immediately without triggering blocking Spark ``.count()`` calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import Settings
from src.lakehouse.paths import LakehousePaths


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
    generated_at: str


def build_dataset_manifest(
    silver_posts_df: DataFrame,
    quarantine_df: DataFrame,
    output_path: str | Path,
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
        quarantine_df.groupBy("_rejection_reason")
        .agg(F.count("*").alias("row_count"))
        .collect()
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

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(
    manifest_path: str | Path | None = None,
    *,
    settings: Settings | None = None,
) -> DatasetManifest:
    """Load the precomputed manifest JSON without starting or touching Spark."""
    path = Path(manifest_path or LakehousePaths.from_settings(settings).gold_dataset_manifest)
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
