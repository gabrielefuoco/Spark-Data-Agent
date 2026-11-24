"""Spark integration tests for the Silver layer and dead-letter quarantine."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, TimestampType
from scripts.generate_dev_sample import generate_synthetic_sample
from src.lakehouse.bronze import ingest_bronze_posts, run_bronze_ingestion
from src.lakehouse.silver import (
    REASON_DUPLICATE_POST,
    REASON_MALFORMED_DATE,
    REASON_MISSING_SUBJECT_ID,
    REASON_NULL_OR_BLANK_TEXT,
    REASON_ORPHAN_SUBJECT_ID,
    build_silver,
)


@pytest.mark.spark
def test_silver_quarantine_reasons_partitioning_and_accounting(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    settings, paths, _ = lakehouse_env
    bronze_posts, _ = run_bronze_ingestion(
        spark,
        paths.sample_chunks,
        paths.sample_truth,
        settings=settings,
        paths=paths,
    )
    bronze_count = bronze_posts.count()

    silver_df, quarantine_df = build_silver(spark, settings=settings, paths=paths)
    silver_count = silver_df.count()
    quarantine_count = quarantine_df.count()

    # Every Bronze row ends in either Silver or Quarantine; none silently dropped.
    assert bronze_count == silver_count + quarantine_count
    assert silver_count == 80
    assert quarantine_count == 4

    reasons = {
        row["_rejection_reason"]
        for row in quarantine_df.select("_rejection_reason").collect()
    }
    assert reasons == {
        REASON_MISSING_SUBJECT_ID,
        REASON_MALFORMED_DATE,
        REASON_NULL_OR_BLANK_TEXT,
        REASON_ORPHAN_SUBJECT_ID,
    }

    # Types and physical partitioning on disk by (label, year).
    assert isinstance(silver_df.schema["date"].dataType, TimestampType)
    assert isinstance(silver_df.schema["year"].dataType, IntegerType)
    silver_root = Path(paths.silver_posts)
    label_dirs = sorted(p.name for p in silver_root.iterdir() if p.is_dir())
    assert label_dirs == ["label=0", "label=1"]
    year_dirs = [
        p.name
        for label_dir in silver_root.glob("label=*")
        for p in label_dir.iterdir()
        if p.is_dir()
    ]
    assert year_dirs
    assert all(name.startswith("year=") for name in year_dirs)


@pytest.mark.spark
def test_silver_deduplicates_exact_duplicates_only(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    sample_dir = generate_synthetic_sample(
        tmp_path / "raw",
        seed=5,
        subjects=2,
        posts_per_subject=2,
        include_broken=False,
    )
    chunk_dir = sample_dir / "chunk" / "chunk1"

    # Exact duplicate of subject001_1.xml (2 posts) + same subject/timestamp with different text.
    original_xml = (chunk_dir / "subject001_1.xml").read_text(encoding="utf-8")
    (chunk_dir / "subject001_2.xml").write_text(original_xml, encoding="utf-8")

    # Extract one timestamp from subject001_1.xml and write a distinct post at the exact same timestamp.
    first_date = [
        line.strip().removeprefix("<DATE>").removesuffix("</DATE>").strip()
        for line in original_xml.splitlines()
        if "<DATE>" in line
    ][0]
    (chunk_dir / "subject001_3.xml").write_text(
        "<INDIVIDUAL>\n"
        "  <WRITING>\n"
        "    <TITLE> Distinct title </TITLE>\n"
        f"    <DATE> {first_date} </DATE>\n"
        "    <INFO> reddit post </INFO>\n"
        "    <TEXT> Different text sharing the same timestamp. </TEXT>\n"
        "  </WRITING>\n"
        "</INDIVIDUAL>\n",
        encoding="utf-8",
    )

    bronze_posts = ingest_bronze_posts(spark, sample_dir / "chunk", tmp_path / "bronze" / "posts")
    from src.lakehouse.bronze import ingest_bronze_subjects

    ingest_bronze_subjects(
        spark,
        sample_dir / "risk-golden-truth-test.txt",
        tmp_path / "bronze" / "subjects",
    )

    silver_df, quarantine_df = build_silver(
        spark,
        bronze_posts_path=tmp_path / "bronze" / "posts",
        bronze_subjects_path=tmp_path / "bronze" / "subjects",
        silver_posts_path=tmp_path / "silver" / "posts",
        silver_quarantine_path=tmp_path / "silver" / "quarantine",
    )

    # 2 original subject001 + 2 subject002 + 1 distinct-text post = 5 kept; 2 exact dupes quarantined.
    assert bronze_posts.count() == 7
    assert silver_df.count() == 5
    assert quarantine_df.count() == 2
    assert (
        quarantine_df.filter(F.col("_rejection_reason") == REASON_DUPLICATE_POST).count() == 2
    )
    assert silver_df.count() + quarantine_df.count() == bronze_posts.count()


@pytest.mark.spark
def test_silver_handles_empty_quarantine(spark: SparkSession, tmp_path: Path) -> None:
    sample_dir = generate_synthetic_sample(
        tmp_path / "clean_raw",
        seed=11,
        subjects=4,
        posts_per_subject=3,
        include_broken=False,
    )
    from src.lakehouse.bronze import ingest_bronze_subjects

    ingest_bronze_posts(spark, sample_dir / "chunk", tmp_path / "bronze" / "posts")
    ingest_bronze_subjects(
        spark,
        sample_dir / "risk-golden-truth-test.txt",
        tmp_path / "bronze" / "subjects",
    )

    silver_df, quarantine_df = build_silver(
        spark,
        bronze_posts_path=tmp_path / "bronze" / "posts",
        bronze_subjects_path=tmp_path / "bronze" / "subjects",
        silver_posts_path=tmp_path / "silver" / "posts",
        silver_quarantine_path=tmp_path / "silver" / "quarantine",
    )

    assert silver_df.count() == 12
    assert quarantine_df.count() == 0

