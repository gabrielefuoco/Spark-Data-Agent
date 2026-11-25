"""Spark integration tests for the Bronze ingestion layer."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.lakehouse.bronze import ingest_bronze_posts, ingest_bronze_subjects, run_bronze_ingestion
from src.lakehouse.paths import LakehousePaths


@pytest.mark.spark
def test_bronze_lineage_columns_populated(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    settings, paths, _ = lakehouse_env
    posts_df, subjects_df = run_bronze_ingestion(
        spark,
        paths.sample_chunks,
        paths.sample_truth,
        settings=settings,
        paths=paths,
    )

    # 10 subjects * 8 posts + 4 deliberately broken rows = 84 raw writings.
    assert posts_df.count() == 84
    assert subjects_df.count() == 10

    for col_name in ("_source_file", "_ingested_at", "_row_hash"):
        assert col_name in posts_df.columns
        assert col_name in subjects_df.columns
        assert posts_df.filter(F.col(col_name).isNull()).count() == 0
        assert subjects_df.filter(F.col(col_name).isNull()).count() == 0

    # Bronze keeps raw DATE untrimmed and immutable.
    raw_dates = [r["date"] for r in posts_df.select("date").limit(5).collect()]
    assert all(d.startswith(" ") and d.endswith(" ") for d in raw_dates)


@pytest.mark.spark
def test_bronze_row_hash_is_stable_across_runs(
    spark: SparkSession,
    lakehouse_env: tuple,
    tmp_path: Path,
) -> None:
    _, paths, _ = lakehouse_env
    paths_1 = LakehousePaths(str(tmp_path / "b1"))
    paths_2 = LakehousePaths(str(tmp_path / "b2"))
    run_1 = ingest_bronze_posts(spark, paths.sample_chunks, paths_1.bronze_posts)
    run_2 = ingest_bronze_posts(spark, paths.sample_chunks, paths_2.bronze_posts)

    hashes_1 = sorted(r["_row_hash"] for r in run_1.select("_row_hash").collect())
    hashes_2 = sorted(r["_row_hash"] for r in run_2.select("_row_hash").collect())
    assert hashes_1 == hashes_2

    subj_1 = ingest_bronze_subjects(spark, paths.sample_truth, paths_1.bronze_subjects)
    subj_2 = ingest_bronze_subjects(spark, paths.sample_truth, paths_2.bronze_subjects)
    assert sorted(r["_row_hash"] for r in subj_1.select("_row_hash").collect()) == sorted(
        r["_row_hash"] for r in subj_2.select("_row_hash").collect()
    )
    assert Path(paths_1.bronze_posts).exists()


@pytest.mark.spark
def test_bronze_reads_corpus_split_across_several_chunk_directories(
    spark: SparkSession,
    lakehouse_env: tuple,
    tmp_path: Path,
) -> None:
    """A real eRISK corpus is chunk1..chunkN; spark-xml takes one path, not a list.

    Feeding the reader a list of globs raises
    "IllegalArgumentException: 'path' must be specified for XML data", and
    recursiveFileLookup is not usable because spark-xml needs block locations.
    The synthetic fixture writes a single chunk directory, so only a nested
    layout catches this.
    """
    settings, paths, sample_dir = lakehouse_env
    nested_root = tmp_path / "nested_corpus"
    second_chunk = nested_root / "chunk2"
    second_chunk.mkdir(parents=True)

    source_files = sorted((sample_dir / "chunk").rglob("*.xml"))
    for index, source in enumerate(source_files):
        target_dir = nested_root / f"chunk{index % 2 + 1}"
        target_dir.mkdir(exist_ok=True)
        (target_dir / source.name).write_text(source.read_text(encoding="utf-8"))

    posts_df = ingest_bronze_posts(spark, nested_root, paths.bronze_posts)

    assert posts_df.count() == 84
    # Bronze keeps raw values: a file whose name carries no id yields a null
    # subject_id here, and the Silver layer is what rejects it.
    named = {
        row["subject_id"]
        for row in posts_df.select("subject_id").distinct().collect()
        if row["subject_id"] is not None
    }
    # The generator ships ten labelled subjects plus one orphan id on purpose.
    assert {f"subject{index:03d}" for index in range(1, 11)} <= named
    assert len(named) == 11
