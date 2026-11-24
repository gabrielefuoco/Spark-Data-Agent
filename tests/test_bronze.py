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
