"""Spark and zero-job tests for the Gold layer and static manifest."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.lakehouse.bronze import run_bronze_ingestion
from src.lakehouse.gold import build_gold
from src.lakehouse.manifest import load_manifest
from src.lakehouse.silver import build_silver


@pytest.mark.spark
def test_gold_tables_and_manifest_zero_spark_jobs(
    spark: SparkSession,
    lakehouse_env: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, paths, _ = lakehouse_env
    run_bronze_ingestion(
        spark,
        paths.sample_chunks,
        paths.sample_truth,
        settings=settings,
        paths=paths,
    )
    build_silver(spark, settings=settings, paths=paths)
    user_features_df, lexical_df, built_manifest = build_gold(
        spark,
        settings=settings,
        paths=paths,
    )

    # User longitudinal features cover all 10 valid subjects with expected metrics.
    assert user_features_df.count() == 10
    expected_user_cols = {
        "subject_id",
        "label",
        "post_count",
        "first_post_at",
        "last_post_at",
        "span_days",
        "active_days",
        "posts_per_active_day",
        "mean_text_length",
        "stddev_text_length",
        "night_post_count",
        "night_post_ratio",
    }
    assert expected_user_cols.issubset(set(user_features_df.columns))

    # Lexical matrix has both unigrams and bigrams for both cohorts.
    ngram_types = {r["ngram_type"] for r in lexical_df.select("ngram_type").distinct().collect()}
    assert ngram_types == {"unigram", "bigram"}
    assert lexical_df.filter((F.col("label") == 1) & (F.col("ngram_type") == "bigram")).count() > 0

    # Manifest values match ETL counts.
    assert built_manifest.total_posts == 80
    assert built_manifest.total_subjects == 10
    assert built_manifest.subjects_per_class == {"0": 5, "1": 5}
    assert built_manifest.quarantine_count == 4
    assert built_manifest.min_date is not None
    assert built_manifest.max_date is not None
    assert built_manifest.min_date <= built_manifest.max_date

    # Loading the manifest must NOT issue any Spark jobs or touch SparkSession.
    jobs_before = spark._jsc.sc().dagScheduler().nextJobId()

    def _forbid_spark(*args, **kwargs):
        raise AssertionError("load_manifest must never invoke SparkSession")

    monkeypatch.setattr(SparkSession, "getActiveSession", staticmethod(_forbid_spark))
    loaded = load_manifest(settings=settings)
    jobs_after = spark._jsc.sc().dagScheduler().nextJobId()

    assert jobs_after == jobs_before
    assert loaded == built_manifest
