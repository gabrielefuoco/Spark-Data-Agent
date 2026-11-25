"""Spark and zero-job tests for the Gold layer and static manifest."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.agent.nodes import make_spark_job_executor
from src.analytics.safe_sql import execute_safe_spark_sql
from src.lakehouse.gold import build_gold
from src.lakehouse.manifest import load_manifest
from src.lakehouse.silver import build_silver


@pytest.mark.spark
def test_gold_tables_and_manifest_zero_spark_jobs(
    spark: SparkSession,
    bronze_lakehouse: tuple,
    forbid_active_spark: Callable[[], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, paths, _, _ = bronze_lakehouse
    build_silver(spark, settings=settings, paths=paths)
    user_features_df, lexical_df, built_manifest = build_gold(
        spark,
        settings=settings,
        paths=paths,
        lda_k=4,
        lda_max_iter=3,
    )

    # User longitudinal features cover all 10 valid subjects with features.py-derived metrics.
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
        "session_count",
        "mean_inter_post_gap_seconds",
        "mean_night_post_rolling_30d",
        "mean_verbosity_zscore",
        "max_abs_verbosity_zscore",
    }
    assert expected_user_cols.issubset(set(user_features_df.columns))

    # Lexical matrix has unigrams and bigrams with TF-IDF weights and ranks for both cohorts.
    expected_lexical_cols = {
        "label",
        "ngram_type",
        "ngram",
        "frequency",
        "relative_frequency",
        "idf_weight",
        "tfidf_weight",
        "tfidf_rank",
    }
    assert expected_lexical_cols.issubset(set(lexical_df.columns))
    ngram_types = {r["ngram_type"] for r in lexical_df.select("ngram_type").distinct().collect()}
    assert ngram_types == {"unigram", "bigram"}
    dep_bigrams = lexical_df.filter((F.col("label") == 1) & (F.col("ngram_type") == "bigram"))
    assert dep_bigrams.count() > 0
    assert dep_bigrams.filter(F.col("tfidf_weight") > 0.0).count() > 0

    # Gold LDA cohort topic matrix exists as ZSTD Parquet with valid per-cohort distributions.
    topics_df = spark.read.parquet(paths.gold_cohort_topic_matrix)
    assert {"label", "topic", "mean_topic_probability", "document_count", "top_terms"}.issubset(
        set(topics_df.columns)
    )
    topic_rows = topics_df.collect()
    assert {r["label"] for r in topic_rows} == {0, 1}
    assert all(r["top_terms"] for r in topic_rows)
    for cohort_label in (0, 1):
        cohort_prob_sum = sum(
            r["mean_topic_probability"] for r in topic_rows if r["label"] == cohort_label
        )
        assert cohort_prob_sum == pytest.approx(1.0, abs=1e-4)

    # Manifest values match ETL counts and JSON includes non-zero storage volumes.
    assert built_manifest.total_posts == 80
    assert built_manifest.total_subjects == 10
    assert built_manifest.subjects_per_class == {"0": 5, "1": 5}
    assert built_manifest.quarantine_count == 4
    assert built_manifest.min_date is not None
    assert built_manifest.max_date is not None
    assert built_manifest.min_date <= built_manifest.max_date

    raw_manifest = json.loads(Path(paths.gold_dataset_manifest).read_text(encoding="utf-8"))
    volumes = raw_manifest["storage_volumes"]
    assert volumes["total_bytes"] > 0
    assert set(volumes["by_layer"]) == {"bronze", "silver", "gold"}
    assert all(volumes["by_layer"][layer] > 0 for layer in ("bronze", "silver", "gold"))
    assert volumes["by_table"]["gold_cohort_topic_matrix"] > 0
    assert volumes["total_bytes"] == sum(volumes["by_table"].values())

    # Loading the manifest must NOT issue any Spark jobs or touch SparkSession.
    jobs_before = spark._jsc.sc().dagScheduler().nextJobId()
    forbid_active_spark()
    loaded = load_manifest(settings=settings)
    jobs_after = spark._jsc.sc().dagScheduler().nextJobId()

    assert jobs_after == jobs_before
    assert loaded == built_manifest

    # Runtime Spark job executor & Safe SQL view registration cover the new Gold topic table.
    monkeypatch.undo()
    sql_topics = execute_safe_spark_sql(
        "SELECT label, topic, mean_topic_probability FROM gold_cohort_topic_matrix",
        spark,
        settings=settings,
        paths=paths,
    ).collect()
    assert len(sql_topics) == 8

    job_exec = make_spark_job_executor(spark, settings=settings, paths=paths)
    circadian_out = job_exec("circadian_features", {}, settings)
    assert circadian_out["row_count"] == 10
    assert "session_count" in circadian_out["columns"]

    topic_out = job_exec("topic_model", {"k": 4, "max_iter": 2}, settings)
    assert topic_out["row_count"] == 8
    assert "top_terms" in topic_out["columns"]
