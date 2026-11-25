"""Spark tests for AQE config, broadcast join helper, and post window features."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType
from src.analytics.features import (
    compute_post_window_features,
    summarize_user_longitudinal_features,
)
from src.spark import broadcast_join

_POSTS_SCHEMA = StructType(
    [
        StructField("subject_id", StringType(), False),
        StructField("text", StringType(), False),
        StructField("date", TimestampType(), False),
        StructField("label", IntegerType(), False),
        StructField("_row_hash", StringType(), False),
    ]
)


@pytest.mark.spark
def test_spark_session_has_aqe_enabled_and_broadcast_join_works(spark: SparkSession) -> None:
    assert spark.conf.get("spark.sql.adaptive.enabled") == "true"
    assert spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled") == "true"
    assert spark.conf.get("spark.sql.adaptive.skewJoin.enabled") == "true"

    posts = spark.createDataFrame(
        [("s1", "hello"), ("s2", "world")],
        schema="subject_id STRING, text STRING",
    )
    subjects = spark.createDataFrame(
        [("s1", 1), ("s2", 0)],
        schema="subject_id STRING, label INT",
    )

    joined = broadcast_join(posts, subjects, on="subject_id", how="inner")
    plan_text = joined._jdf.queryExecution().logical().toString()
    assert "ResolvedHint" in plan_text or "Broadcast" in plan_text

    rows = {r["subject_id"]: r["label"] for r in joined.collect()}
    assert rows == {"s1": 1, "s2": 0}


@pytest.mark.spark
def test_compute_post_window_features_exact_values_and_30d_range_window(
    spark: SparkSession,
) -> None:
    # Subject s1: 3 posts on Jan 1 (burst + later session), then 1 post 35 days later (Feb 5).
    # Subject s2: 1 baseline post on Jan 1, then a burst of 35 posts on Feb 10 (outside 30d).
    rows: list[tuple[str, str, datetime, int, str]] = [
        ("s1", "aaaa", datetime(2024, 1, 1, 1, 0, 0), 1, "h1"),
        ("s1", "aaaa", datetime(2024, 1, 1, 1, 10, 0), 1, "h2"),
        ("s1", "aaaaaaaaaa", datetime(2024, 1, 1, 12, 0, 0), 1, "h3"),
        ("s1", "aaaaaaaa", datetime(2024, 2, 5, 3, 0, 0), 1, "h4"),
        ("s2", "bb", datetime(2024, 1, 1, 2, 0, 0), 0, "s2_0"),
    ]
    burst_start = datetime(2024, 2, 10, 0, 15, 0)
    for idx in range(35):
        rows.append(
            (
                "s2",
                "bbbb" if idx % 2 == 0 else "bbbbbbbb",
                burst_start + timedelta(minutes=idx),
                0,
                f"s2_{idx + 1}",
            )
        )

    df = spark.createDataFrame(rows, schema=_POSTS_SCHEMA)
    result = compute_post_window_features(df, window_days=30, session_gap_seconds=1800)

    s1_rows = result.filter(F.col("subject_id") == "s1").orderBy("date").collect()
    assert len(s1_rows) == 4

    # Post 1: single post in window -> zero-variance guard returns 0.0, no previous post.
    p1 = s1_rows[0]
    assert p1["inter_post_gap_seconds"] is None
    assert p1["is_new_session"] == 1
    assert p1["session_index"] == 1
    assert p1["is_night_post"] == 1
    assert p1["night_post_rolling_30d"] == pytest.approx(1.0)
    assert p1["rolling_30d_text_length_mean"] == pytest.approx(4.0)
    assert p1["rolling_30d_text_length_std"] == pytest.approx(0.0)
    assert p1["verbosity_zscore"] == pytest.approx(0.0)

    # Post 2: 600s later (same session), identical length -> std=0.0, zscore=0.0.
    p2 = s1_rows[1]
    assert p2["inter_post_gap_seconds"] == 600
    assert p2["is_new_session"] == 0
    assert p2["session_index"] == 1
    assert p2["is_night_post"] == 1
    assert p2["night_post_rolling_30d"] == pytest.approx(1.0)
    assert p2["rolling_30d_text_length_mean"] == pytest.approx(4.0)
    assert p2["rolling_30d_text_length_std"] == pytest.approx(0.0)
    assert p2["verbosity_zscore"] == pytest.approx(0.0)

    # Post 3: 10h50m later (new session), daytime post (12:00), lengths [4, 4, 10].
    p3 = s1_rows[2]
    assert p3["inter_post_gap_seconds"] == 39_000
    assert p3["is_new_session"] == 1
    assert p3["session_index"] == 2
    assert p3["is_night_post"] == 0
    assert p3["night_post_rolling_30d"] == pytest.approx(2.0 / 3.0)
    assert p3["rolling_30d_text_length_mean"] == pytest.approx(6.0)
    expected_std = math.sqrt(((4.0 - 6.0) ** 2 + (4.0 - 6.0) ** 2 + (10.0 - 6.0) ** 2) / 2.0)
    assert p3["rolling_30d_text_length_std"] == pytest.approx(expected_std)
    assert p3["verbosity_zscore"] == pytest.approx((10.0 - 6.0) / expected_std)

    # Post 4: 35 days after Jan 1 -> time rangeBetween(-30*86400, 0) excludes Jan 1 posts!
    p4 = s1_rows[3]
    assert p4["is_new_session"] == 1
    assert p4["session_index"] == 3
    assert p4["is_night_post"] == 1
    assert p4["night_post_rolling_30d"] == pytest.approx(1.0)
    assert p4["rolling_30d_text_length_mean"] == pytest.approx(8.0)
    assert p4["rolling_30d_text_length_std"] == pytest.approx(0.0)
    assert p4["verbosity_zscore"] == pytest.approx(0.0)

    # Subject s2: burst of 35 night posts on Feb 10 excludes the Jan 1 post (40 days earlier),
    # and the final burst post sees all 35 burst posts (not capped at 31 rows as rowsBetween would).
    s2_last = result.filter(F.col("subject_id") == "s2").orderBy(F.col("date").desc()).first()
    assert s2_last is not None
    assert s2_last["night_post_rolling_30d"] == pytest.approx(1.0)
    assert s2_last["session_index"] == 2
    # 18 posts of length 4 and 17 posts of length 8 -> mean = (18*4 + 17*8)/35 = 208/35.
    assert s2_last["rolling_30d_text_length_mean"] == pytest.approx(208.0 / 35.0)
    assert not math.isnan(s2_last["verbosity_zscore"])
    assert not math.isinf(s2_last["verbosity_zscore"])

    # Per-subject summary derives directly from compute_post_window_features.
    summary = summarize_user_longitudinal_features(df, window_days=30, session_gap_seconds=1800)
    s1_summary = summary.filter(F.col("subject_id") == "s1").first()
    assert s1_summary is not None
    assert s1_summary["post_count"] == 4
    assert s1_summary["night_post_count"] == 3
    assert s1_summary["night_post_ratio"] == pytest.approx(0.75)
    assert s1_summary["session_count"] == 3
    assert s1_summary["max_abs_verbosity_zscore"] > 0.0
