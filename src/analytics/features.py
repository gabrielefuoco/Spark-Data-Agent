"""Advanced PySpark window features on atomic Silver posts.

Computes per-subject circadian posting rhythm, inter-post burstiness/session
boundaries, and rolling 30-day verbosity z-scores while preserving the atomic
1-row-per-post granularity (never aggregating text via ``collect_list``).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType

SECONDS_PER_DAY = 86_400
DEFAULT_WINDOW_DAYS = 30
DEFAULT_SESSION_GAP_SECONDS = 1_800
_STD_EPSILON = 1e-12


def compute_post_window_features(
    posts_df: DataFrame,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    session_gap_seconds: int = DEFAULT_SESSION_GAP_SECONDS,
) -> DataFrame:
    """Enrich atomic posts with per-subject temporal, circadian, and verbosity features.

    Uses two distinct window definitions because Spark's ``rangeBetween`` requires a
    single numeric ordering column, while ``lag`` benefits from deterministic tie-breaking:

    - ``seq_window``: ordered by ``(date, _row_hash)`` when ``_row_hash`` is present
      to compute exact inter-post gaps and session boundaries.
    - ``time_window``: ordered by epoch seconds with ``rangeBetween(-window_days * 86400, 0)``
      so the 30-day lookback reflects true calendar time rather than a fixed row count
      (which would shrink to minutes for a user posting 300 times in one night).

    When rolling standard deviation is zero (e.g., first post or identical lengths in
    window), ``verbosity_zscore`` is defined as ``0.0`` instead of ``NaN``/``inf``.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if session_gap_seconds < 0:
        raise ValueError("session_gap_seconds must be non-negative")

    order_cols = [F.col("date").asc()]
    if "_row_hash" in posts_df.columns:
        order_cols.append(F.col("_row_hash").asc())

    seq_window = Window.partitionBy("subject_id").orderBy(*order_cols)
    cumulative_seq_window = seq_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    lookback_seconds = window_days * SECONDS_PER_DAY
    time_window = (
        Window.partitionBy("subject_id")
        .orderBy(F.col("_post_ts_sec"))
        .rangeBetween(-lookback_seconds, 0)
    )

    prev_ts_sec = F.lag("_post_ts_sec", 1).over(seq_window)
    inter_post_gap = (F.col("_post_ts_sec") - prev_ts_sec).cast(LongType())

    with_base = (
        posts_df.withColumn("_post_ts_sec", F.col("date").cast(LongType()))
        .withColumn(
            "text_length",
            F.length(F.coalesce(F.col("text"), F.lit(""))).cast(IntegerType()),
        )
        .withColumn(
            "is_night_post",
            F.when((F.hour("date") >= 0) & (F.hour("date") < 5), F.lit(1))
            .otherwise(F.lit(0))
            .cast(IntegerType()),
        )
    )

    with_sessions = (
        with_base.withColumn("inter_post_gap_seconds", inter_post_gap)
        .withColumn(
            "is_new_session",
            F.when(
                F.col("inter_post_gap_seconds").isNull()
                | (F.col("inter_post_gap_seconds") > F.lit(session_gap_seconds)),
                F.lit(1),
            )
            .otherwise(F.lit(0))
            .cast(IntegerType()),
        )
        .withColumn(
            "session_index",
            F.sum("is_new_session").over(cumulative_seq_window).cast(LongType()),
        )
    )

    with_rolling = (
        with_sessions.withColumn(
            "night_post_rolling_30d",
            F.avg(F.col("is_night_post").cast(DoubleType())).over(time_window),
        )
        .withColumn(
            "rolling_30d_text_length_mean",
            F.avg(F.col("text_length").cast(DoubleType())).over(time_window),
        )
        .withColumn(
            "rolling_30d_text_length_std",
            F.coalesce(
                F.stddev_samp(F.col("text_length").cast(DoubleType())).over(time_window),
                F.lit(0.0),
            ),
        )
    )

    with_zscore = with_rolling.withColumn(
        "verbosity_zscore",
        F.when(
            F.col("rolling_30d_text_length_std") > F.lit(_STD_EPSILON),
            (F.col("text_length").cast(DoubleType()) - F.col("rolling_30d_text_length_mean"))
            / F.col("rolling_30d_text_length_std"),
        )
        .otherwise(F.lit(0.0))
        .cast(DoubleType()),
    )

    return with_zscore.drop("_post_ts_sec")


def summarize_user_longitudinal_features(
    posts_df: DataFrame,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    session_gap_seconds: int = DEFAULT_SESSION_GAP_SECONDS,
) -> DataFrame:
    """Aggregate circadian, session, and rolling verbosity metrics per subject."""
    windowed = compute_post_window_features(
        posts_df,
        window_days=window_days,
        session_gap_seconds=session_gap_seconds,
    ).withColumn("post_day", F.to_date("date"))

    return (
        windowed.groupBy("subject_id", "label")
        .agg(
            F.count("*").cast(LongType()).alias("post_count"),
            F.min("date").alias("first_post_at"),
            F.max("date").alias("last_post_at"),
            F.datediff(F.max("date"), F.min("date")).cast(IntegerType()).alias("span_days"),
            F.countDistinct("post_day").cast(LongType()).alias("active_days"),
            F.avg("text_length").cast(DoubleType()).alias("mean_text_length"),
            F.coalesce(F.stddev("text_length"), F.lit(0.0))
            .cast(DoubleType())
            .alias("stddev_text_length"),
            F.sum("is_night_post").cast(LongType()).alias("night_post_count"),
            F.max("session_index").cast(LongType()).alias("session_count"),
            F.avg("inter_post_gap_seconds")
            .cast(DoubleType())
            .alias("mean_inter_post_gap_seconds"),
            F.avg("night_post_rolling_30d")
            .cast(DoubleType())
            .alias("mean_night_post_rolling_30d"),
            F.avg("verbosity_zscore").cast(DoubleType()).alias("mean_verbosity_zscore"),
            F.max(F.abs(F.col("verbosity_zscore")))
            .cast(DoubleType())
            .alias("max_abs_verbosity_zscore"),
        )
        .withColumn(
            "posts_per_active_day",
            (F.col("post_count") / F.greatest(F.col("active_days"), F.lit(1))).cast(
                DoubleType()
            ),
        )
        .withColumn(
            "night_post_ratio",
            (F.col("night_post_count") / F.greatest(F.col("post_count"), F.lit(1))).cast(
                DoubleType()
            ),
        )
    )
