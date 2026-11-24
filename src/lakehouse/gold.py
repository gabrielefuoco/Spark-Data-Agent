"""Gold layer: precomputed analytical tables and dataset manifest.

Materializes per-subject longitudinal metrics and cohort-level unigram/bigram
frequencies as ZSTD Parquet, alongside ``gold_dataset_manifest.json``.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.ml.feature import NGram, RegexTokenizer, StopWordsRemover
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel

from src.config import Settings
from src.lakehouse.manifest import DatasetManifest, build_dataset_manifest
from src.lakehouse.paths import LakehousePaths
from src.spark import write_parquet_zstd


def build_user_longitudinal_features(
    silver_posts_df: DataFrame,
    output_path: str | Path,
) -> DataFrame:
    """Aggregate per-subject behavioral and circadian posting metrics."""
    enriched = silver_posts_df.select(
        "subject_id",
        "label",
        "date",
        F.to_date("date").alias("post_day"),
        F.length("text").alias("text_length"),
        F.when((F.hour("date") >= 0) & (F.hour("date") < 5), 1)
        .otherwise(0)
        .alias("is_night_post"),
    )

    features_df = (
        enriched.groupBy("subject_id", "label")
        .agg(
            F.count("*").cast("long").alias("post_count"),
            F.min("date").alias("first_post_at"),
            F.max("date").alias("last_post_at"),
            F.datediff(F.max("date"), F.min("date")).cast("int").alias("span_days"),
            F.countDistinct("post_day").cast("long").alias("active_days"),
            F.avg("text_length").cast("double").alias("mean_text_length"),
            F.coalesce(F.stddev("text_length"), F.lit(0.0)).cast("double").alias("stddev_text_length"),
            F.sum("is_night_post").cast("long").alias("night_post_count"),
        )
        .withColumn(
            "posts_per_active_day",
            (F.col("post_count") / F.greatest(F.col("active_days"), F.lit(1))).cast("double"),
        )
        .withColumn(
            "night_post_ratio",
            (F.col("night_post_count") / F.greatest(F.col("post_count"), F.lit(1))).cast("double"),
        )
    )

    write_parquet_zstd(features_df, output_path, mode="overwrite")
    return silver_posts_df.sparkSession.read.parquet(str(output_path))


def build_cohort_lexical_matrix(
    silver_posts_df: DataFrame,
    output_path: str | Path,
) -> DataFrame:
    """Compute unigram and bigram frequencies per cohort label (without LDA)."""
    combined = silver_posts_df.select(
        "label",
        F.concat_ws(" ", F.coalesce(F.col("title"), F.lit("")), F.col("text")).alias("combined_text"),
    )

    tokenizer = RegexTokenizer(
        inputCol="combined_text",
        outputCol="raw_tokens",
        pattern=r"[a-zA-Z]{3,}",
        gaps=False,
        toLowercase=True,
    )
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="tokens")
    bigrammer = NGram(n=2, inputCol="tokens", outputCol="bigrams")

    tokenized = remover.transform(tokenizer.transform(combined))
    with_ngrams = bigrammer.transform(tokenized)

    unigrams = with_ngrams.select(
        "label",
        F.lit("unigram").alias("ngram_type"),
        F.explode("tokens").alias("ngram"),
    )
    bigrams = with_ngrams.select(
        "label",
        F.lit("bigram").alias("ngram_type"),
        F.explode("bigrams").alias("ngram"),
    )

    cohort_window = Window.partitionBy("label", "ngram_type")
    matrix_df = (
        unigrams.unionByName(bigrams)
        .groupBy("label", "ngram_type", "ngram")
        .agg(F.count("*").cast("long").alias("frequency"))
        .withColumn(
            "relative_frequency",
            (F.col("frequency") / F.sum("frequency").over(cohort_window)).cast("double"),
        )
    )

    write_parquet_zstd(matrix_df, output_path, mode="overwrite")
    return silver_posts_df.sparkSession.read.parquet(str(output_path))


def build_gold(
    spark: SparkSession,
    *,
    silver_posts_path: str | Path | None = None,
    silver_quarantine_path: str | Path | None = None,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
) -> tuple[DataFrame, DataFrame, DatasetManifest]:
    """Materialize all Gold tables and the static dataset manifest."""
    resolved = paths or LakehousePaths.from_settings(settings)
    silver_df = spark.read.parquet(str(silver_posts_path or resolved.silver_posts))
    quarantine_df = spark.read.parquet(str(silver_quarantine_path or resolved.silver_quarantine))

    # Reused across longitudinal features, lexical matrix, and manifest aggregations.
    silver_df.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        build_user_longitudinal_features(silver_df, resolved.gold_user_longitudinal_features)
        build_cohort_lexical_matrix(silver_df, resolved.gold_cohort_lexical_matrix)
        manifest = build_dataset_manifest(silver_df, quarantine_df, resolved.gold_dataset_manifest)
    finally:
        silver_df.unpersist()

    return (
        spark.read.parquet(resolved.gold_user_longitudinal_features),
        spark.read.parquet(resolved.gold_cohort_lexical_matrix),
        manifest,
    )
