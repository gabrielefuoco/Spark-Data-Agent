"""Silver layer: atomic posts with deterministic dedup and dead-letter quarantine.

Every Bronze post row is routed either to partitioned Silver posts
(``partitionBy("label", "year")``) or to ``silver/quarantine/`` with an explicit
``_rejection_reason`` so no record is ever silently dropped.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.storagelevel import StorageLevel

from src.config import Settings
from src.lakehouse.paths import LakehousePaths
from src.lakehouse.validation import find_post_contract_violations
from src.spark import write_parquet_zstd

REASON_MISSING_SUBJECT_ID = "missing_subject_id"
REASON_MALFORMED_DATE = "malformed_date"
REASON_NULL_OR_BLANK_TEXT = "null_or_blank_text"
REASON_ORPHAN_SUBJECT_ID = "orphan_subject_id"
REASON_DUPLICATE_POST = "duplicate_post"
REASON_CONTRACT_VIOLATION = "contract_violation"


def _clean_text_expr(column_name: str) -> F.Column:
    return F.regexp_replace(F.trim(F.col(column_name)), r"\s+", " ")


def _deduplicate_subjects(subjects_df: DataFrame) -> DataFrame:
    valid_subjects = subjects_df.filter(
        F.col("subject_id").isNotNull()
        & (F.length(F.trim(F.col("subject_id"))) > 0)
        & F.col("label").isin(0, 1)
    ).withColumn("subject_id", F.trim(F.col("subject_id")))

    window = Window.partitionBy("subject_id").orderBy(
        F.col("_ingested_at").asc_nulls_last(),
        F.col("_source_file").asc_nulls_last(),
    )
    return (
        valid_subjects.withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") == 1)
        .select("subject_id", F.col("label").cast(IntegerType()).alias("label"))
    )


def build_silver(
    spark: SparkSession,
    *,
    bronze_posts_path: str | Path | None = None,
    bronze_subjects_path: str | Path | None = None,
    silver_posts_path: str | Path | None = None,
    silver_quarantine_path: str | Path | None = None,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
) -> tuple[DataFrame, DataFrame]:
    """Transform Bronze posts and subjects into Silver posts and Quarantine."""
    resolved = paths or LakehousePaths.from_settings(settings)
    posts_in = str(bronze_posts_path or resolved.bronze_posts)
    subjects_in = str(bronze_subjects_path or resolved.bronze_subjects)
    posts_out = str(silver_posts_path or resolved.silver_posts)
    quarantine_out = str(silver_quarantine_path or resolved.silver_quarantine)

    bronze_posts = spark.read.parquet(posts_in)
    bronze_subjects = _deduplicate_subjects(spark.read.parquet(subjects_in))

    normalized_posts = bronze_posts.select(
        F.when(F.length(F.trim(F.col("subject_id"))) > 0, F.trim(F.col("subject_id")))
        .otherwise(F.lit(None).cast(StringType()))
        .alias("subject_id"),
        _clean_text_expr("title").alias("title"),
        F.col("date").alias("raw_date"),
        F.to_timestamp(F.trim(F.col("date")), "yyyy-MM-dd HH:mm:ss").alias("date"),
        _clean_text_expr("info").alias("info"),
        _clean_text_expr("text").alias("text"),
        F.col("_source_file"),
        F.col("_ingested_at"),
        F.col("_row_hash"),
    )

    joined = normalized_posts.join(F.broadcast(bronze_subjects), on="subject_id", how="left")

    validation_error = (
        F.when(F.col("subject_id").isNull(), F.lit(REASON_MISSING_SUBJECT_ID))
        .when(F.col("date").isNull(), F.lit(REASON_MALFORMED_DATE))
        .when(
            F.col("text").isNull() | (F.length(F.col("text")) == 0),
            F.lit(REASON_NULL_OR_BLANK_TEXT),
        )
        .when(F.col("label").isNull() | ~F.col("label").isin(0, 1), F.lit(REASON_ORPHAN_SUBJECT_ID))
        .otherwise(F.lit(None).cast(StringType()))
    )

    dedup_window = Window.partitionBy("subject_id", "date", "_row_hash").orderBy(
        F.col("_ingested_at").asc_nulls_last(),
        F.col("_source_file").asc_nulls_last(),
    )
    dedup_rank = F.when(validation_error.isNull(), F.row_number().over(dedup_window)).otherwise(
        F.lit(1)
    )

    classified = (
        joined.withColumn("_validation_error", validation_error)
        .withColumn("_dedup_rank", dedup_rank)
        .withColumn(
            "_rejection_reason",
            F.coalesce(
                F.col("_validation_error"),
                F.when(F.col("_dedup_rank") > 1, F.lit(REASON_DUPLICATE_POST)),
            ),
        )
    )

    # Reused across two diverging writes (silver posts vs dead-letter quarantine).
    classified.persist(StorageLevel.MEMORY_AND_DISK)
    silver_columns = (
        "subject_id",
        "title",
        "text",
        "info",
        "date",
        "year",
        "label",
        "_source_file",
        "_ingested_at",
        "_row_hash",
    )
    quarantine_columns = (
        "subject_id",
        "title",
        "raw_date",
        "info",
        "text",
        "label",
        "_source_file",
        "_ingested_at",
        "_row_hash",
        "_rejection_reason",
    )

    # Persist candidates so monotonically_increasing_id stays stable across validation and anti-join.
    silver_candidates = (
        classified.filter(F.col("_rejection_reason").isNull())
        .withColumn("_validation_row_id", F.monotonically_increasing_id())
        .withColumn("year", F.year("date").cast(IntegerType()))
        .withColumn("label", F.col("label").cast(IntegerType()))
    )
    silver_candidates.persist(StorageLevel.MEMORY_AND_DISK)
    try:
        violations_df = find_post_contract_violations(
            silver_candidates,
            key_col="_validation_row_id",
        )

        silver_df = silver_candidates.join(
            violations_df,
            on="_validation_row_id",
            how="left_anti",
        ).select(*silver_columns)

        pre_quarantine_df = classified.filter(F.col("_rejection_reason").isNotNull()).select(
            *quarantine_columns
        )
        contract_quarantine_df = (
            silver_candidates.join(violations_df, on="_validation_row_id", how="inner")
            .withColumn(
                "_rejection_reason",
                F.concat(F.lit(f"{REASON_CONTRACT_VIOLATION}:"), F.col("violation_reason")),
            )
            .select(*quarantine_columns)
        )
        quarantine_df = pre_quarantine_df.unionByName(contract_quarantine_df)

        write_parquet_zstd(
            silver_df,
            posts_out,
            mode="overwrite",
            partition_by=("label", "year"),
        )
        write_parquet_zstd(quarantine_df, quarantine_out, mode="overwrite")
    finally:
        silver_candidates.unpersist()
        classified.unpersist()

    return spark.read.parquet(posts_out), spark.read.parquet(quarantine_out)
