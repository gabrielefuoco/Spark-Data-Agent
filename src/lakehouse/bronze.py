"""Bronze layer: immutable raw ingestion of eRISK XML posts and TSV labels.

Reads the source XML writings and ground-truth TSV with explicit schemas and
attaches lineage metadata (``_source_file``, ``_ingested_at``, ``_row_hash``)
without mutating or filtering raw business values.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.config import Settings
from src.lakehouse.paths import LakehousePaths
from src.spark import write_parquet_zstd

RAW_POST_XML_SCHEMA = StructType(
    [
        StructField("TITLE", StringType(), True),
        StructField("DATE", StringType(), True),
        StructField("INFO", StringType(), True),
        StructField("TEXT", StringType(), True),
    ]
)

SUBJECT_LABEL_SCHEMA = StructType(
    [
        StructField("subject_id", StringType(), True),
        StructField("label", IntegerType(), True),
    ]
)

_SUBJECT_ID_REGEX = r"(?:^|/)(subject[A-Za-z0-9]+?)(?:_\d+)?\.xml$"


def _resolve_xml_inputs(posts_input_path: str | Path) -> str | list[str]:
    raw = str(posts_input_path)
    if any(token in raw for token in ("*", "?", "://")):
        return raw

    candidate = Path(raw)
    if candidate.is_dir():
        glob_dirs = sorted(
            {str(xml_file.parent / "*.xml") for xml_file in candidate.rglob("*.xml")}
        )
        if glob_dirs:
            return glob_dirs
        return str(candidate / "*.xml")
    return raw


def _row_hash_expr(columns: tuple[str, ...]) -> F.Column:
    # Coalesce nulls before concat_ws so missing fields cannot shift delimiters.
    normalized = [F.coalesce(F.col(name).cast(StringType()), F.lit("")) for name in columns]
    return F.sha2(F.concat_ws("||", *normalized), 256)


def ingest_bronze_posts(
    spark: SparkSession,
    posts_input_path: str | Path,
    output_path: str | Path,
) -> DataFrame:
    """Ingest raw eRISK XML posts into Bronze Parquet ZSTD with lineage."""
    xml_inputs = _resolve_xml_inputs(posts_input_path)
    raw_df = (
        spark.read.format("xml")
        .option("rootTag", "INDIVIDUAL")
        .option("rowTag", "WRITING")
        .schema(RAW_POST_XML_SCHEMA)
        .load(xml_inputs)
    )

    with_source = raw_df.withColumn("_source_file", F.input_file_name())
    extracted_id = F.regexp_extract(F.col("_source_file"), _SUBJECT_ID_REGEX, 1)

    bronze_df = with_source.select(
        F.when(F.length(extracted_id) > 0, extracted_id)
        .otherwise(F.lit(None).cast(StringType()))
        .alias("subject_id"),
        F.col("TITLE").alias("title"),
        F.col("DATE").alias("date"),
        F.col("INFO").alias("info"),
        F.col("TEXT").alias("text"),
        F.col("_source_file"),
        F.current_timestamp().alias("_ingested_at"),
    ).withColumn("_row_hash", _row_hash_expr(("subject_id", "title", "date", "info", "text")))

    write_parquet_zstd(bronze_df, output_path, mode="overwrite")
    return spark.read.parquet(str(output_path))


def ingest_bronze_subjects(
    spark: SparkSession,
    subjects_input_path: str | Path,
    output_path: str | Path,
) -> DataFrame:
    """Ingest raw subject ground-truth labels TSV into Bronze Parquet ZSTD."""
    raw_df = spark.read.csv(
        str(subjects_input_path),
        sep="\t",
        header=False,
        schema=SUBJECT_LABEL_SCHEMA,
    )

    bronze_df = raw_df.select(
        F.col("subject_id"),
        F.col("label"),
        F.input_file_name().alias("_source_file"),
        F.current_timestamp().alias("_ingested_at"),
    ).withColumn("_row_hash", _row_hash_expr(("subject_id", "label")))

    write_parquet_zstd(bronze_df, output_path, mode="overwrite")
    return spark.read.parquet(str(output_path))


def run_bronze_ingestion(
    spark: SparkSession,
    posts_input_path: str | Path,
    subjects_input_path: str | Path,
    *,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
) -> tuple[DataFrame, DataFrame]:
    """Run Bronze ingestion for posts and subjects using configured paths."""
    resolved = paths or LakehousePaths.from_settings(settings)
    posts_df = ingest_bronze_posts(spark, posts_input_path, resolved.bronze_posts)
    subjects_df = ingest_bronze_subjects(spark, subjects_input_path, resolved.bronze_subjects)
    return posts_df, subjects_df
