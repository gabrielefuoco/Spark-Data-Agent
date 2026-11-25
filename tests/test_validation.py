"""Spark integration tests for Pandera Silver contracts and quarantine routing."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType
from src.lakehouse.bronze import run_bronze_ingestion
from src.lakehouse.silver import REASON_CONTRACT_VIOLATION, build_silver
from src.lakehouse.validation import (
    VIOLATION_INVALID_LABEL,
    VIOLATION_NULL_DATE,
    VIOLATION_NULL_OR_BLANK_SUBJECT_ID,
    VIOLATION_NULL_OR_EMPTY_TEXT,
    SchemaContractError,
    find_post_contract_violations,
)

_TEST_POST_SCHEMA = StructType(
    [
        StructField("row_id", IntegerType(), False),
        StructField("subject_id", StringType(), True),
        StructField("date", TimestampType(), True),
        StructField("text", StringType(), True),
        StructField("label", IntegerType(), True),
    ]
)


@pytest.mark.spark
def test_validation_returns_empty_for_valid_posts(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [
            (1, "subject001", datetime(2020, 1, 1, 12, 0, 0), "Valid post text", 0),
            (2, "subject002", datetime(2020, 1, 2, 3, 15, 0), "Another valid post", 1),
        ],
        schema=_TEST_POST_SCHEMA,
    )

    violations = find_post_contract_violations(df, key_col="row_id")
    assert violations.count() == 0
    assert violations.columns == ["row_id", "violation_reason"]


@pytest.mark.spark
def test_validation_flags_each_contract_rule_with_row_key(spark: SparkSession) -> None:
    ts = datetime(2020, 5, 1, 10, 0, 0)
    df = spark.createDataFrame(
        [
            (1, "subject001", ts, "Healthy row", 1),
            (2, None, ts, "Missing subject id", 0),
            (3, "", ts, "Blank subject id", 0),
            (4, "subject004", None, "Missing timestamp", 1),
            (5, "subject005", ts, None, 1),
            (6, "subject006", ts, "", 0),
            (7, "subject007", ts, "Out of domain label", 2),
            (8, "subject008", ts, "Null label", None),
        ],
        schema=_TEST_POST_SCHEMA,
    )

    violations = find_post_contract_violations(df, key_col="row_id")
    by_row = {row["row_id"]: row["violation_reason"] for row in violations.collect()}

    assert set(by_row.keys()) == {2, 3, 4, 5, 6, 7, 8}
    assert by_row[2] == VIOLATION_NULL_OR_BLANK_SUBJECT_ID
    assert by_row[3] == VIOLATION_NULL_OR_BLANK_SUBJECT_ID
    assert by_row[4] == VIOLATION_NULL_DATE
    assert by_row[5] == VIOLATION_NULL_OR_EMPTY_TEXT
    assert by_row[6] == VIOLATION_NULL_OR_EMPTY_TEXT
    assert by_row[7] == VIOLATION_INVALID_LABEL
    assert by_row[8] == VIOLATION_INVALID_LABEL

    kept_ids = [r["row_id"] for r in df.join(violations, on="row_id", how="left_anti").collect()]
    assert kept_ids == [1]


@pytest.mark.spark
def test_validation_rejects_structural_type_mismatch(spark: SparkSession) -> None:
    wrong_schema = StructType(
        [
            StructField("row_id", IntegerType(), False),
            StructField("subject_id", StringType(), True),
            StructField("date", TimestampType(), True),
            StructField("text", StringType(), True),
            StructField("label", StringType(), True),  # wrong type: should be IntegerType
        ]
    )
    df = spark.createDataFrame(
        [(1, "subject001", datetime(2020, 1, 1), "Text", "1")],
        schema=wrong_schema,
    )

    with pytest.raises(SchemaContractError):
        find_post_contract_violations(df, key_col="row_id")


@pytest.mark.spark
def test_silver_accounting_invariant_with_contract_violation_in_mix(
    spark: SparkSession,
    lakehouse_env: tuple,
    monkeypatch: pytest.MonkeyPatch,
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

    # Flag the first valid Silver candidate deterministically to exercise contract quarantine routing.
    def _inject_contract_violation(df: DataFrame, *, key_col: str = "_row_hash") -> DataFrame:
        first_row = df.orderBy("_row_hash").select(F.col(key_col)).limit(1)
        return first_row.withColumn("violation_reason", F.lit("synthetic_contract_breach"))

    monkeypatch.setattr(
        "src.lakehouse.silver.find_post_contract_violations", _inject_contract_violation
    )

    silver_df, quarantine_df = build_silver(spark, settings=settings, paths=paths)
    silver_count = silver_df.count()
    quarantine_count = quarantine_df.count()

    assert bronze_count == 84
    assert silver_count == 79
    assert quarantine_count == 5
    assert bronze_count == silver_count + quarantine_count

    contract_rows = quarantine_df.filter(
        F.col("_rejection_reason") == f"{REASON_CONTRACT_VIOLATION}:synthetic_contract_breach"
    )
    assert contract_rows.count() == 1
