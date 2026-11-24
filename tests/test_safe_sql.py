"""Tests for AST-validated Spark SQL execution (`safe_sql.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from src.analytics.safe_sql import (
    SafeSQLError,
    execute_safe_spark_sql,
    register_lakehouse_views,
    validate_and_rewrite_sql,
)
from src.config import Settings
from src.lakehouse.paths import LakehousePaths
from src.spark import write_parquet_zstd


@pytest.mark.parametrize(
    "bad_query",
    [
        "DROP TABLE silver_posts",
        "ALTER TABLE silver_posts ADD COLUMNS (x INT)",
        "INSERT INTO silver_posts VALUES ('s1', 'hi')",
        "DELETE FROM silver_posts WHERE label = 0",
        "CREATE TABLE hacked (id INT)",
        "TRUNCATE TABLE silver_posts",
        "UPDATE silver_posts SET label = 0",
        "MERGE INTO silver_posts t USING other s ON t.subject_id = s.subject_id WHEN MATCHED THEN DELETE",
        "WITH cte AS (SELECT 1) INSERT INTO silver_posts SELECT * FROM cte",
    ],
)
def test_rejects_ddl_and_dml_statements(bad_query: str) -> None:
    with pytest.raises(SafeSQLError) as exc_info:
        validate_and_rewrite_sql(bad_query)
    assert exc_info.value.code == "DISALLOWED_STATEMENT"


def test_rejects_multiple_statements_empty_and_syntax_errors() -> None:
    with pytest.raises(SafeSQLError) as multi_exc:
        validate_and_rewrite_sql("SELECT 1; DROP TABLE silver_posts")
    assert multi_exc.value.code == "MULTIPLE_STATEMENTS"

    with pytest.raises(SafeSQLError) as empty_exc:
        validate_and_rewrite_sql("   ")
    assert empty_exc.value.code == "EMPTY_QUERY"

    with pytest.raises(SafeSQLError) as syntax_exc:
        validate_and_rewrite_sql("SELECT FROM WHERE")
    assert syntax_exc.value.code == "SQL_SYNTAX_ERROR"


@pytest.mark.parametrize(
    "forbidden_table_query",
    [
        "SELECT * FROM csv.`/workspace/.env.example`",
        "SELECT * FROM text.`/workspace/src/config.py`",
        "SELECT * FROM parquet.`/workspace/data`",
        "SELECT * FROM read_files('/workspace/.env.example')",
        "SELECT * FROM default.silver_posts",
        "SELECT * FROM some_other_table",
    ],
)
def test_rejects_file_paths_qualified_and_non_allowlisted_tables(
    forbidden_table_query: str,
) -> None:
    with pytest.raises(SafeSQLError) as exc_info:
        validate_and_rewrite_sql(forbidden_table_query)
    assert exc_info.value.code == "TABLE_NOT_ALLOWED"


def test_limit_injection_and_capping() -> None:
    injected = validate_and_rewrite_sql("SELECT subject_id FROM silver_posts", row_limit=500)
    assert injected.upper().endswith("LIMIT 500")

    capped = validate_and_rewrite_sql("SELECT subject_id FROM silver_posts LIMIT 2000", row_limit=500)
    assert capped.upper().endswith("LIMIT 500")
    assert "2000" not in capped

    preserved = validate_and_rewrite_sql("SELECT subject_id FROM silver_posts LIMIT 25", row_limit=500)
    assert preserved.upper().endswith("LIMIT 25")

    cte_query = validate_and_rewrite_sql(
        "WITH c AS (SELECT subject_id FROM silver_posts) SELECT * FROM C UNION ALL SELECT * FROM c",
        row_limit=500,
    )
    assert cte_query.upper().endswith("LIMIT 500")


@pytest.mark.spark
def test_execute_safe_spark_sql_registers_existing_views_and_enforces_limit(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        spark_storage_uri=str(tmp_path / "data"),
        sql_row_limit=3,
    )
    paths = LakehousePaths.from_settings(settings)

    # Materialize only silver_posts; leave quarantine and gold tables absent.
    sample_df = spark.createDataFrame(
        [(f"s{i}", i % 2) for i in range(10)],
        schema="subject_id STRING, label INT",
    )
    write_parquet_zstd(sample_df, paths.silver_posts, mode="overwrite")

    registered = register_lakehouse_views(spark, settings=settings, paths=paths)
    assert registered == ["silver_posts"]

    result_df = execute_safe_spark_sql(
        "SELECT subject_id, label FROM silver_posts ORDER BY subject_id LIMIT 100",
        spark,
        settings=settings,
        paths=paths,
    )
    assert result_df.count() == 3

    # CTE referencing an allowlisted lakehouse view passes validation and executes.
    cte_df = execute_safe_spark_sql(
        "WITH depressed AS (SELECT subject_id FROM silver_posts WHERE label = 1) "
        "SELECT subject_id FROM depressed ORDER BY subject_id",
        spark,
        settings=settings,
        paths=paths,
    )
    assert cte_df.count() == 3

    # Allowlisted lakehouse view whose Parquet files are absent raises TABLE_NOT_FOUND.
    with pytest.raises(SafeSQLError) as missing_exc:
        execute_safe_spark_sql(
            "SELECT * FROM silver_quarantine",
            spark,
            settings=settings,
            paths=paths,
        )
    assert missing_exc.value.code == "TABLE_NOT_FOUND"
    assert missing_exc.value.to_dict()["code"] == "TABLE_NOT_FOUND"
