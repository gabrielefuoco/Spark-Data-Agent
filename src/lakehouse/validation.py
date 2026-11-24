"""Pandera data contracts for the Silver posts layer.

Provides ``PostSilverSchema`` (backed by ``pandera.pyspark``) and row-level
violation extraction so ``build_silver`` can route contract-failing rows to
``silver/quarantine/`` by anti-join instead of failing only at table level.
"""

from __future__ import annotations

from typing import Any, cast

# pandera.pyspark imports pyspark.pandas, which needs pandas<3, numpy<2, and pyarrow<17
# pinned in pyproject.toml for PySpark 3.5 compatibility.
import pandera.pyspark as pap
from pandera.typing.pyspark import DataFrame as PanderaSparkDataFrame
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType

VIOLATION_NULL_OR_BLANK_SUBJECT_ID = "null_or_blank_subject_id"
VIOLATION_NULL_DATE = "null_date"
VIOLATION_NULL_OR_EMPTY_TEXT = "null_or_empty_text"
VIOLATION_INVALID_LABEL = "invalid_label"


class SchemaContractError(ValueError):
    """Raised when a DataFrame fails structural schema expectations (missing/wrong-type columns)."""


class PostSilverSchema(pap.DataFrameModel):
    """Formal Pandera contract for validated atomic Silver posts."""

    subject_id: StringType = pap.Field(nullable=False, str_length={"min_value": 1})
    date: TimestampType = pap.Field(nullable=False)
    text: StringType = pap.Field(nullable=False, str_length={"min_value": 1})
    label: IntegerType = pap.Field(nullable=False, isin=[0, 1])

    class Config:
        """Keep extra Silver lineage/partition columns allowed without implicit coercion."""

        coerce = False
        strict = False


def _extract_structural_errors(errors: dict[str, Any]) -> list[dict[str, Any]]:
    schema_errors = errors.get("SCHEMA", {})
    structural: list[dict[str, Any]] = []
    for key in ("COLUMN_NOT_IN_DATAFRAME", "WRONG_DATATYPE"):
        structural.extend(schema_errors.get(key, []))
    return structural


def find_post_contract_violations(
    posts_df: DataFrame,
    *,
    key_col: str = "_row_hash",
) -> DataFrame:
    """Validate ``posts_df`` against ``PostSilverSchema`` and return offending row keys.

    Because Pandera's PySpark backend reports column/check-level failures rather
    than row identifiers, this runs lazy Pandera validation first and, whenever
    data or nullability checks fail, evaluates the corresponding row predicates
    so callers can anti-join offending rows out of Silver and into quarantine.

    Raises:
        ValueError: If ``key_col`` is absent from ``posts_df``.
        SchemaContractError: If required columns are missing or have wrong Spark types.
    """
    if key_col not in posts_df.columns:
        raise ValueError(f"Key column {key_col!r} not found in DataFrame columns: {posts_df.columns}")

    # Pandera's PySpark typing only admits its own DataFrame wrapper, while the
    # backend duck-types a plain pyspark DataFrame at runtime.
    validated = PostSilverSchema.validate(cast("PanderaSparkDataFrame[Any]", posts_df), lazy=True)
    pandera_attr = getattr(validated, "pandera", None)
    errors: dict[str, Any] = dict(getattr(pandera_attr, "errors", {}) or {})

    structural = _extract_structural_errors(errors)
    if structural:
        messages = "; ".join(str(item.get("error", item)) for item in structural)
        raise SchemaContractError(f"PostSilverSchema structural mismatch: {messages}")

    if not errors:
        empty_schema = StructType(
            [
                posts_df.schema[key_col],
                StructField("violation_reason", StringType(), False),
            ]
        )
        return posts_df.sparkSession.createDataFrame([], schema=empty_schema)

    reasons = F.array_compact(
        F.array(
            F.when(
                F.col("subject_id").isNull() | (F.length(F.col("subject_id")) == 0),
                F.lit(VIOLATION_NULL_OR_BLANK_SUBJECT_ID),
            ),
            F.when(F.col("date").isNull(), F.lit(VIOLATION_NULL_DATE)),
            F.when(
                F.col("text").isNull() | (F.length(F.col("text")) == 0),
                F.lit(VIOLATION_NULL_OR_EMPTY_TEXT),
            ),
            F.when(
                F.col("label").isNull() | ~F.col("label").isin(0, 1),
                F.lit(VIOLATION_INVALID_LABEL),
            ),
        )
    )

    return (
        posts_df.select(F.col(key_col), F.concat_ws(";", reasons).alias("violation_reason"))
        .filter(F.length(F.col("violation_reason")) > 0)
    )
