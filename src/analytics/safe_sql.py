"""AST-validated Spark SQL execution for agent analytical queries.

Validates every query with ``sqlglot`` before touching Spark: rejects DDL/DML
and multi-statement payloads, enforces ``Settings.sql_fetch_limit`` (injecting
or lowering ``LIMIT``), registers available Silver/Gold Parquet views, and
raises a structured ``SafeSQLError`` with machine-readable error codes for retry
loops.
"""

from __future__ import annotations

from pathlib import Path

import sqlglot
from pyspark.errors import AnalysisException, ParseException
from pyspark.sql import DataFrame, SparkSession
from sqlglot import exp
from sqlglot.errors import ParseError as SqlglotParseError

from src.config import Settings, get_settings
from src.lakehouse.paths import LakehousePaths

_LAKEHOUSE_VIEW_ATTRS: tuple[str, ...] = (
    "silver_posts",
    "silver_quarantine",
    "gold_user_longitudinal_features",
    "gold_cohort_lexical_matrix",
    "gold_cohort_topic_matrix",
)
LAKEHOUSE_VIEW_NAMES: frozenset[str] = frozenset(_LAKEHOUSE_VIEW_ATTRS)

# SparkConf key recording which storage root the current temporary views belong to.
_VIEWS_ROOT_KEY = "spark.dataagent.views_root"

_ALLOWED_ROOT_TYPES = (exp.Select, exp.SetOperation, exp.Subquery)
_FORBIDDEN_EXPRESSIONS = (
    exp.Drop,
    exp.Alter,
    exp.Insert,
    exp.Delete,
    exp.Create,
    exp.Update,
    exp.Merge,
    exp.TruncateTable,
    exp.Command,
    exp.Into,
)


class SafeSQLError(ValueError):
    """Structured SQL validation or execution error surfaced to the LangGraph agent."""

    def __init__(self, code: str, message: str, *, query: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.query = query

    def to_dict(self) -> dict[str, str | None]:
        """Serialize error details for deterministic LLM self-correction prompts."""
        return {"code": self.code, "message": self.message, "query": self.query}


def _lakehouse_view_paths(paths: LakehousePaths) -> dict[str, str]:
    return {name: getattr(paths, name) for name in _LAKEHOUSE_VIEW_ATTRS}


def _is_local_uri(uri: str) -> bool:
    return "://" not in uri


def register_lakehouse_views(
    spark: SparkSession,
    *,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
) -> list[str]:
    """Register existing Silver and Gold Parquet tables as Spark temporary views.

    Registration is skipped when this session already registered the same storage
    root: reading a Parquet tree to infer its schema and list its files costs
    ~2s per table (9s for the five lakehouse views), and the tool loop was paying
    that on **every** call to run a query that takes 0.9s.

    The marker is keyed on the resolved storage root, not just the view names, so
    a test with its own `tmp_path` re-registers instead of querying another
    fixture's data.

    // ponytail: a view stays bound to the files it saw when registered. Rebuild
    // Gold inside the same session and you must reopen the session; the ETL runs
    // as a separate job, so this has not bitten yet.
    """
    resolved = paths or LakehousePaths.from_settings(settings)
    views = _lakehouse_view_paths(resolved)
    storage_root = str(resolved.storage_root)

    if spark.conf.get(_VIEWS_ROOT_KEY, None) == storage_root:
        already_registered = True
    else:
        already_registered = False

    registered: list[str] = []

    for view_name, table_path in views.items():
        # Per view, not all-or-nothing: a lakehouse missing one Gold table must
        # still skip re-reading the four that are there.
        if already_registered and spark.catalog.tableExists(view_name):
            registered.append(view_name)
            continue

        if _is_local_uri(table_path):
            if not Path(table_path).exists():
                spark.catalog.dropTempView(view_name)
                continue
            spark.read.parquet(table_path).createOrReplaceTempView(view_name)
            registered.append(view_name)
            continue

        try:
            spark.read.parquet(table_path).createOrReplaceTempView(view_name)
            registered.append(view_name)
        except AnalysisException:
            spark.catalog.dropTempView(view_name)
            continue

    spark.conf.set(_VIEWS_ROOT_KEY, storage_root)
    return registered


def validate_and_rewrite_sql(
    query: str,
    *,
    row_limit: int = 50_000,
    allowed_tables: frozenset[str] | None = None,
) -> str:
    """Validate Spark SQL AST with sqlglot, enforce table allowlist and ``LIMIT <= row_limit``."""
    if not isinstance(query, str) or not query.strip():
        raise SafeSQLError("EMPTY_QUERY", "SQL query must be a non-empty string.", query=query)

    if row_limit <= 0:
        raise SafeSQLError("INVALID_ROW_LIMIT", "row_limit must be positive.", query=query)

    try:
        parsed = [stmt for stmt in sqlglot.parse(query, read="spark") if stmt is not None]
    except SqlglotParseError as exc:
        raise SafeSQLError(
            "SQL_SYNTAX_ERROR",
            f"SQL syntax error: {exc}",
            query=query,
        ) from exc

    if not parsed:
        raise SafeSQLError("EMPTY_QUERY", "SQL query parsed to zero statements.", query=query)

    if len(parsed) > 1:
        raise SafeSQLError(
            "MULTIPLE_STATEMENTS",
            f"Only a single SQL SELECT statement is permitted; received {len(parsed)}.",
            query=query,
        )

    tree = parsed[0]
    if not isinstance(tree, _ALLOWED_ROOT_TYPES):
        raise SafeSQLError(
            "DISALLOWED_STATEMENT",
            f"Only SELECT/WITH/UNION queries are allowed; got {type(tree).__name__}.",
            query=query,
        )

    if isinstance(tree, exp.Subquery) and not isinstance(tree.this, (exp.Select, exp.SetOperation)):
        raise SafeSQLError(
            "DISALLOWED_STATEMENT",
            f"Subquery root must wrap SELECT/UNION; got {type(tree.this).__name__}.",
            query=query,
        )

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_EXPRESSIONS):
            raise SafeSQLError(
                "DISALLOWED_STATEMENT",
                f"Forbidden SQL operation detected in AST: {type(node).__name__}.",
                query=query,
            )

    base_allowed = LAKEHOUSE_VIEW_NAMES if allowed_tables is None else allowed_tables
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name}
    effective_allowed = {name.lower() for name in base_allowed} | cte_names

    for table in tree.find_all(exp.Table):
        table_ref = table.sql(dialect="spark")
        if table.catalog or table.db:
            raise SafeSQLError(
                "TABLE_NOT_ALLOWED",
                f"Qualified table or file-path reference is not allowed: {table_ref}",
                query=query,
            )
        if not table.name or table.name.lower() not in effective_allowed:
            raise SafeSQLError(
                "TABLE_NOT_ALLOWED",
                f"Table reference is not in the allowlist: {table_ref}",
                query=query,
            )

    limit_node = tree.args.get("limit")
    if limit_node is None:
        tree = tree.limit(row_limit)
    else:
        limit_expr = limit_node.expression
        if not isinstance(limit_expr, exp.Literal) or limit_expr.is_string:
            raise SafeSQLError(
                "INVALID_LIMIT",
                "LIMIT must be a non-negative integer literal.",
                query=query,
            )
        try:
            requested_limit = int(limit_expr.this)
        except (TypeError, ValueError) as exc:
            raise SafeSQLError(
                "INVALID_LIMIT",
                "LIMIT must be a valid integer literal.",
                query=query,
            ) from exc

        if requested_limit < 0:
            raise SafeSQLError(
                "INVALID_LIMIT",
                "LIMIT must be non-negative.",
                query=query,
            )

        if requested_limit > row_limit:
            tree = tree.limit(row_limit)

    return tree.sql(dialect="spark")


def map_spark_analysis_exception(
    exc: AnalysisException,
    *,
    query: str | None = None,
) -> SafeSQLError:
    """Classify a Spark ``AnalysisException`` into a structured ``SafeSQLError``."""
    msg = str(exc)
    code = (
        "TABLE_NOT_FOUND"
        if "TABLE_OR_VIEW_NOT_FOUND" in msg or "Table or view not found" in msg
        else "SPARK_ANALYSIS_ERROR"
    )
    return SafeSQLError(code, msg, query=query)


def execute_safe_spark_sql(
    query: str,
    spark: SparkSession,
    *,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
    register_views: bool = True,
    row_limit: int | None = None,
) -> DataFrame:
    """Validate ``query`` via sqlglot, register existing views, and execute in Spark.

    Injects ``Settings.sql_fetch_limit`` unless an explicit ``row_limit`` override
    is supplied by the caller.
    """
    cfg = settings or get_settings()
    effective_limit = cfg.sql_fetch_limit if row_limit is None else row_limit
    safe_sql = validate_and_rewrite_sql(query, row_limit=effective_limit)

    if register_views:
        register_lakehouse_views(spark, settings=cfg, paths=paths)

    try:
        return spark.sql(safe_sql)
    except AnalysisException as exc:
        raise map_spark_analysis_exception(exc, query=safe_sql) from exc
    except ParseException as exc:
        raise SafeSQLError("SQL_SYNTAX_ERROR", str(exc), query=safe_sql) from exc
