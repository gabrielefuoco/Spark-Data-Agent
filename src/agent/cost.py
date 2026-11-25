"""SQL computational cost estimator driving the HITL approval gate.

Uses Spark's ``explain(mode="cost")`` when an active/injected ``SparkSession``
is available, and falls back to a deterministic ``sqlglot`` AST heuristic when
running offline without Spark (e.g. unit tests or dry planning).
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import Any

import sqlglot
from pyspark.errors import AnalysisException, ParseException
from pyspark.sql import SparkSession
from sqlglot import exp

from src.analytics.safe_sql import SafeSQLError
from src.config import Settings, get_settings

_TABLE_BASE_COSTS: dict[str, float] = {
    "silver_posts": 1.2,
    "silver_quarantine": 0.3,
    "gold_user_longitudinal_features": 0.2,
    "gold_cohort_lexical_matrix": 0.2,
    "gold_cohort_topic_matrix": 0.2,
}

_SPARK_JOB_COSTS: dict[str, tuple[float, str]] = {
    "topic_model": (2.5, "spark_ml_lda_full_recompute (+2.50)"),
    "circadian_features": (1.5, "spark_window_circadian_features (+1.50)"),
}

_SIZE_BYTES_PATTERN = re.compile(
    r"sizeInBytes\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)",
    re.IGNORECASE,
)

_UNIT_TO_MIB: dict[str, float] = {
    "b": 1.0 / (1024.0 * 1024.0),
    "kib": 1.0 / 1024.0,
    "kb": 1.0 / 1024.0,
    "mib": 1.0,
    "mb": 1.0,
    "gib": 1024.0,
    "gb": 1024.0,
    "tib": 1024.0 * 1024.0,
    "tb": 1024.0 * 1024.0,
}

# 50 MiB of estimated Spark scan corresponds to cost 1.0 (the default HITL threshold).
_MIB_PER_COST_UNIT = 50.0


def _extract_spark_size_mib(explain_text: str) -> float | None:
    """Parse the largest ``sizeInBytes=...`` reported in Spark's cost explain plan."""
    matches = _SIZE_BYTES_PATTERN.findall(explain_text)
    if not matches:
        return None
    sizes_mib = [
        float(val) * _UNIT_TO_MIB.get(unit.lower(), 1.0)
        for val, unit in matches
    ]
    return max(sizes_mib)


def estimate_ast_cost(query: str) -> tuple[float, list[str]]:
    """Compute a deterministic computational cost score from a SQL query AST."""
    tree = sqlglot.parse_one(query, read="spark")
    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }

    referenced_tables = [
        table.name.lower()
        for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    ]

    reasons: list[str] = []
    if referenced_tables:
        base_cost = max(_TABLE_BASE_COSTS.get(name, 0.5) for name in referenced_tables)
        for name in sorted(set(referenced_tables)):
            reasons.append(f"table:{name} (base={_TABLE_BASE_COSTS.get(name, 0.5):.2f})")
    else:
        base_cost = 0.1
        reasons.append("literal_or_cte_only (base=0.10)")

    score = base_cost

    if any(tree.find_all(exp.Join)):
        score += 0.5
        reasons.append("join_operation (+0.50)")

    if any(tree.find_all(exp.Group)) or any(tree.find_all(exp.Window)):
        score += 0.3
        reasons.append("aggregation_or_window (+0.30)")

    # sqlglot accepts a tuple of node types at runtime but types `find_all` for a
    # single type, so the three text-pattern scans are asked for separately.
    if any(
        any(tree.find_all(node_type)) for node_type in (exp.Like, exp.ILike, exp.RegexpLike)
    ):
        score += 0.25
        reasons.append("text_pattern_scan (+0.25)")

    if "silver_posts" in referenced_tables:
        where_nodes = list(tree.find_all(exp.Where))
        where_columns = {
            col.name.lower()
            for where in where_nodes
            for col in where.find_all(exp.Column)
            if col.name
        }
        has_star = any(isinstance(sel, exp.Star) for sel in tree.find_all(exp.Star))
        if not where_nodes or has_star:
            score += 0.2
            reasons.append("silver_posts_unfiltered_or_wide_scan (+0.20)")
        elif "label" in where_columns and "year" in where_columns:
            score -= 0.3
            reasons.append("silver_posts_partition_pruned_label_year (-0.30)")
        elif "label" in where_columns or "year" in where_columns:
            score -= 0.15
            reasons.append("silver_posts_partial_partition_pruned (-0.15)")

    return round(max(score, 0.05), 4), reasons


def estimate_sql_cost(
    query: str,
    *,
    settings: Settings | None = None,
    spark: SparkSession | None = None,
) -> dict[str, Any]:
    """Estimate SQL execution cost via Spark ``EXPLAIN COST`` or AST heuristics.

    When ``spark`` is provided, ``spark.sql(query).explain(mode="cost")`` is
    inspected and combined with the AST score so physical scan sizes and
    logical query complexity are both reflected. Without Spark, the pure AST
    heuristic runs offline.
    """
    cfg = settings or get_settings()
    ast_score, reasons = estimate_ast_cost(query)

    if spark is None:
        return {
            "estimated_cost": ast_score,
            "threshold": cfg.hitl_cost_threshold,
            "triggers_hitl": ast_score > cfg.hitl_cost_threshold,
            "method": "ast_heuristic",
            "reasons": reasons,
            "explain_plan": None,
        }

    buffer = io.StringIO()
    try:
        df = spark.sql(query)
        with contextlib.redirect_stdout(buffer):
            df.explain(mode="cost")
    except AnalysisException as exc:
        msg = str(exc)
        code = (
            "TABLE_NOT_FOUND"
            if "TABLE_OR_VIEW_NOT_FOUND" in msg or "Table or view not found" in msg
            else "SPARK_ANALYSIS_ERROR"
        )
        raise SafeSQLError(code, msg, query=query) from exc
    except ParseException as exc:
        raise SafeSQLError("SQL_SYNTAX_ERROR", str(exc), query=query) from exc

    plan_text = buffer.getvalue().strip()
    size_mib = _extract_spark_size_mib(plan_text)
    spark_score = round(size_mib / _MIB_PER_COST_UNIT, 4) if size_mib is not None else 0.0
    if size_mib is not None:
        reasons = [*reasons, f"spark_explain_size={size_mib:.2f}MiB (score={spark_score:.2f})"]

    combined = round(max(ast_score, spark_score), 4)
    return {
        "estimated_cost": combined,
        "threshold": cfg.hitl_cost_threshold,
        "triggers_hitl": combined > cfg.hitl_cost_threshold,
        "method": "spark_explain",
        "reasons": reasons,
        "explain_plan": plan_text[:2000] if plan_text else None,
    }


def estimate_job_cost(
    job_name: str,
    params: dict[str, Any] | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Estimate computational cost for non-SQL Spark jobs driving the HITL gate."""
    cfg = settings or get_settings()
    normalized = job_name.strip().lower()
    base_cost, reason = _SPARK_JOB_COSTS.get(
        normalized, (1.5, f"spark_job:{normalized} (+1.50)")
    )
    reasons = [f"spark_job:{normalized} (base={base_cost:.2f})", reason]
    if params:
        reasons.append(f"params={','.join(sorted(str(k) for k in params))}")

    estimated = round(base_cost, 4)
    return {
        "estimated_cost": estimated,
        "threshold": cfg.hitl_cost_threshold,
        "triggers_hitl": estimated > cfg.hitl_cost_threshold,
        "method": "ast_heuristic",
        "reasons": reasons,
        "explain_plan": None,
    }
