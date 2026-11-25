"""Tool schema definitions and call canonicalization for the Data Agent loop."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool


@tool
def run_safe_sql(query: str) -> str:
    """Execute a read-only Spark SQL SELECT/WITH/UNION query over allowlisted lakehouse views."""
    return query


@tool
def read_dataset_manifest() -> str:
    """Read precomputed global corpus counts, date range, and quarantine totals from gold_dataset_manifest.json without starting Spark."""
    return "read_dataset_manifest"


@tool
def run_spark_job(job: str, params: dict[str, Any] | None = None) -> str:
    """Run a non-SQL Spark analytics job ('circadian_features' or 'topic_model') over silver_posts."""
    return json.dumps({"job": job, "params": params or {}}, sort_keys=True)


AGENT_TOOLS: tuple[BaseTool, ...] = (
    run_safe_sql,
    read_dataset_manifest,
    run_spark_job,
)


def canonical_tool_signature(name: str, arguments: dict[str, Any] | None) -> str:
    """Return a deterministic signature for detecting consecutive identical tool calls."""
    normalized_args = dict(arguments) if isinstance(arguments, dict) else {}
    return f"{name}:{json.dumps(normalized_args, sort_keys=True, default=str)}"
