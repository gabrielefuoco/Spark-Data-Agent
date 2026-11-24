"""Cyclic LangGraph StateGraph orchestrating routing, SQL safety, HITL, and critique."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pyspark.sql import SparkSession

from src.agent.cost import estimate_sql_cost
from src.agent.guards import StatisticalGuardResult, evaluate_statistical_guard
from src.agent.models import create_chat_model
from src.agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    build_critic_system_prompt,
    build_critic_user_prompt,
    build_planner_user_prompt,
    format_manifest_answer,
)
from src.agent.state import DataAgentState
from src.analytics.safe_sql import SafeSQLError, execute_safe_spark_sql, validate_and_rewrite_sql
from src.config import Settings, get_settings
from src.lakehouse.manifest import DatasetManifest, load_manifest
from src.lakehouse.paths import LakehousePaths

SqlExecutor = Callable[[str, Settings], dict[str, Any] | list[dict[str, Any]]]
CostEstimator = Callable[[str, Settings], dict[str, Any]]
ManifestLoader = Callable[[], DatasetManifest]

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_EXCLUDED_METRIC_COLUMNS = frozenset(
    {"label", "year", "subject_id", "first_post_at", "last_post_at"}
)


def _to_json_value(value: Any) -> Any:
    """Convert Spark/Python values to JSON-safe primitives for SQLite checkpointing."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_to_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    return str(value)


def make_spark_sql_executor(
    spark: SparkSession,
    *,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
) -> SqlExecutor:
    """Wrap ``execute_safe_spark_sql`` into a JSON-serializable executor callable."""

    def _execute(query: str, cfg: Settings) -> dict[str, Any]:
        resolved_settings = settings or cfg
        df = execute_safe_spark_sql(
            query,
            spark,
            settings=resolved_settings,
            paths=paths,
        )
        raw_rows = df.collect()
        rows = [
            {str(k): _to_json_value(v) for k, v in row.asDict(recursive=True).items()}
            for row in raw_rows
        ]
        return {
            "columns": list(df.columns),
            "rows": rows,
            "row_count": len(rows),
        }

    return _execute


@contextmanager
def open_sqlite_checkpointer(db_path: str | Path = ":memory:") -> Iterator[SqliteSaver]:
    """Open a ``SqliteSaver`` checkpointer as a managed context."""
    target = str(db_path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(target) as saver:
        yield saver


def create_sqlite_checkpointer(db_path: str | Path = ":memory:") -> SqliteSaver:
    """Create a persistent ``SqliteSaver`` suitable for long-lived UI sessions."""
    target = str(db_path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _latest_user_question(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _message_text(msg)
    return _message_text(messages[-1]) if messages else ""


def _parse_router_output(raw_text: str) -> dict[str, Any]:
    """Parse JSON routing decision from an LLM response, tolerating code fences."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Router model returned an empty response.")

    fence_match = _JSON_BLOCK_RE.search(cleaned)
    candidate = fence_match.group(1) if fence_match else cleaned

    if not candidate.startswith("{"):
        first_brace = candidate.find("{")
        last_brace = candidate.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidate = candidate[first_brace : last_brace + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        upper = cleaned.lstrip().upper()
        if upper.startswith("SELECT ") or upper.startswith("WITH "):
            return {
                "user_intent": "ad_hoc_sql",
                "answer_from_manifest": False,
                "sql_query": cleaned,
                "metric_name": None,
                "direct_response": None,
            }
        raise ValueError(f"Router output is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Router JSON response must be an object.")
    return parsed


def _normalize_query_result(raw: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(raw, list):
        rows = [{str(k): _to_json_value(v) for k, v in row.items()} for row in raw]
        columns = list(rows[0].keys()) if rows else []
        return {"columns": columns, "rows": rows, "row_count": len(rows)}

    if isinstance(raw, dict):
        rows_raw = raw.get("rows", [])
        rows = [{str(k): _to_json_value(v) for k, v in row.items()} for row in rows_raw]
        columns = list(raw.get("columns") or (rows[0].keys() if rows else []))
        return {"columns": columns, "rows": rows, "row_count": len(rows)}

    raise TypeError(f"Unexpected SQL executor return type: {type(raw).__name__}")


def _as_finite_float(value: object) -> float | None:
    """Return ``value`` as a float when it is a finite real number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _extract_group_metrics(
    rows: list[dict[str, Any]],
    preferred_metric: str | None = None,
) -> tuple[list[float], list[float], str]:
    """Extract depressed (label=1) and control (label=0) series for the statistical guard."""
    metric_label = preferred_metric or "metric"
    if not rows:
        return [], [], metric_label

    has_label = any("label" in row for row in rows)
    if not has_label:
        return [], [], metric_label

    candidate_cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in candidate_cols:
                candidate_cols.append(key)

    metric_col: str | None = None
    if preferred_metric and any(_as_finite_float(r.get(preferred_metric)) is not None for r in rows):
        metric_col = preferred_metric
    else:
        for col in candidate_cols:
            if col.lower() in _EXCLUDED_METRIC_COLUMNS:
                continue
            if any(_as_finite_float(r.get(col)) is not None for r in rows):
                metric_col = col
                break

    if metric_col is None:
        return [], [], metric_label

    depressed: list[float] = []
    control: list[float] = []
    for row in rows:
        raw_label = row.get("label")
        val = _as_finite_float(row.get(metric_col))
        if raw_label is None or val is None:
            continue
        try:
            label_int = int(raw_label)
        except (TypeError, ValueError):
            continue
        if label_int == 1:
            depressed.append(val)
        elif label_int == 0:
            control.append(val)

    return depressed, control, metric_col


def _normalize_hitl_decision(decision: Any) -> bool:
    """Interpret human resume payloads into approve (True) or reject (False)."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict):
        if "approved" in decision:
            return bool(decision["approved"])
        if "decision" in decision:
            return _normalize_hitl_decision(decision["decision"])
    if isinstance(decision, str):
        token = decision.strip().lower()
        if token in {"approve", "approved", "autorizza", "yes", "true", "1", "ok"}:
            return True
        if token in {"reject", "rejected", "cancel", "cancelled", "annulla", "no", "false", "0"}:
            return False
    return False


def build_agent_graph(
    *,
    settings: Settings | None = None,
    router_model: BaseChatModel | None = None,
    critic_model: BaseChatModel | None = None,
    sql_executor: SqlExecutor | None = None,
    spark: SparkSession | None = None,
    cost_estimator: CostEstimator | None = None,
    manifest_loader: ManifestLoader | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the cyclic LangGraph ``StateGraph`` for lakehouse analytics.

    Dependencies (chat models, Spark/SQL executor, cost estimator, manifest
    loader, SQLite checkpointer) are injected so offline tests never touch
    networked LLMs or start a JVM SparkSession.
    """
    cfg = settings or get_settings()

    def _get_router_model() -> BaseChatModel:
        nonlocal router_model
        if router_model is None:
            router_model = create_chat_model("router", settings=cfg, temperature=0.0)
        return router_model

    def _get_critic_model() -> BaseChatModel:
        nonlocal critic_model
        if critic_model is None:
            critic_model = create_chat_model("critic", settings=cfg, temperature=0.0)
        return critic_model

    def _get_sql_executor() -> SqlExecutor:
        if sql_executor is not None:
            return sql_executor
        if spark is not None:
            return make_spark_sql_executor(spark, settings=cfg)

        def _missing_spark(query: str, _cfg: Settings) -> dict[str, Any]:
            raise SafeSQLError(
                "SPARK_UNAVAILABLE",
                "No SparkSession or SQL executor is configured for query execution.",
                query=query,
            )

        return _missing_spark

    def _estimate_cost(query: str) -> dict[str, Any]:
        if cost_estimator is not None:
            return cost_estimator(query, cfg)
        return estimate_sql_cost(query, settings=cfg, spark=spark)

    def _load_manifest() -> DatasetManifest:
        if manifest_loader is not None:
            return manifest_loader()
        return load_manifest(settings=cfg)

    def intent_router_planner(state: DataAgentState) -> dict[str, Any]:
        question = _latest_user_question(state.get("messages", []))
        last_error = state.get("last_error")
        retry_count = state.get("retry_count", 0)
        planner_prompt = build_planner_user_prompt(
            question,
            last_error=last_error,
            planner_notes=state.get("planner_notes"),
            retry_count=retry_count,
        )

        llm_response = _get_router_model().invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=planner_prompt),
            ]
        )
        raw_text = _message_text(llm_response)

        try:
            plan = _parse_router_output(raw_text)
        except ValueError as exc:
            return {
                "user_intent": "ad_hoc_sql",
                "generated_query": None,
                "last_error": {
                    "code": "PLANNER_PARSE_ERROR",
                    "message": str(exc),
                    "query": None,
                },
                "status": "running",
            }

        intent = str(plan.get("user_intent") or "ad_hoc_sql")
        is_retry = last_error is not None
        wants_manifest = bool(plan.get("answer_from_manifest")) or intent == "catalog_metadata"

        if not is_retry and wants_manifest:
            try:
                manifest = _load_manifest()
                answer = format_manifest_answer(manifest, question)
                status = "completed"
            except FileNotFoundError:
                answer = (
                    "The Gold dataset manifest (`gold_dataset_manifest.json`) is not available yet. "
                    "Run the Medallion ETL pipeline first before querying catalog metadata."
                )
                status = "failed"
            return {
                "user_intent": "catalog_metadata",
                "generated_query": None,
                "last_error": None,
                "final_response": answer,
                "status": status,
                "messages": [AIMessage(content=answer)],
            }

        if not is_retry and intent == "general_conversation" and not plan.get("sql_query"):
            reply = str(
                plan.get("direct_response")
                or "Ask me a question about the eRISK dataset, circadian rhythms, or lexical features."
            )
            return {
                "user_intent": "general_conversation",
                "generated_query": None,
                "last_error": None,
                "final_response": reply,
                "status": "completed",
                "messages": [AIMessage(content=reply)],
            }

        sql_query = plan.get("sql_query")
        if not isinstance(sql_query, str) or not sql_query.strip():
            return {
                "user_intent": intent,
                "generated_query": None,
                "last_error": {
                    "code": "NO_QUERY_GENERATED",
                    "message": "Planner did not produce a SQL SELECT query.",
                    "query": None,
                },
                "status": "running",
            }

        return {
            "user_intent": intent,
            "generated_query": sql_query.strip(),
            "analysis_metric": plan.get("metric_name"),
            "last_error": None,
            "status": "running",
        }

    def ast_validator_and_cost(state: DataAgentState) -> dict[str, Any]:
        raw_query = state.get("generated_query") or ""
        try:
            safe_sql = validate_and_rewrite_sql(raw_query, row_limit=cfg.sql_row_limit)
            cost_info = _estimate_cost(safe_sql)
        except SafeSQLError as exc:
            return {
                "last_error": exc.to_dict(),
                "sql_cost_estimate": None,
                "status": "running",
            }

        return {
            "generated_query": safe_sql,
            "sql_cost_estimate": cost_info,
            "last_error": None,
            "status": "awaiting_approval" if cost_info.get("triggers_hitl") else "running",
        }

    def hitl_gate(state: DataAgentState) -> dict[str, Any]:
        cost_info = state.get("sql_cost_estimate") or {}
        if not cost_info.get("triggers_hitl"):
            return {"hitl_decision": "not_required", "status": "running"}

        decision = interrupt(
            {
                "type": "hitl_approval_required",
                "query": state.get("generated_query"),
                "estimated_cost": cost_info.get("estimated_cost"),
                "threshold": cost_info.get("threshold", cfg.hitl_cost_threshold),
                "reasons": cost_info.get("reasons", []),
                "message": "Estimated query cost exceeds threshold; human authorization required.",
            }
        )
        if _normalize_hitl_decision(decision):
            return {"hitl_decision": "approved", "status": "running"}

        cancel_msg = (
            "Query execution cancelled by user at the human-in-the-loop approval gate."
        )
        return {
            "hitl_decision": "rejected",
            "status": "cancelled",
            "final_response": cancel_msg,
            "messages": [AIMessage(content=cancel_msg)],
        }

    def execute_sql(state: DataAgentState) -> dict[str, Any]:
        query = state.get("generated_query") or ""
        executor = _get_sql_executor()
        try:
            raw_result = executor(query, cfg)
            normalized = _normalize_query_result(raw_result)
        except SafeSQLError as exc:
            return {
                "last_error": exc.to_dict(),
                "query_result": None,
                "status": "running",
            }

        if normalized["row_count"] == 0:
            return {
                "query_result": normalized,
                "last_error": {
                    "code": "EMPTY_RESULT",
                    "message": (
                        "Query executed successfully but returned 0 rows unexpectedly. "
                        "Verify filter predicates (label is 0 or 1) and table selection."
                    ),
                    "query": query,
                },
                "status": "running",
            }

        return {
            "query_result": normalized,
            "last_error": None,
            "status": "running",
        }

    def query_rewriter(state: DataAgentState) -> dict[str, Any]:
        next_retry = state.get("retry_count", 0) + 1
        err = state.get("last_error") or {}
        notes = (
            f"Retry {next_retry}/{cfg.max_query_retries} triggered after error "
            f"[{err.get('code', 'UNKNOWN')}]: {err.get('message', '')}"
        )
        return {
            "retry_count": next_retry,
            "planner_notes": notes,
            "status": "running",
        }

    def statistical_guard(state: DataAgentState) -> dict[str, Any]:
        query_result = state.get("query_result") or {}
        rows = query_result.get("rows", [])
        depressed_vals, control_vals, metric_name = _extract_group_metrics(
            rows,
            preferred_metric=state.get("analysis_metric"),
        )
        guard_result = evaluate_statistical_guard(
            depressed_vals,
            control_vals,
            settings=cfg,
            metric_name=metric_name,
        )
        return {
            "statistical_validation": guard_result.model_dump(mode="json"),
            "analysis_metric": metric_name,
            "status": "running",
        }

    def scientific_critic(state: DataAgentState) -> dict[str, Any]:
        question = _latest_user_question(state.get("messages", []))
        raw_guard = state.get("statistical_validation")
        guard_result = (
            StatisticalGuardResult.model_validate(raw_guard)
            if raw_guard is not None
            else evaluate_statistical_guard([], [], settings=cfg, metric_name="metric")
        )
        system_prompt = build_critic_system_prompt(guard_result)
        user_prompt = build_critic_user_prompt(
            question,
            query=state.get("generated_query"),
            query_result=state.get("query_result"),
            cost_estimate=state.get("sql_cost_estimate"),
            guard_result=guard_result,
        )

        response = _get_critic_model().invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        critique_text = _message_text(response)
        return {
            "critic_system_prompt": system_prompt,
            "critic_critique": critique_text,
            "final_response": critique_text,
            "status": "completed",
            "messages": [AIMessage(content=critique_text)],
        }

    def handle_failure(state: DataAgentState) -> dict[str, Any]:
        err = state.get("last_error") or {}
        attempts = state.get("retry_count", 0) + 1
        code = err.get("code", "EXECUTION_FAILED")
        message = err.get("message", "Unknown query execution error.")
        failed_query = err.get("query") or state.get("generated_query") or "N/A"
        explanation = (
            f"Unable to complete query after {attempts} attempt(s) "
            f"(max retries: {cfg.max_query_retries}).\n"
            f"- Error code: `{code}`\n"
            f"- Details: {message}\n"
            f"- Last query: `{failed_query}`"
        )
        return {
            "status": "failed",
            "final_response": explanation,
            "messages": [AIMessage(content=explanation)],
        }

    def _route_after_planner(state: DataAgentState) -> str:
        if state.get("status") in {"completed", "failed"} and state.get("final_response"):
            return END
        if state.get("last_error") is not None:
            if state.get("retry_count", 0) < cfg.max_query_retries:
                return "query_rewriter"
            return "handle_failure"
        return "ast_validator_and_cost"

    def _route_after_validator(state: DataAgentState) -> str:
        if state.get("last_error") is not None:
            if state.get("retry_count", 0) < cfg.max_query_retries:
                return "query_rewriter"
            return "handle_failure"
        cost_info = state.get("sql_cost_estimate") or {}
        if cost_info.get("triggers_hitl"):
            return "hitl_gate"
        return "execute_sql"

    def _route_after_hitl(state: DataAgentState) -> str:
        if state.get("hitl_decision") == "rejected" or state.get("status") == "cancelled":
            return END
        return "execute_sql"

    def _route_after_execute(state: DataAgentState) -> str:
        if state.get("last_error") is not None:
            if state.get("retry_count", 0) < cfg.max_query_retries:
                return "query_rewriter"
            return "handle_failure"
        return "statistical_guard"

    builder = StateGraph(DataAgentState)
    builder.add_node("intent_router_planner", intent_router_planner)
    builder.add_node("ast_validator_and_cost", ast_validator_and_cost)
    builder.add_node("hitl_gate", hitl_gate)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node("query_rewriter", query_rewriter)
    builder.add_node("statistical_guard", statistical_guard)
    builder.add_node("scientific_critic", scientific_critic)
    builder.add_node("handle_failure", handle_failure)

    builder.add_edge(START, "intent_router_planner")
    builder.add_conditional_edges(
        "intent_router_planner",
        _route_after_planner,
        {
            "ast_validator_and_cost": "ast_validator_and_cost",
            "query_rewriter": "query_rewriter",
            "handle_failure": "handle_failure",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "ast_validator_and_cost",
        _route_after_validator,
        {
            "hitl_gate": "hitl_gate",
            "execute_sql": "execute_sql",
            "query_rewriter": "query_rewriter",
            "handle_failure": "handle_failure",
        },
    )
    builder.add_conditional_edges(
        "hitl_gate",
        _route_after_hitl,
        {
            "execute_sql": "execute_sql",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "execute_sql",
        _route_after_execute,
        {
            "statistical_guard": "statistical_guard",
            "query_rewriter": "query_rewriter",
            "handle_failure": "handle_failure",
        },
    )
    builder.add_edge("query_rewriter", "intent_router_planner")
    builder.add_edge("statistical_guard", "scientific_critic")
    builder.add_edge("scientific_critic", END)
    builder.add_edge("handle_failure", END)

    return builder.compile(checkpointer=checkpointer)
