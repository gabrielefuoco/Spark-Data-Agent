"""LangGraph node implementations, Spark executors, and runtime dependency wiring."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.agent.cost import estimate_job_cost, estimate_sql_cost
from src.agent.critic_check import (
    format_critic_correction_prompt,
    format_inconclusive_critic_fallback,
    validate_critic_output,
)
from src.agent.guards import (
    StatisticalGuardResult,
    evaluate_query_result_guard,
)
from src.agent.memory import build_conversation_window
from src.agent.models import create_chat_model
from src.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    build_critic_system_prompt,
    build_critic_user_prompt,
    extract_message_text,
    format_statistical_appendix,
    format_tool_results_summary,
)
from src.agent.state import DataAgentState, ToolResultEntry, ToolTraceEntry
from src.agent.tools import AGENT_TOOLS, canonical_tool_signature
from src.analytics.features import (
    DEFAULT_SESSION_GAP_SECONDS,
    DEFAULT_WINDOW_DAYS,
    summarize_user_longitudinal_features,
)
from src.analytics.nlp_pipeline import fit_topic_model
from src.analytics.safe_sql import SafeSQLError, execute_safe_spark_sql, validate_and_rewrite_sql
from src.config import Settings
from src.lakehouse.manifest import DatasetManifest, load_manifest
from src.lakehouse.paths import LakehousePaths

SqlExecutor = Callable[[str, Settings], dict[str, Any] | list[dict[str, Any]]]
JobExecutor = Callable[[str, dict[str, Any], Settings], dict[str, Any] | list[dict[str, Any]]]
ManifestLoader = Callable[[], DatasetManifest]

SUPPORTED_SPARK_JOBS: frozenset[str] = frozenset({"circadian_features", "topic_model"})
_SPARK_JOB_PREFIX = "SPARK_JOB:"
_CONSECUTIVE_REPEAT_CAP = 2
_TOOL_PREVIEW_ROW_LIMIT = 20


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


def _collect_spark_result(df: DataFrame, fetch_limit: int) -> dict[str, Any]:
    rows = [
        {str(k): _to_json_value(v) for k, v in row.asDict(recursive=True).items()}
        for row in df.collect()
    ]
    rows_returned = len(rows)
    return {
        "columns": list(df.columns),
        "rows": rows,
        "row_count": rows_returned,
        "rows_returned": rows_returned,
        "truncated": rows_returned >= fetch_limit,
    }


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
        return _collect_spark_result(df, resolved_settings.sql_fetch_limit)

    return _execute


def encode_spark_job(job_name: str, params: dict[str, Any] | None = None) -> str:
    """Serialize a non-SQL Spark job dispatch specification into a readable string."""
    payload = {"job": job_name.strip().lower(), "params": dict(params or {})}
    return f"{_SPARK_JOB_PREFIX}{json.dumps(payload, sort_keys=True)}"


def decode_spark_job(generated_query: str | None) -> tuple[str, dict[str, Any]] | None:
    """Parse a serialized Spark job dispatch spec, or return ``None`` for SQL."""
    if not isinstance(generated_query, str):
        return None
    stripped = generated_query.strip()
    if not stripped.startswith(_SPARK_JOB_PREFIX):
        return None
    body = stripped[len(_SPARK_JOB_PREFIX) :].strip()
    if not body.startswith("{"):
        return None
    parsed = json.loads(body)
    if not isinstance(parsed, dict) or not parsed.get("job"):
        return None
    raw_params = parsed.get("params")
    params = dict(raw_params) if isinstance(raw_params, dict) else {}
    return str(parsed["job"]).strip().lower(), params


def make_spark_job_executor(
    spark: SparkSession,
    *,
    settings: Settings | None = None,
    paths: LakehousePaths | None = None,
) -> JobExecutor:
    """Build a runtime Spark executor for non-SQL analytics jobs (circadian & LDA)."""

    def _execute_job(job_name: str, params: dict[str, Any], cfg: Settings) -> dict[str, Any]:
        resolved_settings = settings or cfg
        resolved_paths = paths or LakehousePaths.from_settings(resolved_settings)
        normalized = job_name.strip().lower()
        if normalized not in SUPPORTED_SPARK_JOBS:
            raise SafeSQLError(
                "UNKNOWN_SPARK_JOB",
                f"Unsupported Spark job {job_name!r}; expected one of {sorted(SUPPORTED_SPARK_JOBS)}.",
                query=encode_spark_job(job_name, params),
            )

        silver_df = spark.read.parquet(str(resolved_paths.silver_posts))
        if normalized == "circadian_features":
            result_df = summarize_user_longitudinal_features(
                silver_df,
                window_days=int(params.get("window_days", DEFAULT_WINDOW_DAYS)),
                session_gap_seconds=int(
                    params.get("session_gap_seconds", DEFAULT_SESSION_GAP_SECONDS)
                ),
            ).limit(resolved_settings.sql_fetch_limit)
        else:
            combined = silver_df.select(
                "label",
                F.concat_ws(" ", F.coalesce(F.col("title"), F.lit("")), F.col("text")).alias(
                    "combined_text"
                ),
            )
            topic_result = fit_topic_model(
                combined,
                input_col="combined_text",
                k=int(params.get("k", 8)),
                max_iter=int(params.get("max_iter", 20)),
                min_df=float(params.get("min_df", 2.0)),
                max_df=float(params.get("max_df", 0.95)),
                vocab_size=int(params.get("vocab_size", 5000)),
                top_n_terms=int(params.get("top_n_terms", 10)),
                seed=int(params.get("seed", 42)),
            )
            if topic_result.cohort_lexical_matrix is None:
                raise SafeSQLError(
                    "SPARK_JOB_ERROR",
                    "Topic model job requires 'label' column on silver_posts.",
                    query=encode_spark_job(job_name, params),
                )
            result_df = topic_result.cohort_lexical_matrix.limit(
                resolved_settings.sql_fetch_limit
            )

        return _collect_spark_result(result_df, resolved_settings.sql_fetch_limit)

    return _execute_job


@dataclass
class AgentRuntimeDeps:
    """Injected or lazily resolved runtime dependencies shared by agent nodes."""

    settings: Settings
    router_model: BaseChatModel | None = None
    critic_model: BaseChatModel | None = None
    sql_executor: SqlExecutor | None = None
    job_executor: JobExecutor | None = None
    spark: SparkSession | None = None
    manifest_loader: ManifestLoader | None = None

    def get_router_model(self) -> BaseChatModel:
        """Lazily build and cache the tool-calling chat model on first use."""
        if self.router_model is None:
            self.router_model = create_chat_model("router", settings=self.settings, temperature=0.0)
        return self.router_model

    def get_critic_model(self) -> BaseChatModel:
        """Lazily build and cache the critic chat model on first use."""
        if self.critic_model is None:
            self.critic_model = create_chat_model("critic", settings=self.settings, temperature=0.0)
        return self.critic_model

    def get_sql_executor(self) -> SqlExecutor:
        """Return the configured SQL executor or one backed by ``spark``."""
        if self.sql_executor is not None:
            return self.sql_executor
        if self.spark is not None:
            return make_spark_sql_executor(self.spark, settings=self.settings)

        def _missing_spark(query: str, _cfg: Settings) -> dict[str, Any]:
            raise SafeSQLError(
                "SPARK_UNAVAILABLE",
                "No SparkSession or SQL executor is configured for query execution.",
                query=query,
            )

        return _missing_spark

    def get_job_executor(self) -> JobExecutor:
        """Return the configured Spark job executor or one backed by ``spark``."""
        if self.job_executor is not None:
            return self.job_executor
        if self.spark is not None:
            return make_spark_job_executor(self.spark, settings=self.settings)

        def _missing_spark_job(
            job_name: str, _params: dict[str, Any], _cfg: Settings
        ) -> dict[str, Any]:
            raise SafeSQLError(
                "SPARK_UNAVAILABLE",
                "No SparkSession or job executor is configured for Spark job execution.",
                query=job_name,
            )

        return _missing_spark_job

    def estimate_cost(self, query: str) -> dict[str, Any]:
        """Estimate query or Spark job execution cost."""
        job_spec = decode_spark_job(query)
        if job_spec is not None:
            job_name, job_params = job_spec
            return estimate_job_cost(job_name, job_params, settings=self.settings)
        return estimate_sql_cost(query, settings=self.settings, spark=self.spark)

    def load_manifest(self) -> DatasetManifest:
        """Load the Gold dataset manifest using the injected loader or settings."""
        if self.manifest_loader is not None:
            return self.manifest_loader()
        return load_manifest(settings=self.settings)


def _filter_prior_conversation(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Keep only user turns and final assistant replies for cross-turn memory."""
    filtered: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage) or (isinstance(msg, AIMessage) and not msg.tool_calls):
            filtered.append(msg)
    return filtered


def _split_turn_messages(
    messages: Sequence[BaseMessage],
) -> tuple[str, list[BaseMessage], list[BaseMessage]]:
    """Return ``(latest_question, prior_conversation, current_turn_messages)``."""
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return (
                extract_message_text(messages[idx]),
                _filter_prior_conversation(messages[:idx]),
                list(messages[idx + 1 :]),
            )
    return (extract_message_text(messages[-1]) if messages else ""), [], []


def _normalize_query_result(raw: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(raw, list):
        rows = [{str(k): _to_json_value(v) for k, v in row.items()} for row in raw]
        columns = list(rows[0].keys()) if rows else []
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "rows_returned": len(rows),
            "truncated": False,
        }

    if isinstance(raw, dict):
        rows_raw = raw.get("rows", [])
        rows = [{str(k): _to_json_value(v) for k, v in row.items()} for row in rows_raw]
        columns = list(raw.get("columns") or (rows[0].keys() if rows else []))
        rows_returned = int(raw.get("rows_returned", len(rows)))
        return {
            "columns": columns,
            "rows": rows,
            "row_count": rows_returned,
            "rows_returned": rows_returned,
            "truncated": bool(raw.get("truncated", False)),
        }

    raise TypeError(f"Unexpected SQL executor return type: {type(raw).__name__}")


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


def _empty_guard_payload(settings: Settings, metric_name: str = "descriptive_metric") -> dict[str, Any]:
    guard = evaluate_query_result_guard([], preferred_metric=metric_name, settings=settings)
    payload = guard.model_dump(mode="json")
    payload["rows_returned"] = 0
    payload["rows_shown"] = 0
    payload["truncated"] = False
    return payload


def _build_guard_and_display(
    raw: dict[str, Any] | list[dict[str, Any]],
    *,
    settings: Settings,
    preferred_metric: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = _normalize_query_result(raw)
    all_rows = normalized["rows"]
    columns = normalized["columns"]
    rows_returned = int(normalized.get("rows_returned", len(all_rows)))
    shown_rows = all_rows[: settings.sql_row_limit]
    rows_shown = len(shown_rows)
    truncated = bool(normalized.get("truncated")) or (rows_returned >= settings.sql_fetch_limit)

    display_payload = {
        "columns": columns,
        "rows": shown_rows,
        "row_count": rows_returned,
        "rows_returned": rows_returned,
        "rows_shown": rows_shown,
        "truncated": truncated,
    }
    guard_result = evaluate_query_result_guard(
        all_rows,
        columns=columns,
        preferred_metric=preferred_metric,
        settings=settings,
    )
    guard_payload = guard_result.model_dump(mode="json")
    guard_payload["rows_returned"] = rows_returned
    guard_payload["rows_shown"] = rows_shown
    guard_payload["truncated"] = truncated
    return display_payload, guard_payload


def _select_ui_summary_entry(tool_results: Sequence[ToolResultEntry]) -> ToolResultEntry | None:
    """Pick the latest assessed comparison for the UI card, falling back to the latest result."""
    for idx in range(len(tool_results) - 1, -1, -1):
        entry = tool_results[idx]
        if entry["guard_payload"].get("assessment_status") == "assessed":
            return entry
    return tool_results[-1] if tool_results else None


def _format_tool_success_message(
    name: str,
    *,
    query: str | None,
    display_payload: dict[str, Any],
    guard_payload: dict[str, Any],
) -> str:
    rows = display_payload["rows"]
    payload: dict[str, Any] = {
        "status": "executed",
        "tool": name,
        "query": query,
        "columns": display_payload["columns"],
        "rows_returned": display_payload["rows_returned"],
        "rows_shown": display_payload["rows_shown"],
        "truncated": display_payload["truncated"],
        "sample_rows": rows[:_TOOL_PREVIEW_ROW_LIMIT],
        "statistical_validation": guard_payload,
    }
    if len(rows) > _TOOL_PREVIEW_ROW_LIMIT:
        payload["preview_note"] = (
            f"Showing first {_TOOL_PREVIEW_ROW_LIMIT} of {display_payload['rows_returned']} rows; "
            "all fetched rows were evaluated by the statistical guard."
        )
    return json.dumps(payload, ensure_ascii=False, default=str)


def _format_tool_error_message(
    *,
    status: str,
    error: dict[str, Any],
) -> str:
    return json.dumps({"status": status, "error": error}, ensure_ascii=False, default=str)


def agent(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Invoke the tool-bound model with system prompt, history window, and tool summary."""
    question, prior_messages, current_turn_messages = _split_turn_messages(state.get("messages", []))
    history_window = (
        build_conversation_window(
            prior_messages,
            token_budget=deps.settings.conversation_token_budget,
        )
        if prior_messages
        else None
    )

    tool_trace = state.get("tool_trace", [])
    tool_results = state.get("tool_results", [])
    tool_call_count = state.get("tool_call_count", 0)
    budget_exhausted = bool(state.get("budget_exhausted", False))

    system_parts = [AGENT_SYSTEM_PROMPT]
    if tool_trace or budget_exhausted:
        system_parts.append(
            format_tool_results_summary(
                tool_trace,
                tool_results,
                tool_call_count=tool_call_count,
                max_tool_calls=deps.settings.max_tool_calls,
                budget_exhausted=budget_exhausted,
            )
        )
    system_prompt = "\n\n".join(system_parts)

    prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if history_window is not None:
        if history_window.dropped_messages > 0:
            noun = "message" if history_window.dropped_messages == 1 else "messages"
            question_text = (
                f"[{history_window.dropped_messages} earlier {noun} omitted due to token budget]\n\n"
                f"{question}"
            )
        else:
            question_text = question
        prompt_messages.extend(history_window.messages)
    else:
        question_text = question

    prompt_messages.append(HumanMessage(content=question_text))
    prompt_messages.extend(current_turn_messages)

    base_model = deps.get_router_model()
    invokable = base_model if budget_exhausted else base_model.bind_tools(list(AGENT_TOOLS))
    response = invokable.invoke(prompt_messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        if budget_exhausted:
            return {"status": "failing"}
        return {"messages": [response], "status": "running"}

    text = extract_message_text(response).strip()
    if budget_exhausted and not text:
        return {"status": "failing"}

    if tool_results:
        return {"agent_draft_response": text, "status": "running"}

    if budget_exhausted and (
        "budget" not in text.lower() and str(deps.settings.max_tool_calls) not in text
    ):
        text = (
            f"{text}\n\n"
            f"(Tool call budget exhausted: max_tool_calls={deps.settings.max_tool_calls}.)"
        )

    final_status = "cancelled" if state.get("hitl_decision") == "rejected" else "completed"
    return {
        "final_response": text,
        "status": final_status,
        "messages": [AIMessage(content=text)],
    }


def _find_pending_tool_calls(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            answered_ids = {
                m.tool_call_id
                for m in messages[idx + 1 :]
                if isinstance(m, ToolMessage) and m.tool_call_id
            }
            return [dict(tc) for tc in msg.tool_calls if tc.get("id") not in answered_ids]
    return []


def _execute_sql_tool(
    args: dict[str, Any],
    deps: AgentRuntimeDeps,
) -> tuple[str, str | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, str, dict[str, Any]]:
    """Validate, cost-gate (HITL), execute, and guard a ``run_safe_sql`` call."""
    raw_query = args.get("query")
    if not isinstance(raw_query, str):
        err = SafeSQLError("EMPTY_QUERY", "Argument 'query' must be a SQL string.").to_dict()
        return "error", None, None, None, err, "not_required", _empty_guard_payload(deps.settings)

    try:
        safe_sql = validate_and_rewrite_sql(raw_query, row_limit=deps.settings.sql_fetch_limit)
        cost_info = deps.estimate_cost(safe_sql)
    except SafeSQLError as exc:
        err = exc.to_dict()
        return (
            "error",
            raw_query,
            None,
            None,
            err,
            "not_required",
            _empty_guard_payload(deps.settings),
        )

    hitl_decision = "not_required"
    if cost_info.get("triggers_hitl"):
        decision = interrupt(
            {
                "type": "hitl_approval_required",
                "query": safe_sql,
                "estimated_cost": cost_info.get("estimated_cost"),
                "threshold": cost_info.get("threshold", deps.settings.hitl_cost_threshold),
                "reasons": cost_info.get("reasons", []),
                "message": "Estimated query cost exceeds threshold; human authorization required.",
            }
        )
        if not _normalize_hitl_decision(decision):
            err = {
                "code": "HITL_REJECTED",
                "message": "Query execution cancelled/rejected by user at the human-in-the-loop approval gate.",
                "query": safe_sql,
            }
            return (
                "hitl_rejected",
                safe_sql,
                cost_info,
                None,
                err,
                "rejected",
                _empty_guard_payload(deps.settings),
            )
        hitl_decision = "approved"

    try:
        raw_result = deps.get_sql_executor()(safe_sql, deps.settings)
    except SafeSQLError as exc:
        err = exc.to_dict()
        return (
            "error",
            safe_sql,
            cost_info,
            None,
            err,
            hitl_decision,
            _empty_guard_payload(deps.settings),
        )

    display_payload, guard_payload = _build_guard_and_display(raw_result, settings=deps.settings)
    return (
        "executed",
        safe_sql,
        cost_info,
        display_payload,
        None,
        hitl_decision,
        guard_payload,
    )


def _execute_manifest_tool(
    deps: AgentRuntimeDeps,
) -> tuple[str, str | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Load the static Gold manifest without Spark and tag it strictly ``descriptive``."""
    try:
        manifest = deps.load_manifest()
    except FileNotFoundError:
        err = {
            "code": "MANIFEST_NOT_FOUND",
            "message": (
                "The Gold dataset manifest (gold_dataset_manifest.json) is not available yet. "
                "Run the Medallion ETL pipeline first."
            ),
            "query": "read_dataset_manifest()",
        }
        return (
            "error",
            "read_dataset_manifest()",
            None,
            None,
            err,
            _empty_guard_payload(deps.settings, "catalog_manifest"),
        )

    manifest_row = manifest.model_dump(mode="json")
    display_payload, guard_payload = _build_guard_and_display(
        [manifest_row],
        settings=deps.settings,
        preferred_metric="total_posts",
    )
    return (
        "executed",
        "read_dataset_manifest()",
        None,
        display_payload,
        None,
        guard_payload,
    )


def _execute_spark_job_tool(
    args: dict[str, Any],
    deps: AgentRuntimeDeps,
) -> tuple[str, str | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, str, dict[str, Any]]:
    """Validate, cost-gate (HITL), execute, and guard a ``run_spark_job`` call."""
    raw_job = args.get("job")
    raw_params = args.get("params")
    if not isinstance(raw_job, str) or not raw_job.strip():
        err = SafeSQLError("INVALID_JOB_SPEC", "Argument 'job' must be a non-empty string.").to_dict()
        return "error", None, None, None, err, "not_required", _empty_guard_payload(deps.settings)

    if raw_params is not None and not isinstance(raw_params, dict):
        err = SafeSQLError(
            "INVALID_JOB_PARAMS",
            "Spark job parameters must be a JSON object or null.",
            query=raw_job,
        ).to_dict()
        return "error", raw_job, None, None, err, "not_required", _empty_guard_payload(deps.settings)

    job_name = raw_job.strip().lower()
    clean_params = dict(raw_params or {})
    canonical = encode_spark_job(job_name, clean_params)

    if job_name not in SUPPORTED_SPARK_JOBS:
        err = SafeSQLError(
            "UNKNOWN_SPARK_JOB",
            f"Unsupported Spark job {job_name!r}; expected one of {sorted(SUPPORTED_SPARK_JOBS)}.",
            query=canonical,
        ).to_dict()
        return (
            "error",
            canonical,
            None,
            None,
            err,
            "not_required",
            _empty_guard_payload(deps.settings),
        )

    cost_info = estimate_job_cost(job_name, clean_params, settings=deps.settings)
    hitl_decision = "not_required"
    if cost_info.get("triggers_hitl"):
        decision = interrupt(
            {
                "type": "hitl_approval_required",
                "query": canonical,
                "estimated_cost": cost_info.get("estimated_cost"),
                "threshold": cost_info.get("threshold", deps.settings.hitl_cost_threshold),
                "reasons": cost_info.get("reasons", []),
                "message": "Estimated Spark job cost exceeds threshold; human authorization required.",
            }
        )
        if not _normalize_hitl_decision(decision):
            err = {
                "code": "HITL_REJECTED",
                "message": "Spark job execution cancelled/rejected by user at the human-in-the-loop approval gate.",
                "query": canonical,
            }
            return (
                "hitl_rejected",
                canonical,
                cost_info,
                None,
                err,
                "rejected",
                _empty_guard_payload(deps.settings),
            )
        hitl_decision = "approved"

    try:
        raw_result = deps.get_job_executor()(job_name, clean_params, deps.settings)
    except SafeSQLError as exc:
        err = exc.to_dict()
        return (
            "error",
            canonical,
            cost_info,
            None,
            err,
            hitl_decision,
            _empty_guard_payload(deps.settings),
        )

    display_payload, guard_payload = _build_guard_and_display(raw_result, settings=deps.settings)
    return (
        "executed",
        canonical,
        cost_info,
        display_payload,
        None,
        hitl_decision,
        guard_payload,
    )


def execute_tools(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Execute one pending tool call per step with budget, repeat-cap, HITL, and guard."""
    pending = _find_pending_tool_calls(state.get("messages", []))
    if not pending:
        return {"status": "running"}

    trace = list(state.get("tool_trace", []))
    results = list(state.get("tool_results", []))
    tool_call_count = state.get("tool_call_count", 0)
    last_sig = state.get("last_tool_signature")
    repeat_count = state.get("consecutive_repeat_count", 0)
    repeat_triggered = bool(state.get("repeat_cap_triggered", False))
    hitl_decision = state.get("hitl_decision")

    if tool_call_count >= deps.settings.max_tool_calls:
        new_messages: list[ToolMessage] = []
        budget_err = {
            "code": "TOOL_BUDGET_EXHAUSTED",
            "message": (
                f"Tool call budget exhausted ({deps.settings.max_tool_calls}/"
                f"{deps.settings.max_tool_calls}). Stop calling tools and answer with "
                "the results gathered so far."
            ),
        }
        for tc in pending:
            call_id = str(tc.get("id") or f"call_{len(trace) + 1}")
            name = str(tc.get("name") or "")
            args = dict(tc.get("args") or {})
            guard_payload = _empty_guard_payload(deps.settings)
            trace.append(
                ToolTraceEntry(
                    tool_call_id=call_id,
                    name=name,
                    arguments=args,
                    rows_returned=0,
                    guard_payload=guard_payload,
                    status="budget_exhausted",
                    cost_estimate=None,
                    error=budget_err,
                )
            )
            new_messages.append(
                ToolMessage(
                    content=_format_tool_error_message(
                        status="budget_exhausted",
                        error=budget_err,
                    ),
                    tool_call_id=call_id,
                    name=name,
                )
            )
        return {
            "messages": new_messages,
            "tool_trace": trace,
            "budget_exhausted": True,
            "status": "running",
        }

    tc = pending[0]
    call_id = str(tc.get("id") or f"call_{len(trace) + 1}")
    name = str(tc.get("name") or "")
    args = dict(tc.get("args") or {})
    sig = canonical_tool_signature(name, args)

    if sig == last_sig and repeat_count >= _CONSECUTIVE_REPEAT_CAP:
        repeat_err = {
            "code": "CONSECUTIVE_REPEAT_CAP",
            "message": (
                f"Identical tool call {name!r} with the same arguments was already invoked "
                f"{_CONSECUTIVE_REPEAT_CAP} times consecutively. Change the query/arguments "
                "or provide your final answer from the data already gathered."
            ),
        }
        guard_payload = _empty_guard_payload(deps.settings)
        if repeat_triggered:
            halted_messages: list[ToolMessage] = []
            for item in pending:
                item_id = str(item.get("id") or f"call_{len(trace) + 1}")
                item_name = str(item.get("name") or "")
                item_args = dict(item.get("args") or {})
                trace.append(
                    ToolTraceEntry(
                        tool_call_id=item_id,
                        name=item_name,
                        arguments=item_args,
                        rows_returned=0,
                        guard_payload=_empty_guard_payload(deps.settings),
                        status="repeat_blocked",
                        cost_estimate=None,
                        error=repeat_err,
                    )
                )
                halted_messages.append(
                    ToolMessage(
                        content=_format_tool_error_message(
                            status="repeat_blocked",
                            error=repeat_err,
                        ),
                        tool_call_id=item_id,
                        name=item_name,
                    )
                )
            return {
                "messages": halted_messages,
                "tool_trace": trace,
                "repeat_cap_triggered": True,
                "repeat_loop_halted": True,
                "status": "running",
            }

        trace.append(
            ToolTraceEntry(
                tool_call_id=call_id,
                name=name,
                arguments=args,
                rows_returned=0,
                guard_payload=guard_payload,
                status="repeat_blocked",
                cost_estimate=None,
                error=repeat_err,
            )
        )
        return {
            "messages": [
                ToolMessage(
                    content=_format_tool_error_message(
                        status="repeat_blocked",
                        error=repeat_err,
                    ),
                    tool_call_id=call_id,
                    name=name,
                )
            ],
            "tool_trace": trace,
            "repeat_cap_triggered": True,
            "status": "running",
        }

    next_count = tool_call_count + 1
    next_repeat = (repeat_count + 1) if sig == last_sig else 1

    if name == "run_safe_sql":
        status, query_label, cost_info, display_payload, err, call_hitl, guard_payload = (
            _execute_sql_tool(args, deps)
        )
        if call_hitl != "not_required":
            hitl_decision = call_hitl
    elif name == "read_dataset_manifest":
        status, query_label, cost_info, display_payload, err, guard_payload = (
            _execute_manifest_tool(deps)
        )
    elif name == "run_spark_job":
        status, query_label, cost_info, display_payload, err, call_hitl, guard_payload = (
            _execute_spark_job_tool(args, deps)
        )
        if call_hitl != "not_required":
            hitl_decision = call_hitl
    else:
        status = "error"
        query_label = None
        cost_info = None
        display_payload = None
        err = {
            "code": "UNKNOWN_TOOL",
            "message": f"Unsupported tool {name!r}; expected run_safe_sql, read_dataset_manifest, or run_spark_job.",
        }
        guard_payload = _empty_guard_payload(deps.settings)

    rows_returned = int(display_payload["rows_returned"]) if display_payload is not None else 0
    trace.append(
        ToolTraceEntry(
            tool_call_id=call_id,
            name=name,
            arguments=args,
            rows_returned=rows_returned,
            guard_payload=guard_payload,
            status=status,
            cost_estimate=cost_info,
            error=err,
        )
    )

    if status == "executed" and display_payload is not None:
        results.append(
            ToolResultEntry(
                tool_call_id=call_id,
                name=name,
                arguments=args,
                query=query_label,
                cost_estimate=cost_info,
                result=display_payload,
                guard_payload=guard_payload,
            )
        )
        tool_msg_content = _format_tool_success_message(
            name,
            query=query_label,
            display_payload=display_payload,
            guard_payload=guard_payload,
        )
    else:
        assert err is not None
        tool_msg_content = _format_tool_error_message(status=status, error=err)

    primary_entry = _select_ui_summary_entry(results)
    updates: dict[str, Any] = {
        "messages": [ToolMessage(content=tool_msg_content, tool_call_id=call_id, name=name)],
        "tool_trace": trace,
        "tool_results": results,
        "tool_call_count": next_count,
        "last_tool_signature": sig,
        "consecutive_repeat_count": next_repeat,
        "repeat_cap_triggered": False,
        "hitl_decision": hitl_decision,
        "status": "running",
    }
    if primary_entry is not None:
        updates["generated_query"] = primary_entry["query"]
        updates["sql_cost_estimate"] = primary_entry["cost_estimate"]
        updates["query_result"] = primary_entry["result"]
        updates["statistical_validation"] = primary_entry["guard_payload"]
    return updates


def _append_statistical_appendix(prose: str, appendix: str) -> str:
    if not appendix:
        return prose
    trimmed = prose.rstrip()
    return f"{trimmed}\n\n{appendix}" if trimmed else appendix


def scientific_critic(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Synthesize the final narrative, append a deterministic statistical appendix, and validate both.

    Because the appendix supplies every guard-computed number deterministically when at least
    one ``assessed`` result exists, numbers in the final answer never depend on the model's
    recollection and the fallback fires only on genuine claim violations (for example, prose
    asserting a difference where the appendix table reports ``non significativo``).
    """
    question, prior_messages, _ = _split_turn_messages(state.get("messages", []))
    history_window = (
        build_conversation_window(
            prior_messages,
            token_budget=deps.settings.conversation_token_budget,
        )
        if prior_messages
        else None
    )

    tool_results = state.get("tool_results", [])
    guard_results: list[StatisticalGuardResult] = [
        StatisticalGuardResult.model_validate(entry["guard_payload"]) for entry in tool_results
    ]
    if not guard_results:
        guard_results = [
            evaluate_query_result_guard([], settings=deps.settings, preferred_metric="metric")
        ]

    truncated_entries = [
        entry for entry in tool_results if bool((entry.get("result") or {}).get("truncated", False))
    ]
    any_truncated = bool(truncated_entries)
    raw_truncated_rows = (
        truncated_entries[0]["result"].get("rows_returned") if truncated_entries else None
    )
    truncated_rows = int(raw_truncated_rows) if raw_truncated_rows is not None else None

    budget_exhausted = bool(state.get("budget_exhausted", False))
    system_prompt = build_critic_system_prompt(
        guard_results,
        truncated=any_truncated,
        rows_returned=truncated_rows,
        sql_fetch_limit=deps.settings.sql_fetch_limit,
        budget_exhausted=budget_exhausted,
        max_tool_calls=deps.settings.max_tool_calls,
    )
    user_prompt = build_critic_user_prompt(
        question,
        tool_results=tool_results,
        history=history_window,
        agent_draft=state.get("agent_draft_response"),
        budget_exhausted=budget_exhausted,
        max_tool_calls=deps.settings.max_tool_calls,
    )

    critic_model = deps.get_critic_model()
    base_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    appendix = format_statistical_appendix(guard_results)
    response = critic_model.invoke(base_messages)
    prose_text = extract_message_text(response)
    critique_text = _append_statistical_appendix(prose_text, appendix)
    critic_calls = 1
    critic_retries = 0
    check = validate_critic_output(critique_text, guard_results)
    check_failed: dict[str, Any] | None = None

    if not check.passed:
        correction_prompt = format_critic_correction_prompt(guard_results, check)
        retry_response = critic_model.invoke(
            [
                *base_messages,
                AIMessage(content=prose_text),
                HumanMessage(content=correction_prompt),
            ]
        )
        critic_calls += 1
        critic_retries = 1
        retry_prose = extract_message_text(retry_response)
        critique_text = _append_statistical_appendix(retry_prose, appendix)
        check = validate_critic_output(critique_text, guard_results)
        if not check.passed:
            critique_text = format_inconclusive_critic_fallback(guard_results, check)
            check_failed = {
                "failed": True,
                "retry_count": critic_retries,
                "critic_call_count": critic_calls,
                "missing_items": list(check.missing_items),
                "forbidden_claims": list(check.forbidden_claims),
            }

    if budget_exhausted and (
        "budget" not in critique_text.lower()
        and str(deps.settings.max_tool_calls) not in critique_text
    ):
        critique_text = (
            f"{critique_text}\n\n"
            f"(Nota: raggiunto il budget massimo di chiamate agli strumenti: "
            f"max_tool_calls={deps.settings.max_tool_calls}.)"
        )

    executed_query = state.get("generated_query")
    ai_kwargs = {"generated_query": executed_query} if executed_query else {}
    return {
        "critic_system_prompt": system_prompt,
        "critic_critique": critique_text,
        "critic_call_count": critic_calls,
        "critic_retry_count": critic_retries,
        "critic_output_check_failed": check_failed,
        "final_response": critique_text,
        "status": "completed",
        "messages": [AIMessage(content=critique_text, additional_kwargs=ai_kwargs)],
    }


def handle_failure(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Format a transparent terminal failure when budget or repeat limits cannot be resolved."""
    tool_count = state.get("tool_call_count", 0)
    if state.get("budget_exhausted"):
        explanation = (
            f"Unable to complete analysis: tool call budget exhausted "
            f"(max_tool_calls={deps.settings.max_tool_calls}, executed={tool_count}) "
            "and the agent could not produce a final answer without calling additional tools."
        )
    elif state.get("repeat_loop_halted"):
        explanation = (
            f"Unable to complete analysis: the agent repeated the same tool call beyond "
            f"the consecutive repeat cap ({_CONSECUTIVE_REPEAT_CAP}) after {tool_count} tool call(s)."
        )
    else:
        explanation = (
            f"Unable to complete analysis after {tool_count} tool call(s)."
        )
    return {
        "status": "failed",
        "final_response": explanation,
        "messages": [AIMessage(content=explanation)],
    }


class NodeFn(Protocol):
    """Callable node signature compatible with LangGraph's ``_Node`` protocol."""

    def __call__(self, state: DataAgentState) -> dict[str, Any]: ...


def _bind_node(
    fn: Callable[[DataAgentState, AgentRuntimeDeps], dict[str, Any]],
    deps: AgentRuntimeDeps,
) -> NodeFn:
    def _node(state: DataAgentState) -> dict[str, Any]:
        return fn(state, deps)

    _node.__name__ = fn.__name__
    return _node


def build_agent_nodes(deps: AgentRuntimeDeps) -> dict[str, NodeFn]:
    """Bind runtime dependencies to each LangGraph node callable."""
    node_fns = (
        agent,
        execute_tools,
        scientific_critic,
        handle_failure,
    )
    return {fn.__name__: _bind_node(fn, deps) for fn in node_fns}
