"""LangGraph node implementations, Spark executors, and runtime dependency wiring."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
    evaluate_statistical_guard,
)
from src.agent.memory import build_conversation_window
from src.agent.models import create_chat_model
from src.agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    build_critic_system_prompt,
    build_critic_user_prompt,
    build_planner_user_prompt,
    extract_message_text,
    format_manifest_answer,
)
from src.agent.state import DataAgentState
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

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


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
    """Serialize a non-SQL Spark job dispatch specification into ``generated_query``."""
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
        """Lazily build and cache the router chat model on first use."""
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


def _split_conversation_messages(
    messages: Sequence[BaseMessage],
) -> tuple[str, list[BaseMessage]]:
    """Return ``(latest_user_question, prior_messages)`` from ``state['messages']``."""
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return extract_message_text(messages[idx]), list(messages[:idx])
    return (extract_message_text(messages[-1]) if messages else ""), []


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


def _resolve_spark_job_intent(raw_intent: str, job_name: str) -> str:
    """Return a coherent non-SQL intent when dispatching a ``SPARK_JOB:`` plan."""
    normalized_job = job_name.strip().lower()
    if raw_intent in {"", "ad_hoc_sql", "general_conversation", "catalog_metadata"}:
        return normalized_job if normalized_job in SUPPORTED_SPARK_JOBS else "spark_job"
    return raw_intent


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


def _run_execution(
    execute_fn: Callable[[], dict[str, Any] | list[dict[str, Any]]],
    *,
    query: str,
    empty_message: str,
    settings: Settings,
) -> dict[str, Any]:
    """Normalize executor results, slice display rows, and track fetch truncation."""
    try:
        normalized = _normalize_query_result(execute_fn())
    except SafeSQLError as exc:
        return {
            "last_error": exc.to_dict(),
            "query_result": None,
            "analysis_rows": None,
            "status": "running",
        }

    all_rows = normalized["rows"]
    rows_returned = int(normalized.get("rows_returned", len(all_rows)))
    if rows_returned == 0:
        empty_payload = {
            "columns": normalized["columns"],
            "rows": [],
            "row_count": 0,
            "rows_returned": 0,
            "rows_shown": 0,
            "truncated": False,
        }
        return {
            "query_result": empty_payload,
            "analysis_rows": None,
            "last_error": {
                "code": "EMPTY_RESULT",
                "message": empty_message,
                "query": query,
            },
            "status": "running",
        }

    shown_rows = all_rows[: settings.sql_row_limit]
    rows_shown = len(shown_rows)
    truncated = bool(normalized.get("truncated")) or (rows_returned >= settings.sql_fetch_limit)
    display_payload = {
        "columns": normalized["columns"],
        "rows": shown_rows,
        "row_count": rows_returned,
        "rows_returned": rows_returned,
        "rows_shown": rows_shown,
        "truncated": truncated,
    }
    return {
        "query_result": display_payload,
        # Preserve full fetched rows only until statistical_guard consumes them.
        "analysis_rows": all_rows if rows_returned > rows_shown else None,
        "last_error": None,
        "status": "running",
    }


def intent_router_planner(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Classify user intent and generate a SQL query, Spark job, or direct reply."""
    question, prior_messages = _split_conversation_messages(state.get("messages", []))
    last_error = state.get("last_error")
    retry_count = state.get("retry_count", 0)
    # Skip history on retries: the correction prompt already carries the failed query and error.
    history_window = (
        build_conversation_window(
            prior_messages,
            token_budget=deps.settings.conversation_token_budget,
        )
        if last_error is None and prior_messages
        else None
    )
    planner_prompt = build_planner_user_prompt(
        question,
        history=history_window,
        last_error=last_error,
        planner_notes=state.get("planner_notes"),
        retry_count=retry_count,
    )

    llm_response = deps.get_router_model().invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=planner_prompt),
        ]
    )
    raw_text = extract_message_text(llm_response)

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
            manifest = deps.load_manifest()
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

    raw_job = plan.get("spark_job") or plan.get("job_name")
    has_job = isinstance(raw_job, str) and bool(raw_job.strip())

    if (
        not is_retry
        and intent == "general_conversation"
        and not plan.get("sql_query")
        and not has_job
    ):
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

    if has_job:
        assert isinstance(raw_job, str)
        coherent_intent = _resolve_spark_job_intent(intent, raw_job)
        raw_params = plan.get("job_params")
        if raw_params is None:
            raw_params = plan.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            return {
                "user_intent": coherent_intent,
                "generated_query": None,
                "last_error": {
                    "code": "INVALID_JOB_PARAMS",
                    "message": "Spark job parameters must be a JSON object or null.",
                    "query": raw_job,
                },
                "status": "running",
            }
        return {
            "user_intent": coherent_intent,
            "generated_query": encode_spark_job(raw_job, raw_params),
            "analysis_metric": plan.get("metric_name"),
            "last_error": None,
            "status": "running",
        }

    sql_query = plan.get("sql_query")
    if isinstance(sql_query, str) and (inline_job := decode_spark_job(sql_query)) is not None:
        job_name, job_params = inline_job
        return {
            "user_intent": _resolve_spark_job_intent(intent, job_name),
            "generated_query": encode_spark_job(job_name, job_params),
            "analysis_metric": plan.get("metric_name"),
            "last_error": None,
            "status": "running",
        }
    if not isinstance(sql_query, str) or not sql_query.strip():
        return {
            "user_intent": intent,
            "generated_query": None,
            "last_error": {
                "code": "NO_QUERY_GENERATED",
                "message": "Planner did not produce a SQL SELECT query or Spark job.",
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


def ast_validator_and_cost(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Validate SQL AST or Spark job spec and estimate execution cost for HITL gating."""
    raw_query = state.get("generated_query") or ""
    job_spec = decode_spark_job(raw_query)
    if job_spec is not None:
        job_name, job_params = job_spec
        if job_name not in SUPPORTED_SPARK_JOBS:
            return {
                "last_error": {
                    "code": "UNKNOWN_SPARK_JOB",
                    "message": (
                        f"Unsupported Spark job {job_name!r}; "
                        f"supported jobs: {sorted(SUPPORTED_SPARK_JOBS)}."
                    ),
                    "query": raw_query,
                },
                "sql_cost_estimate": None,
                "status": "running",
            }
        canonical = encode_spark_job(job_name, job_params)
        cost_info = deps.estimate_cost(canonical)
        return {
            "generated_query": canonical,
            "sql_cost_estimate": cost_info,
            "last_error": None,
            "status": "awaiting_approval" if cost_info.get("triggers_hitl") else "running",
        }

    try:
        safe_sql = validate_and_rewrite_sql(raw_query, row_limit=deps.settings.sql_fetch_limit)
        cost_info = deps.estimate_cost(safe_sql)
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


def hitl_gate(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Pause execution via LangGraph ``interrupt`` when estimated cost exceeds threshold."""
    cost_info = state.get("sql_cost_estimate") or {}
    if not cost_info.get("triggers_hitl"):
        return {"hitl_decision": "not_required", "status": "running"}

    decision = interrupt(
        {
            "type": "hitl_approval_required",
            "query": state.get("generated_query"),
            "estimated_cost": cost_info.get("estimated_cost"),
            "threshold": cost_info.get("threshold", deps.settings.hitl_cost_threshold),
            "reasons": cost_info.get("reasons", []),
            "message": "Estimated query cost exceeds threshold; human authorization required.",
        }
    )
    if _normalize_hitl_decision(decision):
        return {"hitl_decision": "approved", "status": "running"}

    cancel_msg = "Query execution cancelled by user at the human-in-the-loop approval gate."
    return {
        "hitl_decision": "rejected",
        "status": "cancelled",
        "final_response": cancel_msg,
        "messages": [AIMessage(content=cancel_msg)],
    }


def execute_sql(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Execute validated SQL through the configured SQL executor."""
    query = state.get("generated_query") or ""
    executor = deps.get_sql_executor()
    return _run_execution(
        lambda: executor(query, deps.settings),
        query=query,
        empty_message=(
            "Query executed successfully but returned 0 rows unexpectedly. "
            "Verify filter predicates (label is 0 or 1) and table selection."
        ),
        settings=deps.settings,
    )


def execute_job(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Execute a non-SQL Spark analytics job through the configured job executor."""
    raw_query = state.get("generated_query") or ""
    job_spec = decode_spark_job(raw_query)
    if job_spec is None:
        return {
            "last_error": {
                "code": "INVALID_JOB_SPEC",
                "message": "Missing or invalid Spark job specification.",
                "query": raw_query,
            },
            "query_result": None,
            "analysis_rows": None,
            "status": "running",
        }
    job_name, job_params = job_spec
    executor = deps.get_job_executor()
    return _run_execution(
        lambda: executor(job_name, job_params, deps.settings),
        query=raw_query,
        empty_message=f"Spark job {job_name!r} executed successfully but returned 0 rows.",
        settings=deps.settings,
    )


def query_rewriter(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Increment retry counter and prepare structured error feedback for the planner."""
    next_retry = state.get("retry_count", 0) + 1
    err = state.get("last_error") or {}
    notes = (
        f"Retry {next_retry}/{deps.settings.max_query_retries} triggered after error "
        f"[{err.get('code', 'UNKNOWN')}]: {err.get('message', '')}"
    )
    return {
        "retry_count": next_retry,
        "planner_notes": notes,
        "status": "running",
    }


def statistical_guard(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Run deterministic cohort significance, descriptive classification, and power checks."""
    query_result = state.get("query_result") or {}
    analysis_rows = state.get("analysis_rows")
    rows = analysis_rows if analysis_rows is not None else query_result.get("rows", [])
    columns = query_result.get("columns", [])
    guard_result = evaluate_query_result_guard(
        rows,
        columns=columns,
        preferred_metric=state.get("analysis_metric"),
        settings=deps.settings,
    )
    validation_payload = guard_result.model_dump(mode="json")
    validation_payload["rows_returned"] = int(query_result.get("rows_returned", len(rows)))
    validation_payload["rows_shown"] = int(
        query_result.get("rows_shown", len(query_result.get("rows", [])))
    )
    validation_payload["truncated"] = bool(query_result.get("truncated", False))
    return {
        "statistical_validation": validation_payload,
        "analysis_metric": guard_result.metric_name,
        "analysis_rows": None,
        "status": "running",
    }


def scientific_critic(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Synthesize the final analytical narrative constrained by the statistical guard."""
    question, prior_messages = _split_conversation_messages(state.get("messages", []))
    history_window = (
        build_conversation_window(
            prior_messages,
            token_budget=deps.settings.conversation_token_budget,
        )
        if prior_messages
        else None
    )
    raw_guard = state.get("statistical_validation")
    guard_result = (
        StatisticalGuardResult.model_validate(raw_guard)
        if raw_guard is not None
        else evaluate_statistical_guard(
            [],
            [],
            settings=deps.settings,
            metric_name="metric",
            has_subject_dimension=False,
        )
    )
    query_result = state.get("query_result") or {}
    truncated = bool(query_result.get("truncated", False))
    rows_returned = query_result.get("rows_returned")
    system_prompt = build_critic_system_prompt(
        guard_result,
        truncated=truncated,
        rows_returned=rows_returned if isinstance(rows_returned, int) else None,
        sql_fetch_limit=deps.settings.sql_fetch_limit,
    )
    user_prompt = build_critic_user_prompt(
        question,
        query=state.get("generated_query"),
        query_result=state.get("query_result"),
        cost_estimate=state.get("sql_cost_estimate"),
        guard_result=guard_result,
        history=history_window,
    )

    critic_model = deps.get_critic_model()
    base_messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = critic_model.invoke(base_messages)
    critique_text = extract_message_text(response)
    critic_calls = 1
    critic_retries = 0
    check = validate_critic_output(critique_text, guard_result)
    check_failed: dict[str, Any] | None = None

    if not check.passed:
        correction_prompt = format_critic_correction_prompt(guard_result, check)
        retry_response = critic_model.invoke(
            [
                *base_messages,
                AIMessage(content=critique_text),
                HumanMessage(content=correction_prompt),
            ]
        )
        critic_calls += 1
        critic_retries = 1
        critique_text = extract_message_text(retry_response)
        check = validate_critic_output(critique_text, guard_result)
        if not check.passed:
            critique_text = format_inconclusive_critic_fallback(guard_result, check)
            check_failed = {
                "failed": True,
                "retry_count": critic_retries,
                "critic_call_count": critic_calls,
                "missing_items": list(check.missing_items),
                "forbidden_claims": list(check.forbidden_claims),
            }

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
    """Format a terminal error response when retries are exhausted."""
    err = state.get("last_error") or {}
    attempts = state.get("retry_count", 0) + 1
    code = err.get("code", "EXECUTION_FAILED")
    message = err.get("message", "Unknown query execution error.")
    failed_query = err.get("query") or state.get("generated_query") or "N/A"
    explanation = (
        f"Unable to complete query after {attempts} attempt(s) "
        f"(max retries: {deps.settings.max_query_retries}).\n"
        f"- Error code: `{code}`\n"
        f"- Details: {message}\n"
        f"- Last query: `{failed_query}`"
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
        intent_router_planner,
        ast_validator_and_cost,
        hitl_gate,
        execute_sql,
        execute_job,
        query_rewriter,
        statistical_guard,
        scientific_critic,
        handle_failure,
    )
    return {fn.__name__: _bind_node(fn, deps) for fn in node_fns}
