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
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.agent.cost import estimate_job_cost, estimate_sql_cost
from src.agent.guards import (
    StatisticalGuardResult,
    SubjectObservation,
    evaluate_statistical_guard,
)
from src.agent.models import create_chat_model
from src.agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    build_critic_system_prompt,
    build_critic_user_prompt,
    build_planner_user_prompt,
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
CostEstimator = Callable[[str, Settings], dict[str, Any]]
ManifestLoader = Callable[[], DatasetManifest]

SUPPORTED_SPARK_JOBS: frozenset[str] = frozenset({"circadian_features", "topic_model"})
_SPARK_JOB_PREFIX = "SPARK_JOB:"
_ROUTER_JOB_DISPATCH_ADDENDUM = (
    "\n\nNon-SQL Spark job dispatch:\n"
    "- When the user explicitly asks to recompute circadian/window features or fit/recompute "
    "the Spark ML LDA topic model from raw Silver posts, set `sql_query: null` and specify "
    '`"spark_job": "circadian_features" | "topic_model"` with optional `"job_params": {{...}}`.'
)

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
    if not body:
        return None
    if body.startswith("{"):
        parsed = json.loads(body)
        if not isinstance(parsed, dict) or not parsed.get("job"):
            return None
        raw_params = parsed.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        return str(parsed["job"]).strip().lower(), params
    return body.lower(), {}


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
            ).limit(resolved_settings.sql_row_limit)
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
                resolved_settings.sql_row_limit
            )

        raw_rows = result_df.collect()
        rows = [
            {str(k): _to_json_value(v) for k, v in row.asDict(recursive=True).items()}
            for row in raw_rows
        ]
        return {
            "columns": list(result_df.columns),
            "rows": rows,
            "row_count": len(rows),
        }

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
    cost_estimator: CostEstimator | None = None
    manifest_loader: ManifestLoader | None = None

    def get_router_model(self) -> BaseChatModel:
        """Lazily build and cache the router chat model on first use."""
        if self.router_model is None:
            self.router_model = create_chat_model(
                "router", settings=self.settings, temperature=0.0
            )
        return self.router_model

    def get_critic_model(self) -> BaseChatModel:
        """Lazily build and cache the critic chat model on first use."""
        if self.critic_model is None:
            self.critic_model = create_chat_model(
                "critic", settings=self.settings, temperature=0.0
            )
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
        if self.cost_estimator is not None:
            return self.cost_estimator(query, self.settings)
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


@dataclass(frozen=True)
class _ExtractedGroupMetrics:
    """Subject-attributed cohort observations extracted from SQL result rows."""

    depressed_observations: list[SubjectObservation]
    control_observations: list[SubjectObservation]
    metric_name: str
    has_subject_dimension: bool
    raw_observation_count_depressed: int
    raw_observation_count_control: int


def _find_column_key(candidate_cols: Sequence[str], target_lower: str) -> str | None:
    for col in candidate_cols:
        if col.lower() == target_lower:
            return col
    return None


def _extract_group_metrics(
    rows: list[dict[str, Any]],
    preferred_metric: str | None = None,
    columns: Sequence[str] | None = None,
) -> _ExtractedGroupMetrics:
    """Extract subject-level depressed (label=1) and control (label=0) observations."""
    metric_label = preferred_metric or "metric"
    candidate_cols: list[str] = list(columns or [])
    for row in rows:
        for key in row:
            if key not in candidate_cols:
                candidate_cols.append(key)

    subject_col = _find_column_key(candidate_cols, "subject_id")
    label_col = _find_column_key(candidate_cols, "label")
    has_subject_col = subject_col is not None

    if not rows or label_col is None:
        return _ExtractedGroupMetrics(
            depressed_observations=[],
            control_observations=[],
            metric_name=metric_label,
            has_subject_dimension=has_subject_col,
            raw_observation_count_depressed=0,
            raw_observation_count_control=0,
        )

    metric_col: str | None = None
    if preferred_metric:
        preferred_key = _find_column_key(candidate_cols, preferred_metric.lower())
        if preferred_key and any(_as_finite_float(r.get(preferred_key)) is not None for r in rows):
            metric_col = preferred_key
    if metric_col is None:
        for col in candidate_cols:
            if col.lower() in _EXCLUDED_METRIC_COLUMNS:
                continue
            if any(_as_finite_float(r.get(col)) is not None for r in rows):
                metric_col = col
                break

    if metric_col is None:
        return _ExtractedGroupMetrics(
            depressed_observations=[],
            control_observations=[],
            metric_name=metric_label,
            has_subject_dimension=has_subject_col,
            raw_observation_count_depressed=0,
            raw_observation_count_control=0,
        )

    depressed: list[SubjectObservation] = []
    control: list[SubjectObservation] = []
    raw_dep_count = 0
    raw_ctrl_count = 0

    for row in rows:
        raw_label = row.get(label_col)
        val = _as_finite_float(row.get(metric_col))
        if raw_label is None or val is None:
            continue
        try:
            label_int = int(raw_label)
        except (TypeError, ValueError):
            continue
        if label_int not in (0, 1):
            continue

        if label_int == 1:
            raw_dep_count += 1
        else:
            raw_ctrl_count += 1

        if subject_col is None:
            continue
        raw_subject = row.get(subject_col)
        if raw_subject is None:
            continue
        subject_id = str(raw_subject).strip()
        if not subject_id:
            continue

        obs = SubjectObservation(subject_id=subject_id, value=val)
        if label_int == 1:
            depressed.append(obs)
        else:
            control.append(obs)

    has_subject_dimension = has_subject_col and (
        (raw_dep_count + raw_ctrl_count) == 0 or (len(depressed) + len(control)) > 0
    )

    return _ExtractedGroupMetrics(
        depressed_observations=depressed,
        control_observations=control,
        metric_name=metric_col,
        has_subject_dimension=has_subject_dimension,
        raw_observation_count_depressed=raw_dep_count,
        raw_observation_count_control=raw_ctrl_count,
    )


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
) -> dict[str, Any]:
    """Normalize executor results and convert empty/error outcomes to state updates."""
    try:
        normalized = _normalize_query_result(execute_fn())
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
                "message": empty_message,
                "query": query,
            },
            "status": "running",
        }

    return {
        "query_result": normalized,
        "last_error": None,
        "status": "running",
    }


def intent_router_planner(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Classify user intent and generate a SQL query, Spark job, or direct reply."""
    question = _latest_user_question(state.get("messages", []))
    last_error = state.get("last_error")
    retry_count = state.get("retry_count", 0)
    planner_prompt = build_planner_user_prompt(
        question,
        last_error=last_error,
        planner_notes=state.get("planner_notes"),
        retry_count=retry_count,
    )

    llm_response = deps.get_router_model().invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT + _ROUTER_JOB_DISPATCH_ADDENDUM),
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
        raw_params = plan.get("job_params")
        if raw_params is None:
            raw_params = plan.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            return {
                "user_intent": intent,
                "generated_query": None,
                "last_error": {
                    "code": "INVALID_JOB_PARAMS",
                    "message": "Spark job parameters must be a JSON object or null.",
                    "query": raw_job,
                },
                "status": "running",
            }
        return {
            "user_intent": intent,
            "generated_query": encode_spark_job(raw_job, raw_params),
            "analysis_metric": plan.get("metric_name"),
            "last_error": None,
            "status": "running",
        }

    sql_query = plan.get("sql_query")
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
        safe_sql = validate_and_rewrite_sql(raw_query, row_limit=deps.settings.sql_row_limit)
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
            "status": "running",
        }
    job_name, job_params = job_spec
    executor = deps.get_job_executor()
    return _run_execution(
        lambda: executor(job_name, job_params, deps.settings),
        query=raw_query,
        empty_message=f"Spark job {job_name!r} executed successfully but returned 0 rows.",
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
    """Run deterministic cohort significance and power checks on query results."""
    query_result = state.get("query_result") or {}
    rows = query_result.get("rows", [])
    columns = query_result.get("columns", [])
    extracted = _extract_group_metrics(
        rows,
        preferred_metric=state.get("analysis_metric"),
        columns=columns,
    )
    guard_result = evaluate_statistical_guard(
        extracted.depressed_observations,
        extracted.control_observations,
        settings=deps.settings,
        metric_name=extracted.metric_name,
        has_subject_dimension=extracted.has_subject_dimension,
        raw_observation_count_depressed=extracted.raw_observation_count_depressed,
        raw_observation_count_control=extracted.raw_observation_count_control,
    )
    return {
        "statistical_validation": guard_result.model_dump(mode="json"),
        "analysis_metric": extracted.metric_name,
        "status": "running",
    }


def scientific_critic(state: DataAgentState, deps: AgentRuntimeDeps) -> dict[str, Any]:
    """Synthesize the final analytical narrative constrained by the statistical guard."""
    question = _latest_user_question(state.get("messages", []))
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
    system_prompt = build_critic_system_prompt(guard_result)
    user_prompt = build_critic_user_prompt(
        question,
        query=state.get("generated_query"),
        query_result=state.get("query_result"),
        cost_estimate=state.get("sql_cost_estimate"),
        guard_result=guard_result,
    )

    response = deps.get_critic_model().invoke(
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
