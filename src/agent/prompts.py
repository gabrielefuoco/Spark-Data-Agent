"""Prompt builders and deterministic formatters for the LangGraph agent."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agent.guards import StatisticalGuardResult, format_critic_guard_instructions
from src.agent.memory import ConversationWindow, extract_message_text
from src.lakehouse.manifest import DatasetManifest

LAKEHOUSE_SCHEMA_GUIDE = """Available Spark SQL lakehouse views (SELECT/WITH/UNION only, no qualified names):
1. `gold_user_longitudinal_features` (pre-aggregated per subject — prefer for user/cohort comparisons):
   - subject_id (STRING), label (INT: 1=depressed, 0=control), post_count (BIGINT),
     first_post_at (TIMESTAMP), last_post_at (TIMESTAMP), span_days (INT),
     active_days (BIGINT), posts_per_active_day (DOUBLE),
     mean_text_length (DOUBLE), stddev_text_length (DOUBLE),
     night_post_count (BIGINT), night_post_ratio (DOUBLE),
     session_count (BIGINT), mean_inter_post_gap_seconds (DOUBLE),
     mean_night_post_rolling_30d (DOUBLE), mean_verbosity_zscore (DOUBLE),
     max_abs_verbosity_zscore (DOUBLE)
2. `gold_cohort_lexical_matrix` (pre-aggregated n-grams per cohort):
   - label (INT: 1=depressed, 0=control), ngram_type (STRING: 'unigram'|'bigram'),
     ngram (STRING), frequency (BIGINT), relative_frequency (DOUBLE),
     idf_weight (DOUBLE), tfidf_weight (DOUBLE), tfidf_rank (INT)
3. `gold_cohort_topic_matrix` (LDA topics per cohort):
   - label (INT), topic (INT), mean_topic_probability (DOUBLE),
     document_count (BIGINT), top_terms (STRING)
4. `silver_posts` (atomic posts, 1 row per post, partitioned by label and year — expensive full scan):
   - subject_id (STRING), title (STRING), text (STRING), info (STRING),
     date (TIMESTAMP), year (INT), label (INT: 1=depressed, 0=control)
5. `silver_quarantine` (rejected rows from ETL):
   - subject_id (STRING), title (STRING), raw_date (STRING), info (STRING),
     text (STRING), label (INT), _rejection_reason (STRING)

Only the columns listed here exist. Never invent a column name.

Definitions you must not improvise:
- **Night post** means `HOUR(date) >= 0 AND HOUR(date) < 5` (00:00–05:00).
  `night_post_count` and `night_post_ratio` use exactly this window, so any
  ad-hoc SQL that counts night posts must use it too — otherwise two answers to
  the same question disagree.
- **Night post ratio** is `night_post_count / post_count` per subject, a 0–1
  share, not a percentage."""

ROUTER_SYSTEM_PROMPT = f"""You are the Intent Router and Spark SQL Query Planner for the eRISK depression lakehouse.

Classify the user request and respond with a single JSON object (no markdown prose outside JSON):
{{
  "user_intent": "catalog_metadata" | "lexical_analysis" | "circadian_study" | "ad_hoc_sql" | "general_conversation",
  "answer_from_manifest": true | false,
  "sql_query": "<Spark SQL SELECT query or null>",
  "metric_name": "<short metric name for statistical guard or null>",
  "direct_response": "<string when user_intent is general_conversation, else null>"
}}

Rules:
- Set `answer_from_manifest: true` and `sql_query: null` ONLY when the question asks for global corpus counts, subjects per class, posts per class, global min/max date range, or ETL quarantine totals available in `gold_dataset_manifest.json`.
- For cohort statistical comparisons (e.g., circadian night posting, post length, posting rate), query `gold_user_longitudinal_features` selecting per-subject rows including `subject_id`, `label`, and the numeric metric column so the statistical guard can compute sample sizes and hypothesis tests.
- When recent conversation history is provided, resolve follow-up references (such as ellipsis, pronouns, or year/metric overrides) against the prior question and executed query.
- Never use DDL/DML, file paths, or qualified table names.

Non-SQL Spark job dispatch:
- When the user explicitly asks to recompute circadian/window features or fit/recompute the Spark ML LDA topic model from raw Silver posts, set `sql_query: null` and specify `"spark_job": "circadian_features" | "topic_model"` with optional `"job_params": {{...}}`.
- When dispatching a `spark_job`, set `user_intent` to the job name (`"circadian_features"` or `"topic_model"`) or its domain intent (`"circadian_study"` or `"lexical_analysis"`), never `"ad_hoc_sql"`.

{LAKEHOUSE_SCHEMA_GUIDE}"""


def _format_planner_history(
    history: ConversationWindow | Sequence[BaseMessage] | None,
) -> str:
    """Render every kept message at full fidelity with clear block delimiters."""
    if history is None:
        return ""
    if isinstance(history, ConversationWindow):
        messages: Sequence[BaseMessage] = history.messages
        dropped_messages = history.dropped_messages
    else:
        messages = history
        dropped_messages = 0

    if not messages and dropped_messages <= 0:
        return ""

    entries: list[str] = []
    for msg in messages:
        text = extract_message_text(msg)
        if isinstance(msg, HumanMessage):
            if text.strip():
                entries.append(f"User:\n{text}")
        elif isinstance(msg, AIMessage):
            raw_query = msg.additional_kwargs.get("generated_query")
            query_text = (
                raw_query.strip() if isinstance(raw_query, str) and raw_query.strip() else ""
            )
            parts: list[str] = []
            if query_text:
                parts.append(f"Generated query:\n{query_text}")
            if text.strip():
                parts.append(f"Answer:\n{text}")
            if parts:
                entries.append("Assistant:\n" + "\n".join(parts))
        elif text.strip():
            entries.append(f"{msg.type.capitalize()}:\n{text}")

    if not entries and dropped_messages <= 0:
        return ""

    blocks = ["--- Conversation history (oldest to newest) ---"]
    if dropped_messages > 0:
        noun = "message" if dropped_messages == 1 else "messages"
        blocks.append(f"[{dropped_messages} earlier {noun} omitted due to token budget]")
    blocks.extend(entries)
    blocks.append("--- End conversation history ---")
    return "\n\n".join(blocks)


def build_planner_user_prompt(
    question: str,
    *,
    history: ConversationWindow | Sequence[BaseMessage] | None = None,
    last_error: dict[str, Any] | None = None,
    planner_notes: str | None = None,
    retry_count: int = 0,
) -> str:
    """Build the planner prompt, prepending windowed history on non-retry turns."""
    if last_error is not None:
        error_json = json.dumps(last_error, ensure_ascii=False, indent=2)
        notes_block = f"\nCorrection guidance:\n{planner_notes}\n" if planner_notes else ""
        return (
            f"User request:\n{question}\n\n"
            f"PREVIOUS QUERY FAILED (retry {retry_count}). Fix the SQL query and do not repeat the mistake.\n"
            f"Structured error:\n{error_json}"
            f"{notes_block}\n"
            "Return a valid JSON object with the corrected `sql_query`."
        )

    history_block = _format_planner_history(history)
    return (
        f"{history_block}\n\nUser request:\n{question}"
        if history_block
        else f"User request:\n{question}"
    )


def build_critic_system_prompt(
    guard_result: StatisticalGuardResult,
    *,
    truncated: bool = False,
    rows_returned: int | None = None,
    sql_fetch_limit: int | None = None,
) -> str:
    """Build the scientific critic system prompt embedding mandatory guardrails."""
    guard_instructions = format_critic_guard_instructions(guard_result)
    truncation_note = ""
    if truncated:
        limit_detail = (
            f" (`sql_fetch_limit={sql_fetch_limit}`)"
            if sql_fetch_limit is not None
            else " (`sql_fetch_limit`)"
        )
        fetched_detail = (
            f" with {rows_returned} rows returned" if rows_returned is not None else ""
        )
        truncation_note = (
            "\n\nMANDATORY CAPPED-SAMPLE DISCLOSURE (`truncated=true`):\n"
            f"The query result reached the fetch ceiling{limit_detail}{fetched_detail}. "
            "Your narrative MUST explicitly state that the statistical analysis was performed on "
            "a capped/truncated sample rather than the full population, and note that findings "
            "may reflect partition/order selection bias."
        )
    return (
        "You are the Scientific Interpreter & Critic for the eRISK computational linguistics lakehouse.\n"
        "Label 1 represents the depressed cohort and label 0 represents the control cohort.\n"
        "Always respond in the SAME language as the user's question.\n"
        "Never make clinical diagnoses; keep interpretations strictly statistical and population-level, "
        "noting dataset limitations and that correlation does not imply causation.\n\n"
        f"{guard_instructions}"
        f"{truncation_note}"
    )


def build_critic_user_prompt(
    question: str,
    *,
    query: str | None,
    query_result: dict[str, Any] | None,
    cost_estimate: dict[str, Any] | None,
    guard_result: StatisticalGuardResult,
    history: ConversationWindow | Sequence[BaseMessage] | None = None,
) -> str:
    """Format executed SQL evidence, guard statistics, and optional history for the critic."""
    qr = query_result or {}
    rows = qr.get("rows", [])
    rows_returned = qr.get("rows_returned", qr.get("row_count", len(rows)))
    rows_shown = qr.get("rows_shown", len(rows))
    truncated = bool(qr.get("truncated", False))
    preview_rows = rows[:20]
    payload = {
        "question": question,
        "executed_sql": query,
        "cost_estimate": cost_estimate,
        "row_count": qr.get("row_count", rows_returned),
        "rows_returned": rows_returned,
        "rows_shown": rows_shown,
        "truncated": truncated,
        "columns": qr.get("columns", []),
        "sample_rows": preview_rows,
        "statistical_validation": guard_result.model_dump(mode="json"),
    }
    history_block = _format_planner_history(history)
    evidence_block = (
        "Interpret the following Spark SQL results strictly within the mandatory statistical guardrails:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )
    return f"{history_block}\n\n{evidence_block}" if history_block else evidence_block


def format_manifest_answer(manifest: DatasetManifest, question: str) -> str:
    """Format global dataset statistics from ``gold_dataset_manifest.json``."""
    subjects_0 = manifest.subjects_per_class.get("0", 0)
    subjects_1 = manifest.subjects_per_class.get("1", 0)
    posts_0 = manifest.posts_per_class.get("0", 0)
    posts_1 = manifest.posts_per_class.get("1", 0)
    quarantine_breakdown = (
        ", ".join(
            f"{reason}: {count}" for reason, count in sorted(manifest.quarantine_by_reason.items())
        )
        or "none"
    )
    return (
        f"Dataset catalog summary (from `gold_dataset_manifest.json`, zero Spark jobs executed) "
        f"for request: *{question}*\n\n"
        f"- **Total valid posts**: {manifest.total_posts} "
        f"(depressed `label=1`: {posts_1}, control `label=0`: {posts_0})\n"
        f"- **Total valid subjects**: {manifest.total_subjects} "
        f"(depressed `label=1`: {subjects_1}, control `label=0`: {subjects_0})\n"
        f"- **Date range**: {manifest.min_date or 'N/A'} -> {manifest.max_date or 'N/A'}\n"
        f"- **Quarantined rows**: {manifest.quarantine_count} ({quarantine_breakdown})\n"
        f"- **Manifest generated at**: {manifest.generated_at}"
    )
