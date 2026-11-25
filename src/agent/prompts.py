"""Prompt builders and deterministic formatters for the tool-calling LangGraph agent."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agent.guards import (
    StatisticalGuardResult,
    format_multi_critic_guard_instructions,
)
from src.agent.memory import ConversationWindow, extract_message_text
from src.agent.state import ToolResultEntry, ToolTraceEntry
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

AGENT_SYSTEM_PROMPT = f"""You are the Analytical Data Agent for the eRISK depression lakehouse.

Use your tools iteratively to answer the user's question thoroughly and accurately:
- `read_dataset_manifest()`: Read global corpus counts, subjects per class, posts per class, date range, and ETL quarantine totals from `gold_dataset_manifest.json` with zero Spark jobs. Always use this for catalog metadata instead of scanning raw tables; note that catalog numbers are purely descriptive.
- `run_safe_sql(query)`: Execute read-only Spark SQL (`SELECT`/`WITH`/`UNION`) against the allowlisted lakehouse views. Never use DDL/DML, file paths, or qualified table names.
- `run_spark_job(job, params)`: Run heavy non-SQL Spark analytics (`circadian_features` or `topic_model`) over `silver_posts` when asked to recompute circadian features or fit/recompute LDA topics.

Analytical rules:
- Broad exploratory questions (e.g., how the two cohorts differ across multiple dimensions) should issue multiple targeted queries across `gold_user_longitudinal_features`, `gold_cohort_lexical_matrix`, `gold_cohort_topic_matrix`, and the manifest before concluding.
- For cohort statistical comparisons (e.g., circadian night posting, post length, posting rate, sessionization), select per-subject rows including `subject_id`, `label`, and the numeric metric column from `gold_user_longitudinal_features` so the statistical guard can compute sample sizes, p-values, and Cohen's d.
- If a tool call returns an error (such as a SQL syntax or column error), read the error and adjust your query; do NOT repeat an identical tool call.
- If a tool call is rejected at the human-in-the-loop gate, do NOT re-run the same expensive query; either use a cheaper/partition-pruned Gold or Silver query or explain the limitation.
- When you have gathered sufficient evidence (or when the tool budget is exhausted), stop calling tools and write a concise summary of what you observed, in the SAME language as the user's question.

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
    """Build a prompt prepending windowed conversation history."""
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


def format_tool_results_summary(
    tool_trace: Sequence[ToolTraceEntry],
    tool_results: Sequence[ToolResultEntry],
    *,
    tool_call_count: int,
    max_tool_calls: int,
    budget_exhausted: bool = False,
) -> str:
    """Render a compact summary of tool calls and guard outcomes gathered in this turn."""
    lines: list[str] = [f"Tool call budget used: {tool_call_count}/{max_tool_calls}."]
    if tool_trace:
        lines.append("Tool results gathered so far:")
        result_by_id = {entry["tool_call_id"]: entry for entry in tool_results}
        for idx, trace in enumerate(tool_trace, start=1):
            call_id = trace["tool_call_id"]
            guard = trace["guard_payload"]
            status = trace["status"]
            name = trace["name"]
            if call_id in result_by_id and status == "executed":
                metric = guard.get("metric_name", "metric")
                assess = guard.get("assessment_status", "descriptive")
                if assess == "assessed":
                    p_val = guard.get("p_value")
                    d_val = guard.get("effect_size_cohens_d")
                    p_str = f"{p_val:.4g}" if isinstance(p_val, (int, float)) else "N/A"
                    d_str = f"{d_val:.4f}" if isinstance(d_val, (int, float)) else "N/A"
                    verdict = (
                        f"assessed ({metric}: n_dep={guard.get('sample_size_depressed')}, "
                        f"n_ctrl={guard.get('sample_size_control')}, p={p_str}, d={d_str}, "
                        f"sig={guard.get('is_significant')}, underpowered={guard.get('is_underpowered')})"
                    )
                else:
                    verdict = f"{assess} ({metric})"
                lines.append(
                    f"  {idx}. {name} -> {trace['rows_returned']} row(s), guard={verdict}"
                )
            else:
                err_code = (trace.get("error") or {}).get("code", status)
                lines.append(f"  {idx}. {name} -> status={status} ({err_code})")

    if budget_exhausted:
        lines.append(
            f"TOOL CALL BUDGET EXHAUSTED ({max_tool_calls}/{max_tool_calls}): "
            "Do NOT call any more tools. Answer the user's question now using only "
            "the results gathered so far, and explicitly state that the tool-call budget was reached."
        )
    return "\n".join(lines)


def build_critic_system_prompt(
    guard_result: StatisticalGuardResult | Sequence[StatisticalGuardResult],
    *,
    truncated: bool = False,
    rows_returned: int | None = None,
    sql_fetch_limit: int | None = None,
    budget_exhausted: bool = False,
    max_tool_calls: int | None = None,
) -> str:
    """Build the scientific critic system prompt embedding mandatory guardrails."""
    guard_instructions = format_multi_critic_guard_instructions(
        [guard_result] if isinstance(guard_result, StatisticalGuardResult) else guard_result
    )
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
            f"One or more query results reached the fetch ceiling{limit_detail}{fetched_detail}. "
            "Your narrative MUST explicitly state that the statistical analysis was performed on "
            "a capped/truncated sample rather than the full population, and note that findings "
            "may reflect partition/order selection bias."
        )
    budget_note = ""
    if budget_exhausted:
        limit_str = str(max_tool_calls) if max_tool_calls is not None else "configured limit"
        budget_note = (
            "\n\nMANDATORY BUDGET EXHAUSTION DISCLOSURE:\n"
            f"The agent reached the maximum tool-call budget (`max_tool_calls={limit_str}`). "
            "Your response MUST explicitly mention that the tool-call budget limit was reached "
            "and summarize only the results actually gathered."
        )
    return (
        "You are the Scientific Interpreter & Critic for the eRISK computational linguistics lakehouse.\n"
        "Label 1 represents the depressed cohort and label 0 represents the control cohort.\n"
        "Always respond in the SAME language as the user's question.\n"
        "Never make clinical diagnoses; keep interpretations strictly statistical and population-level, "
        "noting dataset limitations and that correlation does not imply causation.\n\n"
        f"{guard_instructions}"
        f"{truncation_note}"
        f"{budget_note}"
    )


def _format_tool_result_payload(entry: ToolResultEntry, index: int) -> dict[str, Any]:
    qr = entry.get("result") or {}
    rows = qr.get("rows", [])
    rows_returned = qr.get("rows_returned", qr.get("row_count", len(rows)))
    rows_shown = qr.get("rows_shown", len(rows))
    return {
        "index": index,
        "tool_name": entry.get("name"),
        "arguments": entry.get("arguments"),
        "executed_query": entry.get("query"),
        "cost_estimate": entry.get("cost_estimate"),
        "row_count": qr.get("row_count", rows_returned),
        "rows_returned": rows_returned,
        "rows_shown": rows_shown,
        "truncated": bool(qr.get("truncated", False)),
        "columns": qr.get("columns", []),
        "sample_rows": rows[:20],
        "statistical_validation": entry.get("guard_payload"),
    }


def build_critic_user_prompt(
    question: str,
    *,
    tool_results: Sequence[ToolResultEntry] | None = None,
    query: str | None = None,
    query_result: dict[str, Any] | None = None,
    cost_estimate: dict[str, Any] | None = None,
    guard_result: StatisticalGuardResult | None = None,
    history: ConversationWindow | Sequence[BaseMessage] | None = None,
    agent_draft: str | None = None,
    budget_exhausted: bool = False,
    max_tool_calls: int | None = None,
) -> str:
    """Format all tool results, guard statistics, and optional history for the critic."""
    if tool_results is not None:
        payload: dict[str, Any] = {
            "question": question,
            "result_count": len(tool_results),
            "results": [
                _format_tool_result_payload(entry, idx)
                for idx, entry in enumerate(tool_results, start=1)
            ],
        }
        if agent_draft:
            payload["agent_draft_summary"] = agent_draft
        if budget_exhausted:
            payload["budget_exhausted"] = True
            if max_tool_calls is not None:
                payload["max_tool_calls"] = max_tool_calls
    else:
        qr = query_result or {}
        rows = qr.get("rows", [])
        rows_returned = qr.get("rows_returned", qr.get("row_count", len(rows)))
        rows_shown = qr.get("rows_shown", len(rows))
        payload = {
            "question": question,
            "executed_sql": query,
            "cost_estimate": cost_estimate,
            "row_count": qr.get("row_count", rows_returned),
            "rows_returned": rows_returned,
            "rows_shown": rows_shown,
            "truncated": bool(qr.get("truncated", False)),
            "columns": qr.get("columns", []),
            "sample_rows": rows[:20],
            "statistical_validation": (
                guard_result.model_dump(mode="json") if guard_result is not None else None
            ),
        }

    history_block = _format_planner_history(history)
    evidence_block = (
        "Interpret all gathered tool results strictly within the mandatory statistical guardrails:\n"
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


_DESCRIPTIVE_FALLBACK_DESCRIPTIONS: dict[str, str] = {
    "descriptive: lexical_frequency_table": (
        "Tabella descrittiva con gli n-grammi principali e le relative frequenze/pesi TF-IDF per coorte"
    ),
    "descriptive: topic_distribution_table": (
        "Tabella descrittiva con i topic latenti (LDA) e i termini principali per coorte"
    ),
    "descriptive: cohort_aggregate_metric": (
        "Conteggi o metriche aggregate per coorte senza granularità per soggetto"
    ),
    "descriptive: single_cohort_only": (
        "Statistiche descrittive riferite a una sola coorte (confronto tra coorti non applicabile)"
    ),
    "descriptive: no_numeric_metric_column": (
        "Tabella descrittiva priva di una metrica numerica continua"
    ),
    "descriptive: empty_result": "La query non ha restituito righe",
}

_DEFAULT_DESCRIPTIVE_FALLBACK = (
    "Statistiche descrittive senza test di significatività a livello di soggetto"
)


def _descriptive_summary_core_it(metric_name: str, descriptive_reason: str | None) -> str:
    if descriptive_reason == "descriptive: no_cohort_label_column":
        if metric_name == "total_posts":
            return (
                "Riepilogo descrittivo del catalogo del dataset (conteggi globali di post e "
                "soggetti senza confronto tra coorti)"
            )
        return "Tabella descrittiva riepilogativa senza colonna di confronto tra coorti"
    if descriptive_reason and descriptive_reason in _DESCRIPTIVE_FALLBACK_DESCRIPTIONS:
        return _DESCRIPTIVE_FALLBACK_DESCRIPTIONS[descriptive_reason]
    return _DEFAULT_DESCRIPTIVE_FALLBACK


def describe_descriptive_guard_it(metric_name: str, descriptive_reason: str | None) -> str:
    """Return a readable Italian description of what a ``descriptive`` result holds."""
    core = _descriptive_summary_core_it(metric_name, descriptive_reason)
    return f"{core}; non viene affermata alcuna differenza tra le coorti."


def _escape_markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def format_statistical_appendix(guards: Sequence[StatisticalGuardResult]) -> str:
    """Render a deterministic Italian statistical appendix table when any ``assessed`` result exists.

    All numeric values (sample sizes, ``p``-values, Cohen's ``d``, and verdicts) come
    exclusively from the deterministic statistical guards, never from the model's
    recollection. As a consequence, deterministic fallback should fire only on genuine
    claim violations rather than transcription misses, and the forbidden-claim tripwire
    continues to catch prose that asserts a difference where this table reports
    ``non significativo``.
    """
    has_assessed = any(
        g.assessment_status == "assessed"
        and g.p_value is not None
        and g.effect_size_cohens_d is not None
        for g in guards
    )
    if not has_assessed:
        return ""

    lines = [
        "## Appendice statistica (generata dal sistema)",
        "",
        "| Metrica / Tabella | n depressi | n controlli | p | Cohen's d | Esito |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for guard in guards:
        metric_cell = _escape_markdown_cell(guard.metric_name)
        if (
            guard.assessment_status == "assessed"
            and guard.p_value is not None
            and guard.effect_size_cohens_d is not None
        ):
            n_dep = str(guard.sample_size_depressed)
            n_ctrl = str(guard.sample_size_control)
            p_cell = f"p = {guard.p_value:.6g}"
            d_display = (
                "0.0000"
                if abs(guard.effect_size_cohens_d) < 1e-12
                else f"{guard.effect_size_cohens_d:.6g}"
            )
            d_cell = f"d = {d_display}"
            if guard.is_significant and not guard.is_underpowered:
                verdict = "significativo"
            elif guard.is_underpowered:
                verdict = "non significativo (campione insufficiente)"
            else:
                verdict = "non significativo"
        elif guard.assessment_status == "descriptive":
            n_dep = "—"
            n_ctrl = "—"
            p_cell = "—"
            d_cell = "—"
            desc = _escape_markdown_cell(
                _descriptive_summary_core_it(guard.metric_name, guard.descriptive_reason)
            )
            verdict = f"descrittivo — {desc}"
        else:
            n_dep = "—"
            n_ctrl = "—"
            p_cell = "—"
            d_cell = "—"
            reason = _escape_markdown_cell(
                guard.cannot_assess_reason or "manca la dimensione per soggetto (`subject_id`)"
            )
            verdict = f"non valutabile — {reason}"
        lines.append(
            f"| `{metric_cell}` | {n_dep} | {n_ctrl} | {p_cell} | {d_cell} | {verdict} |"
        )
    return "\n".join(lines)


def format_critic_fallback_closing_line(
    *,
    has_missing_items: bool,
    has_forbidden_claims: bool,
) -> str:
    """Return the single Italian closing note explaining why narrative prose was withheld."""
    if has_missing_items:
        return (
            "Nota: l'interpretazione narrativa è stata omessa perché la risposta generata "
            "non ha citato le statistiche richieste."
        )
    if has_forbidden_claims:
        return (
            "Nota: l'interpretazione narrativa è stata omessa perché la risposta generata "
            "conteneva affermazioni comparative non supportate dai vincoli statistici."
        )
    return (
        "Nota: l'interpretazione narrativa è stata omessa perché la risposta generata "
        "non ha soddisfatto i vincoli statistici richiesti."
    )
