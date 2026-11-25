"""Offline tests for the tool-calling LangGraph Data Agent (fake LLMs + fake executors)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pyspark.sql import SparkSession
from src.agent.cost import estimate_sql_cost
from src.agent.critic_check import validate_critic_output
from src.agent.graph import build_agent_graph
from src.agent.guards import (
    CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS,
    StatisticalGuardResult,
    evaluate_query_result_guard,
)
from src.agent.memory import count_tokens
from src.agent.nodes import make_spark_sql_executor
from src.agent.prompts import format_manifest_digest, format_statistical_appendix
from src.agent.state import make_initial_state
from src.config import Settings
from src.lakehouse.manifest import DatasetManifest
from tests.conftest import RecordingFakeChatModel


def _tool_msg(
    name: str,
    args: dict[str, Any] | None = None,
    call_id: str = "call_1",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id}],
    )


def _sql_msg(query: str, call_id: str = "call_1") -> AIMessage:
    return _tool_msg("run_safe_sql", {"query": query}, call_id)


def _make_manifest() -> DatasetManifest:
    return DatasetManifest(
        total_posts=5000,
        total_subjects=100,
        subjects_per_class={"0": 50, "1": 50},
        posts_per_class={"0": 2400, "1": 2600},
        min_date="2015-01-01T00:00:00",
        max_date="2016-12-31T23:59:59",
        quarantine_count=12,
        quarantine_by_reason={"malformed_date": 7, "orphan_subject_id": 5},
        generated_at="2026-01-01T00:00:00+00:00",
    )


def _counting_manifest() -> tuple[list[int], Callable[[], DatasetManifest]]:
    calls = [0]

    def _load() -> DatasetManifest:
        calls[0] += 1
        return _make_manifest()

    return calls, _load


def _cohort_rows(
    n_dep: int,
    n_ctrl: int | None = None,
    *,
    metric_name: str = "night_post_ratio",
    dep_base: float = 0.55,
    ctrl_base: float = 0.10,
    mod: int = 5,
    step: float = 0.02,
    single_dep_subject: bool = False,
) -> list[dict[str, Any]]:
    ctrl_count = n_dep if n_ctrl is None else n_ctrl
    dep = [
        {
            "subject_id": "dep_only" if single_dep_subject else f"dep_{i}",
            "label": 1,
            metric_name: dep_base + (i % mod) * step,
        }
        for i in range(n_dep)
    ]
    ctrl = [
        {
            "subject_id": f"ctrl_{i}",
            "label": 0,
            metric_name: ctrl_base + (i % mod) * step,
        }
        for i in range(ctrl_count)
    ]
    return dep + ctrl


def _critic(*texts: str) -> RecordingFakeChatModel:
    return RecordingFakeChatModel([AIMessage(content=t) for t in texts])


def _mixed_sql(cohort_rows: list[dict[str, Any]]) -> Callable[[str, Settings], list[dict[str, Any]]]:
    def _exec(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        if "silver_posts" in query:
            return [{"total_posts": 1200}]
        if "gold_cohort_lexical_matrix" in query:
            return [{"label": 1, "ngram": "feel alone", "frequency": 42}]
        return cohort_rows

    return _exec


def test_three_query_investigation_collects_three_guards_and_one_critic_call() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    cohort_rows = _cohort_rows(35)
    expected_guard = evaluate_query_result_guard(cohort_rows, settings=settings)
    assert expected_guard.p_value is not None
    assert expected_guard.effect_size_cohens_d is not None

    agent_model = RecordingFakeChatModel(
        [
            _sql_msg(
                "SELECT label, topic, mean_topic_probability, document_count, top_terms "
                "FROM gold_cohort_topic_matrix",
                "c1",
            ),
            _sql_msg(
                "SELECT label, ngram, frequency, tfidf_weight "
                "FROM gold_cohort_lexical_matrix ORDER BY tfidf_weight DESC",
                "c2",
            ),
            _sql_msg(
                "SELECT subject_id, label, night_post_ratio FROM gold_user_longitudinal_features",
                "c3",
            ),
            AIMessage(content="Gathered topic distribution, lexical bigrams, and circadian comparison."),
        ]
    )
    critic = _critic(
        "Corpus has 5000 posts (descriptive). Topic 0 covers sleep (descriptive). "
        "Top bigram 'feel alone' (descriptive). "
        "Night posting ratio differed significantly across cohorts "
        f"(depressed=35, control=35, p = {expected_guard.p_value:.6g}, "
        f"Cohen's d = {expected_guard.effect_size_cohens_d:.6g})."
    )
    sql_calls: list[str] = []
    manifest_calls, counting_manifest = _counting_manifest()

    def fake_sql(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        sql_calls.append(query)
        if "gold_cohort_topic_matrix" in query:
            return [
                {
                    "label": 1,
                    "topic": 0,
                    "mean_topic_probability": 0.41,
                    "document_count": 50,
                    "top_terms": "cant sleep, alone",
                }
            ]
        if "gold_cohort_lexical_matrix" in query:
            return [
                {"label": 1, "ngram": "feel alone", "frequency": 42, "tfidf_weight": 0.14},
                {"label": 0, "ngram": "weekend hike", "frequency": 35, "tfidf_weight": 0.11},
            ]
        return cohort_rows

    graph = build_agent_graph(
        settings=settings,
        router_model=agent_model,
        critic_model=critic,
        sql_executor=fake_sql,
        manifest_loader=counting_manifest,
    )
    result = graph.invoke(make_initial_state("Give me a picture of how the two cohorts differ."))

    assert result["status"] == "completed"
    assert [t.name for t in agent_model.bound_tools] == [
        "run_safe_sql",
        "run_spark_job",
    ]
    assert manifest_calls[0] == 1
    assert len(sql_calls) == 3
    assert result["tool_call_count"] == 3
    assert len(result["tool_results"]) == 3
    assert [r["guard_payload"]["assessment_status"] for r in result["tool_results"]] == [
        "descriptive",
        "descriptive",
        "assessed",
    ]
    assert len(critic.recorded_invocations) == 1
    assert result["critic_output_check_failed"] is None

    # Every call appears in the auditable trace with name, arguments, rows_returned, guard_payload.
    trace = result["tool_trace"]
    assert len(trace) == 3
    assert trace[0]["name"] == "run_safe_sql"
    assert "gold_cohort_topic_matrix" in trace[0]["arguments"]["query"]
    assert trace[0]["rows_returned"] == 1
    assert trace[0]["guard_payload"]["assessment_status"] == "descriptive"
    assert trace[1]["name"] == "run_safe_sql"
    assert "gold_cohort_lexical_matrix" in trace[1]["arguments"]["query"]
    assert trace[1]["rows_returned"] == 2
    assert trace[2]["name"] == "run_safe_sql"
    assert trace[2]["rows_returned"] == 70
    assert trace[2]["guard_payload"]["assessment_status"] == "assessed"

    # System prompt embeds the precomputed manifest digest without calling a manifest tool.
    first_system_msg = str(agent_model.recorded_invocations[0][0].content)
    assert "Precomputed Dataset Catalog" in first_system_msg
    assert "5000" in first_system_msg
    assert "read_dataset_manifest" not in first_system_msg

    # Critic received all three results and their guard verdicts.
    critic_user_payload = json.loads(
        str(critic.recorded_invocations[0][1].content).split("\n", 1)[1]
    )
    assert critic_user_payload["result_count"] == 3
    assert len(critic_user_payload["results"]) == 3
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in str(critic.recorded_invocations[0][0].content)


def test_manifest_digest_answers_catalog_without_tools_and_handles_missing_manifest() -> None:
    settings = Settings(_env_file=None)
    digest = format_manifest_digest(_make_manifest())
    assert count_tokens(digest) < 100
    assert "5000" in digest
    assert "depressed=1: 2600" in digest
    assert "control=0: 2400" in digest
    assert "100" in digest
    assert "2015-01-01T00:00:00" in digest
    assert "2016-12-31T23:59:59" in digest
    assert "12" in digest
    assert "not instructions" in digest.lower()

    # Case 1: Present manifest -> agent answers catalog question directly with zero tools/Spark/critic.
    manifest_calls, counting_manifest = _counting_manifest()

    agent_model = RecordingFakeChatModel(
        [AIMessage(content="The dataset has 5000 posts across 100 subjects (50 depressed, 50 control).")]
    )
    critic = _critic("Critic should not run.")
    sql_calls: list[str] = []

    res = build_agent_graph(
        settings=settings,
        router_model=agent_model,
        critic_model=critic,
        sql_executor=lambda q, _cfg: sql_calls.append(q) or [],
        manifest_loader=counting_manifest,
    ).invoke(make_initial_state("How many posts and subjects are in the dataset?"))

    assert manifest_calls[0] == 1
    assert sql_calls == []
    assert res["status"] == "completed"
    assert res["tool_call_count"] == 0
    assert res["tool_trace"] == []
    assert critic.recorded_invocations == []
    assert "5000" in res["final_response"]
    system_prompt = str(agent_model.recorded_invocations[0][0].content)
    assert "Precomputed Dataset Catalog" in system_prompt
    assert "no query needed" in system_prompt.lower()

    # Case 2: Missing manifest -> one-line fallback in prompt, SQL tools answer, deleted tool rejected.
    def missing_manifest() -> DatasetManifest:
        raise FileNotFoundError("gold_dataset_manifest.json not found")

    fallback_agent = RecordingFakeChatModel(
        [
            _tool_msg("read_dataset_manifest", {}, "legacy1"),
            _tool_msg("run_safe_sql", {"query": "SELECT COUNT(*) AS total_posts FROM silver_posts WHERE year = 2015"}, "q1"),
            AIMessage(content="Counted 2500 posts in 2015 via SQL."),
        ]
    )
    fallback_res = build_agent_graph(
        settings=settings,
        router_model=fallback_agent,
        critic_model=_critic("Sono presenti 2500 post nel 2015 (statistica descrittiva)."),
        sql_executor=lambda _q, _cfg: [{"total_posts": 2500}],
        manifest_loader=missing_manifest,
    ).invoke(make_initial_state("How many posts in 2015?"))

    assert fallback_res["status"] == "completed"
    assert fallback_res["tool_trace"][0]["status"] == "error"
    assert fallback_res["tool_trace"][0]["error"]["code"] == "UNKNOWN_TOOL"
    assert fallback_res["tool_trace"][1]["status"] == "executed"
    missing_sys_prompt = str(fallback_agent.recorded_invocations[0][0].content)
    assert "Manifest unavailable" in missing_sys_prompt


def _budget_responses(n_calls: int, final_text: str | None = None) -> list[AIMessage]:
    msgs = [
        _sql_msg(
            f"SELECT COUNT(*) AS c FROM gold_user_longitudinal_features WHERE post_count > {i}",
            f"call_{i}",
        )
        for i in range(n_calls)
    ]
    if final_text is not None:
        msgs.append(AIMessage(content=final_text))
    return msgs


def test_budget_stops_21st_tool_call_and_synthesizes_or_fails_transparently() -> None:
    settings = Settings(_env_file=None, max_tool_calls=20, min_subjects_per_class=30)

    # Case 1: Model asks for 21 tool calls, then obeys the budget-exhausted instruction.
    executed_queries: list[str] = []
    res_obey = build_agent_graph(
        settings=settings,
        router_model=RecordingFakeChatModel(
            _budget_responses(21, "Summarizing the 20 queries gathered within budget.")
        ),
        critic_model=_critic(
            "Sintesi basata sulle 20 query eseguite (statistiche descrittive; budget raggiunto)."
        ),
        sql_executor=lambda q, _cfg: executed_queries.append(q) or [{"c": len(executed_queries)}],
    ).invoke(make_initial_state("Explore all thresholds."))

    assert len(executed_queries) == 20
    assert res_obey["tool_call_count"] == 20
    assert res_obey["budget_exhausted"] is True
    assert res_obey["status"] == "completed"
    assert "budget" in res_obey["final_response"].lower()
    assert len(res_obey["tool_trace"]) == 21
    assert res_obey["tool_trace"][-1]["status"] == "budget_exhausted"
    assert res_obey["tool_trace"][-1]["rows_returned"] == 0

    # Case 2: Model still asks for a tool after being told budget is exhausted -> transparent failure.
    executed_2: list[str] = []
    res_fail = build_agent_graph(
        settings=settings,
        router_model=RecordingFakeChatModel(_budget_responses(22)),
        critic_model=_critic("unused"),
        sql_executor=lambda q, _cfg: executed_2.append(q) or [{"c": 1}],
    ).invoke(make_initial_state("Keep querying forever."))

    assert len(executed_2) == 20
    assert res_fail["status"] == "failed"
    assert "budget" in res_fail["final_response"].lower()
    assert "max_tool_calls=20" in res_fail["final_response"]


def test_consecutive_identical_tool_call_cap_fires_on_third_repeat() -> None:
    settings = Settings(_env_file=None, max_tool_calls=20)
    repeated_query = "SELECT COUNT(*) AS n FROM gold_user_longitudinal_features"

    # 3 identical calls, then model accepts the cap message and pivots to final answer.
    agent_model = RecordingFakeChatModel(
        [
            _tool_msg("run_safe_sql", {"query": repeated_query}, "r1"),
            _tool_msg("run_safe_sql", {"query": repeated_query}, "r2"),
            _tool_msg("run_safe_sql", {"query": repeated_query}, "r3"),
            AIMessage(content="Observed 100 subjects total."),
        ]
    )
    executed: list[str] = []
    res = build_agent_graph(
        settings=settings,
        router_model=agent_model,
        critic_model=_critic("Sono presenti 100 soggetti (statistica descrittiva)."),
        sql_executor=lambda q, _cfg: executed.append(q) or [{"n": 100}],
    ).invoke(make_initial_state("How many subjects?"))

    assert len(executed) == 2
    assert res["status"] == "completed"
    assert res["tool_call_count"] == 2
    assert len(res["tool_trace"]) == 3
    assert res["tool_trace"][2]["status"] == "repeat_blocked"
    assert res["tool_trace"][2]["error"]["code"] == "CONSECUTIVE_REPEAT_CAP"
    # ToolMessage sent back to agent explained the repeat cap.
    last_tool_msg = next(
        m for m in reversed(agent_model.recorded_invocations[-1]) if isinstance(m, ToolMessage)
    )
    assert "CONSECUTIVE_REPEAT_CAP" in str(last_tool_msg.content)

    # If the model ignores the block and tries the exact same call a 4th time, loop halts cleanly.
    stuck_model = RecordingFakeChatModel(
        [_tool_msg("run_safe_sql", {"query": repeated_query}, f"s{i}") for i in range(5)]
    )
    executed_stuck: list[str] = []
    res_stuck = build_agent_graph(
        settings=settings,
        router_model=stuck_model,
        critic_model=_critic("unused"),
        sql_executor=lambda q, _cfg: executed_stuck.append(q) or [{"n": 100}],
    ).invoke(make_initial_state("Loop forever."))

    assert len(executed_stuck) == 2
    assert res_stuck["status"] == "failed"
    assert "repeat cap" in res_stuck["final_response"].lower()


def test_hitl_interrupt_approval_executes_once_and_refusal_feeds_model_without_executing(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    expensive_sql = "SELECT subject_id, label, text FROM silver_posts"

    # 1. Approval path across SQLite checkpointer reopen.
    db_approve = tmp_path / "hitl_approve.sqlite"
    approve_agent = RecordingFakeChatModel(
        [
            _tool_msg("run_safe_sql", {"query": expensive_sql}, "h1"),
            AIMessage(content="Scan complete."),
        ]
    )
    approve_critic = _critic(
        "Approved query completed (depressed=0, control=0, p = 1.0, Cohen's d = 0.0)."
    )
    approve_calls: list[str] = []
    cfg_approve = {"configurable": {"thread_id": "loop-hitl-approve"}}

    with SqliteSaver.from_conn_string(str(db_approve)) as saver1:
        g1 = build_agent_graph(
            settings=settings,
            router_model=approve_agent,
            critic_model=approve_critic,
            sql_executor=lambda q, _cfg: approve_calls.append(q) or [{"subject_id": "s1", "label": 1, "text": "hi"}],
            checkpointer=saver1,
        )
        interrupted = g1.invoke(make_initial_state("Scan raw silver_posts."), config=cfg_approve)

    assert "__interrupt__" in interrupted
    assert interrupted["__interrupt__"][0].value["estimated_cost"] > 1.0
    assert "silver_posts" in interrupted["__interrupt__"][0].value["query"]
    assert approve_calls == []

    with SqliteSaver.from_conn_string(str(db_approve)) as saver2:
        g2 = build_agent_graph(
            settings=settings,
            router_model=approve_agent,
            critic_model=approve_critic,
            sql_executor=lambda q, _cfg: approve_calls.append(q) or [{"subject_id": "s1", "label": 1, "text": "hi"}],
            checkpointer=saver2,
        )
        resumed = g2.invoke(Command(resume={"approved": True}), config=cfg_approve)

    assert len(approve_calls) == 1
    assert resumed["hitl_decision"] == "approved"
    assert resumed["status"] == "completed"
    assert resumed["tool_trace"][0]["status"] == "executed"
    assert resumed["tool_trace"][0]["rows_returned"] == 1

    # 2. Refusal path: zero executions, refusal fed back to model, model pivots or explains.
    db_reject = tmp_path / "hitl_reject.sqlite"
    reject_agent = RecordingFakeChatModel(
        [
            _sql_msg(expensive_sql, "h2"),
            AIMessage(content="Execution was refused at the HITL gate; no raw posts were scanned."),
        ]
    )
    reject_calls: list[str] = []
    cfg_reject = {"configurable": {"thread_id": "loop-hitl-reject"}}

    with SqliteSaver.from_conn_string(str(db_reject)) as saver:
        g_rej = build_agent_graph(
            settings=settings,
            router_model=reject_agent,
            critic_model=_critic("unused"),
            sql_executor=lambda q, _cfg: reject_calls.append(q) or [],
            checkpointer=saver,
        )
        interrupted_rej = g_rej.invoke(make_initial_state("Dump silver_posts."), config=cfg_reject)
        assert "__interrupt__" in interrupted_rej
        assert reject_calls == []

        refused = g_rej.invoke(Command(resume={"approved": False}), config=cfg_reject)

    assert reject_calls == []
    assert refused["hitl_decision"] == "rejected"
    assert refused["status"] == "cancelled"
    assert refused["tool_trace"][0]["status"] == "hitl_rejected"
    assert refused["tool_trace"][0]["rows_returned"] == 0
    # Refusal payload was fed back to the model as a ToolMessage.
    tool_msgs = [m for m in reject_agent.recorded_invocations[-1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "HITL_REJECTED" in str(tool_msgs[0].content)


def test_spark_job_tool_hitl_and_offline_execution(tmp_path: Path) -> None:
    db_path = tmp_path / "hitl_job.sqlite"
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    agent_model = RecordingFakeChatModel(
        [
            _tool_msg("run_spark_job", {"job": "topic_model", "params": {"k": 8}}, "j1"),
            AIMessage(content="Topic model finished."),
        ]
    )
    critic = _critic("Topic 0 mostra termini descrittivi (statistiche descrittive).")
    job_calls: list[tuple[str, dict[str, Any]]] = []
    config = {"configurable": {"thread_id": "hitl-job-loop"}}

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_agent_graph(
            settings=settings,
            router_model=agent_model,
            critic_model=critic,
            job_executor=lambda name, params, _cfg: job_calls.append((name, params)) or [
                {
                    "label": 1,
                    "topic": 0,
                    "mean_topic_probability": 0.42,
                    "document_count": 40,
                    "top_terms": "feel alone, cant sleep",
                },
                {
                    "label": 0,
                    "topic": 0,
                    "mean_topic_probability": 0.11,
                    "document_count": 40,
                    "top_terms": "morning coffee, weekend hike",
                },
            ],
            checkpointer=saver,
        )
        interrupted = graph.invoke(make_initial_state("Recompute LDA topics."), config=config)
        assert "__interrupt__" in interrupted
        assert job_calls == []
        resumed = graph.invoke(Command(resume={"approved": True}), config=config)

    assert job_calls == [("topic_model", {"k": 8})]
    assert resumed["status"] == "completed"
    assert resumed["tool_results"][0]["guard_payload"]["assessment_status"] == "descriptive"


def test_drop_table_and_csv_path_return_safe_sql_error_without_executing() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    cohort_rows = _cohort_rows(30)
    expected_guard = evaluate_query_result_guard(cohort_rows, settings=settings)
    assert expected_guard.p_value is not None
    assert expected_guard.effect_size_cohens_d is not None

    agent_model = RecordingFakeChatModel(
        [
            _tool_msg("run_safe_sql", {"query": "DROP TABLE silver_posts"}, "e1"),
            _tool_msg("run_safe_sql", {"query": "SELECT * FROM csv.leaked_file"}, "e2"),
            _tool_msg(
                "run_safe_sql",
                {
                    "query": (
                        "SELECT subject_id, label, night_post_ratio "
                        "FROM gold_user_longitudinal_features"
                    )
                },
                "e3",
            ),
            AIMessage(content="Recovered to valid allowlisted Gold query."),
        ]
    )
    critic = _critic(
        "Recovered comparison "
        f"(depressed=30, control=30, p = {expected_guard.p_value:.6g}, "
        f"Cohen's d = {expected_guard.effect_size_cohens_d:.6g})."
    )
    executed: list[str] = []

    res = build_agent_graph(
        settings=settings,
        router_model=agent_model,
        critic_model=critic,
        sql_executor=lambda q, _cfg: executed.append(q) or cohort_rows,
    ).invoke(make_initial_state("Compare night post ratio."))

    assert len(executed) == 1
    assert "gold_user_longitudinal_features" in executed[0]
    assert res["status"] == "completed"
    assert res["tool_trace"][0]["status"] == "error"
    assert res["tool_trace"][0]["error"]["code"] == "DISALLOWED_STATEMENT"
    assert res["tool_trace"][1]["status"] == "error"
    assert res["tool_trace"][1]["error"]["code"] == "TABLE_NOT_ALLOWED"
    assert res["tool_trace"][2]["status"] == "executed"

    # Both SafeSQLErrors came back to the model as ToolMessages.
    final_agent_msgs = agent_model.recorded_invocations[-1]
    tool_contents = [str(m.content) for m in final_agent_msgs if isinstance(m, ToolMessage)]
    assert any("DISALLOWED_STATEMENT" in content for content in tool_contents)
    assert any("TABLE_NOT_ALLOWED" in content for content in tool_contents)


def test_significant_comparison_among_descriptive_results_requires_citations_and_survives_fallback() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    cohort_rows = _cohort_rows(35)
    sig_guard = evaluate_query_result_guard(cohort_rows, settings=settings)
    assert sig_guard.p_value is not None
    assert sig_guard.effect_size_cohens_d is not None

    def _build_mixed_agent() -> RecordingFakeChatModel:
        return RecordingFakeChatModel(
            [
                _sql_msg("SELECT COUNT(*) AS total_posts FROM silver_posts WHERE year = 2015", "m1"),
                _sql_msg("SELECT label, ngram, frequency FROM gold_cohort_lexical_matrix", "m2"),
                _sql_msg(
                    "SELECT subject_id, label, night_post_ratio FROM gold_user_longitudinal_features",
                    "m3",
                ),
                AIMessage(content="Mixed descriptive + significant results ready."),
            ]
        )

    # 1. Prose omits every number; deterministic appendix supplies n, p, d and passes in 1 call.
    prose_no_numbers = (
        "Catalog has 5000 posts (descriptive). Top bigram 'feel alone' (descriptive). "
        "Night post ratio differed significantly across cohorts."
    )
    critic_ok = _critic(prose_no_numbers)
    res_with_appendix = build_agent_graph(
        settings=settings,
        router_model=_build_mixed_agent(),
        critic_model=critic_ok,
        sql_executor=_mixed_sql(cohort_rows),
        manifest_loader=_make_manifest,
    ).invoke(make_initial_state("How do the cohorts differ?"))

    assert len(critic_ok.recorded_invocations) == 1
    assert res_with_appendix["critic_output_check_failed"] is None
    final_with_appendix = res_with_appendix["final_response"]
    assert final_with_appendix.startswith(prose_no_numbers)
    assert "## Appendice statistica (generata dal sistema)" in final_with_appendix
    assert "35" in final_with_appendix
    assert f"{sig_guard.p_value:.6g}" in final_with_appendix
    assert f"{sig_guard.effect_size_cohens_d:.6g}" in final_with_appendix
    assert "significativo" in final_with_appendix
    assert "descrittivo" in final_with_appendix
    mixed_guards = [
        StatisticalGuardResult.model_validate(entry["guard_payload"])
        for entry in res_with_appendix["tool_results"]
    ]
    assert validate_critic_output(final_with_appendix, mixed_guards).passed is True
    # Rows appear in tool-call order.
    assert (
        final_with_appendix.index("`total_posts`")
        < final_with_appendix.index("`lexical_ngram`")
        < final_with_appendix.index("`night_post_ratio`")
    )

    # 2. Genuine claim violation (asserting significance on a non-significant result twice)
    # still triggers retry and reaches the readable deterministic fallback.
    ns_rows = _cohort_rows(35, dep_base=0.20, ctrl_base=0.20)
    rogue_claim = "Depressed users post at night significantly more often than controls."
    critic_fallback = _critic(rogue_claim, rogue_claim)
    res_fallback = build_agent_graph(
        settings=settings,
        router_model=_build_mixed_agent(),
        critic_model=critic_fallback,
        sql_executor=_mixed_sql(ns_rows),
        manifest_loader=_make_manifest,
    ).invoke(make_initial_state("How do the cohorts differ?"))

    assert len(critic_fallback.recorded_invocations) == 2
    assert res_fallback["critic_output_check_failed"] is not None
    assert res_fallback["critic_output_check_failed"]["forbidden_claims"]
    fallback_text = res_fallback["final_response"]
    assert "Risultato 1" in fallback_text
    assert "Risultato 2" in fallback_text
    assert "Risultato 3" in fallback_text
    assert "risultato non conclusivo" in fallback_text.lower()
    assert "35" in fallback_text

    guards = [
        StatisticalGuardResult.model_validate(entry["guard_payload"])
        for entry in res_fallback["tool_results"]
    ]
    assert validate_critic_output(fallback_text, guards).passed is True


def test_statistical_appendix_is_deterministic_and_omitted_for_all_descriptive() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    sig_guard = evaluate_query_result_guard(_cohort_rows(35), settings=settings)
    desc_guard = StatisticalGuardResult(
        sample_size_depressed=0,
        sample_size_control=0,
        aggregation_method="none",
        assessment_status="descriptive",
        descriptive_reason="descriptive: lexical_frequency_table",
        statistical_confidence="descriptive",
        test_used="none",
        metric_name="lexical_ngram",
    )

    first = format_statistical_appendix([desc_guard, sig_guard])
    second = format_statistical_appendix([desc_guard, sig_guard])
    assert first == second
    assert "## Appendice statistica (generata dal sistema)" in first
    assert format_statistical_appendix([desc_guard]) == ""

    # End-to-end all-descriptive graph run never appends the statistical appendix.
    agent_model = RecordingFakeChatModel(
        [
            _tool_msg(
                "run_safe_sql",
                {"query": "SELECT label, topic, top_terms FROM gold_cohort_topic_matrix"},
                "d1",
            ),
            _tool_msg(
                "run_safe_sql",
                {"query": "SELECT label, ngram, frequency FROM gold_cohort_lexical_matrix"},
                "d2",
            ),
            AIMessage(content="Descriptive results ready."),
        ]
    )
    prose = "Il dataset contiene 5000 post e i bigrammi principali sono elencati a titolo descrittivo."
    desc_critic = _critic(prose)
    res = build_agent_graph(
        settings=settings,
        router_model=agent_model,
        critic_model=desc_critic,
        sql_executor=lambda q, _cfg: (
            [{"label": 1, "topic": 0, "top_terms": "feel alone"}]
            if "gold_cohort_topic_matrix" in q
            else [{"label": 1, "ngram": "feel alone", "frequency": 42}]
        ),
        manifest_loader=_make_manifest,
    ).invoke(make_initial_state("Mostrami i topic e i bigrammi principali."))

    assert res["status"] == "completed"
    assert len(desc_critic.recorded_invocations) == 1
    assert res["final_response"] == prose
    assert "Appendice statistica" not in res["final_response"]


def test_prose_claiming_significance_when_appendix_is_nonsignificant_still_fails() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    ns_rows = _cohort_rows(35, dep_base=0.25, ctrl_base=0.25, mod=4)
    agent_model = RecordingFakeChatModel(
        [
            _sql_msg(
                "SELECT subject_id, label, night_post_ratio FROM gold_user_longitudinal_features",
                "ns1",
            ),
            AIMessage(content="Comparison ready."),
        ]
    )
    # Attempt 1 asserts a significant difference (contradicts appendix's "non significativo");
    # Attempt 2 drops the forbidden claim and omits numbers, passing with the appendix.
    retry_critic = _critic(
        "Depressed users post significantly more at night.",
        "Non si osserva alcuna differenza statisticamente significativa tra le coorti.",
    )
    res = build_agent_graph(
        settings=settings,
        router_model=agent_model,
        critic_model=retry_critic,
        sql_executor=lambda _q, _cfg: ns_rows,
    ).invoke(make_initial_state("I soggetti depressi postano più di notte?"))

    assert res["status"] == "completed"
    assert len(retry_critic.recorded_invocations) == 2
    assert res["critic_output_check_failed"] is None
    assert "## Appendice statistica (generata dal sistema)" in res["final_response"]
    assert "non significativo" in res["final_response"]


def test_cannot_assess_and_underpowered_never_ship_significance_claims() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)

    # cannot_assess (missing subject_id)
    ca_agent = RecordingFakeChatModel(
        [
            _tool_msg(
                "run_safe_sql",
                {"query": "SELECT label, night_post_ratio FROM gold_user_longitudinal_features"},
                "ca1",
            ),
            AIMessage(content="Done."),
        ]
    )
    rogue_ca = "There is a statistically significant difference between depressed and control users."
    ca_res = build_agent_graph(
        settings=settings,
        router_model=ca_agent,
        critic_model=_critic(rogue_ca, rogue_ca),
        sql_executor=lambda _q, _cfg: [
            {"label": 1, "night_post_ratio": 0.6},
            {"label": 0, "night_post_ratio": 0.1},
        ],
    ).invoke(make_initial_state("Confronta i post notturni per coorte."))

    assert ca_res["statistical_validation"]["assessment_status"] == "cannot_assess"
    assert ca_res["critic_output_check_failed"] is not None
    assert "impossibile valutare" in ca_res["final_response"].lower()
    assert "p_value =" not in ca_res["final_response"]

    # underpowered (pseudo-replicated single subject)
    up_agent = RecordingFakeChatModel(
        [
            _sql_msg(
                "SELECT subject_id, label, night_post_ratio FROM gold_user_longitudinal_features",
                "up1",
            ),
            AIMessage(content="Done."),
        ]
    )
    rogue_up = (
        "Depressed users post at night significantly more often "
        "(depressed=1, control=35, p = 1.0, Cohen's d = 0.0)."
    )
    up_rows = _cohort_rows(40, 35, dep_base=0.85, ctrl_base=0.10, single_dep_subject=True)
    up_res = build_agent_graph(
        settings=settings,
        router_model=up_agent,
        critic_model=_critic(rogue_up, rogue_up),
        sql_executor=lambda _q, _cfg: up_rows,
    ).invoke(make_initial_state("Compare night post ratio."))

    assert up_res["statistical_validation"]["is_underpowered"] is True
    assert up_res["statistical_validation"]["is_significant"] is False
    assert up_res["critic_output_check_failed"] is not None
    assert "risultato non conclusivo" in up_res["final_response"].lower()


def test_followup_turn_receives_prior_conversation_without_tool_noise(tmp_path: Path) -> None:
    db_path = tmp_path / "followup.sqlite"
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    agent_model = RecordingFakeChatModel(
        [
            _tool_msg(
                "run_safe_sql",
                {"query": "SELECT COUNT(*) AS post_count FROM silver_posts WHERE year = 2014"},
                "t1",
            ),
            AIMessage(content="Counted 2014 posts."),
            _tool_msg(
                "run_safe_sql",
                {"query": "SELECT COUNT(*) AS post_count FROM silver_posts WHERE year = 2015"},
                "t2",
            ),
            AIMessage(content="Counted 2015 posts."),
        ]
    )
    critic = _critic(
        "Nel 2014 sono presenti 1250 post.",
        "Nel 2015 sono presenti 980 post.",
    )
    config = {"configurable": {"thread_id": "followup-loop-thread"}}

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_agent_graph(
            settings=settings,
            router_model=agent_model,
            critic_model=critic,
            sql_executor=lambda q, _cfg: [{"post_count": 1250 if "2014" in q else 980}],
            checkpointer=saver,
        )
        r1 = graph.invoke(make_initial_state("Quanti post ci sono nel 2014?"), config=config)
        r2 = graph.invoke(make_initial_state("e per il 2015?"), config=config)

    assert r1["status"] == "completed"
    assert r2["status"] == "completed"
    # Turn 2 initial invocation includes Turn 1 question & final critic reply, without Turn 1 ToolMessages.
    turn2_msgs = agent_model.recorded_invocations[2]
    assert any("Quanti post ci sono nel 2014?" in str(m.content) for m in turn2_msgs)
    assert any("Nel 2014 sono presenti 1250 post." in str(m.content) for m in turn2_msgs)
    assert not any(isinstance(m, ToolMessage) for m in turn2_msgs)


@pytest.mark.spark
def test_cost_estimation_registers_lakehouse_views_before_explaining(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    settings, paths, _ = lakehouse_env
    rows = [(f"subject{index:03d}", index % 2, 10 + index, 0.25) for index in range(6)]
    spark.createDataFrame(
        rows, "subject_id STRING, label INT, post_count INT, night_post_ratio DOUBLE"
    ).write.mode("overwrite").parquet(str(paths.gold_user_longitudinal_features))

    estimate = estimate_sql_cost(
        "SELECT subject_id, label, night_post_ratio FROM gold_user_longitudinal_features",
        settings=settings,
        spark=spark,
    )

    assert estimate["method"] == "spark_explain"
    assert estimate["estimated_cost"] >= 0.0


def _run_spark_cohort_query(
    spark: SparkSession,
    lakehouse_env: tuple,
    *,
    rows: list[tuple[str, int, float]],
    query: str,
    critic_text: str,
    min_subjects_per_class: int,
    sql_fetch_limit: int,
) -> tuple[dict[str, Any], RecordingFakeChatModel]:
    base_settings, paths, _ = lakehouse_env
    settings = Settings(
        _env_file=None,
        spark_storage_uri=base_settings.spark_storage_uri,
        min_subjects_per_class=min_subjects_per_class,
        sql_row_limit=500,
        sql_fetch_limit=sql_fetch_limit,
    )
    spark.createDataFrame(
        rows,
        "subject_id STRING, label INT, night_post_ratio DOUBLE",
    ).write.mode("overwrite").parquet(str(paths.gold_user_longitudinal_features))

    critic = _critic(critic_text)
    result = build_agent_graph(
        settings=settings,
        router_model=RecordingFakeChatModel(
            [_sql_msg(query, "sp1"), AIMessage(content="Cohort query complete.")]
        ),
        critic_model=critic,
        sql_executor=make_spark_sql_executor(spark, settings=settings, paths=paths),
        spark=spark,
    ).invoke(make_initial_state("Compare night_post_ratio across cohorts."))
    return result, critic


@pytest.mark.spark
def test_full_cohort_819_subjects_analyzed_while_state_rows_capped_at_500(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    synthetic_rows = [
        (f"dep_{i:04d}", 1, 0.55 + (i % 7) * 0.02) for i in range(100)
    ] + [
        (f"ctrl_{i:04d}", 0, 0.15 + (i % 7) * 0.02) for i in range(719)
    ]
    result, critic = _run_spark_cohort_query(
        spark,
        lakehouse_env,
        rows=synthetic_rows,
        query="SELECT subject_id, label, night_post_ratio FROM gold_user_longitudinal_features",
        critic_text=(
            "Full cohort comparison across depressed=100 and control=719 subjects "
            "(p = 5.84328e-60, Cohen's d = 9.96672)."
        ),
        min_subjects_per_class=30,
        sql_fetch_limit=50_000,
    )

    assert result["status"] == "completed"
    assert len(critic.recorded_invocations) == 1
    assert result["critic_output_check_failed"] is None
    assert "LIMIT 50000" in result["generated_query"].upper()

    guard = result["statistical_validation"]
    assert guard["sample_size_depressed"] == 100
    assert guard["sample_size_control"] == 719

    qr = result["query_result"]
    assert qr["rows_returned"] == 819
    assert qr["rows_shown"] == 500
    assert len(qr["rows"]) == 500
    assert qr["truncated"] is False


def test_multi_tool_call_batch_with_hitl_does_not_reexecute_prior_tool(tmp_path: Path) -> None:
    db_path = tmp_path / "batch_hitl.sqlite"
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    cheap_sql = "SELECT COUNT(*) AS n FROM gold_user_longitudinal_features"
    expensive_sql = "SELECT subject_id, label, text FROM silver_posts"

    agent_model = RecordingFakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "run_safe_sql", "args": {"query": cheap_sql}, "id": "b1"},
                    {"name": "run_safe_sql", "args": {"query": expensive_sql}, "id": "b2"},
                ],
            ),
            AIMessage(content="Both queries completed."),
        ]
    )
    critic = _critic("Risultati descrittivi raccolti (statistiche descrittive).")
    executed: list[str] = []
    config = {"configurable": {"thread_id": "batch-hitl-thread"}}

    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_agent_graph(
            settings=settings,
            router_model=agent_model,
            critic_model=critic,
            sql_executor=lambda q, _cfg: executed.append(q) or [{"n": 100}],
            checkpointer=saver,
        )
        interrupted = graph.invoke(make_initial_state("Run cheap then expensive."), config=config)
        assert "__interrupt__" in interrupted
        assert len(executed) == 1
        assert "gold_user_longitudinal_features" in executed[0]

        resumed = graph.invoke(Command(resume={"approved": True}), config=config)

    assert len(executed) == 2
    assert "gold_user_longitudinal_features" in executed[0]
    assert "silver_posts" in executed[1]
    assert resumed["status"] == "completed"
    assert len(resumed["tool_trace"]) == 2


@pytest.mark.spark
def test_fetch_limit_cap_marks_truncated_and_demands_critic_disclosure(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    synthetic_rows = [(f"s_{i:03d}", i % 2, 0.2 + (i % 5) * 0.05) for i in range(60)]
    result, critic = _run_spark_cohort_query(
        spark,
        lakehouse_env,
        rows=synthetic_rows,
        query=(
            "SELECT subject_id, label, night_post_ratio "
            "FROM gold_user_longitudinal_features ORDER BY subject_id"
        ),
        critic_text=(
            "Capped sample analysis (truncated at 20 rows): "
            "depressed=10, control=10, p = 1.0, Cohen's d = 0.0."
        ),
        min_subjects_per_class=5,
        sql_fetch_limit=20,
    )

    assert result["status"] == "completed"
    assert result["query_result"]["truncated"] is True
    assert result["statistical_validation"]["truncated"] is True
    critic_sys = str(critic.recorded_invocations[0][0].content)
    assert "truncated=true" in critic_sys.lower()
    assert "sql_fetch_limit=20" in critic_sys
