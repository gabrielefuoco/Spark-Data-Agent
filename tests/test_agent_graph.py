"""Offline tests for the cyclic LangGraph Data Agent (fake LLMs + fake executor)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import Command
from pyspark.sql import SparkSession
from src.agent.cost import estimate_sql_cost
from src.agent.critic_check import validate_critic_output
from src.agent.graph import build_agent_graph, open_sqlite_checkpointer
from src.agent.guards import CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS, StatisticalGuardResult
from src.agent.nodes import make_spark_sql_executor
from src.agent.state import make_initial_state
from src.analytics.safe_sql import SafeSQLError
from src.config import Settings
from src.lakehouse.manifest import DatasetManifest


class RecordingFakeChatModel(FakeMessagesListChatModel):
    """Fake chat model that captures every prompt message sequence for assertions."""

    recorded_invocations: list[list[BaseMessage]] = []

    def __init__(self, responses: list[BaseMessage]) -> None:
        super().__init__(responses=responses)
        self.recorded_invocations = []

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.recorded_invocations.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


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


def _cohort_rows(n_per_class: int, *, depressed_offset: float = 0.25) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(n_per_class):
        rows.append(
            {
                "subject_id": f"dep_{i}",
                "label": 1,
                "night_post_ratio": 0.30 + depressed_offset + (i % 5) * 0.02,
            }
        )
        rows.append(
            {
                "subject_id": f"ctrl_{i}",
                "label": 0,
                "night_post_ratio": 0.10 + (i % 5) * 0.02,
            }
        )
    return rows


def _router(*payloads: dict[str, Any]) -> RecordingFakeChatModel:
    return RecordingFakeChatModel([AIMessage(content=json.dumps(p)) for p in payloads])


def _critic(*texts: str) -> RecordingFakeChatModel:
    return RecordingFakeChatModel([AIMessage(content=t) for t in texts])


def test_manifest_only_question_never_invokes_sql_executor() -> None:
    settings = Settings(_env_file=None)
    router = _router(
        {
            "user_intent": "catalog_metadata",
            "answer_from_manifest": True,
            "sql_query": None,
            "metric_name": None,
            "direct_response": None,
        }
    )
    critic = _critic("Critic must not be called for manifest queries.")
    executor_calls: list[str] = []

    def fake_executor(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        executor_calls.append(query)
        return []

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
        manifest_loader=_make_manifest,
    )

    result = graph.invoke(make_initial_state("How many posts and subjects are in the dataset?"))

    assert executor_calls == []
    assert result["status"] == "completed"
    assert result["user_intent"] == "catalog_metadata"
    assert "5000" in result["final_response"]
    assert "100" in result["final_response"]
    assert "quarantined" in result["final_response"].lower()


def test_happy_path_validates_executes_guards_and_wires_critic_prompt() -> None:
    settings = Settings(
        _env_file=None, min_subjects_per_class=30, sql_row_limit=100, sql_fetch_limit=200
    )
    router = _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT subject_id, label, night_post_ratio "
                "FROM gold_user_longitudinal_features"
            ),
            "metric_name": "night_post_ratio",
            "direct_response": None,
        }
    )
    critic = _critic(
        "Significant difference observed "
        "(sample_size_depressed=35, sample_size_control=35, "
        "p = 5.04483e-13, Cohen's d = 15.681)."
    )
    executor_calls: list[str] = []

    def fake_executor(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        executor_calls.append(query)
        return _cohort_rows(35)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
    )

    result = graph.invoke(
        make_initial_state("Compare night posting ratio between depressed and control users.")
    )

    assert len(executor_calls) == 1
    assert "LIMIT 200" in executor_calls[0].upper()
    assert result["status"] == "completed"
    assert result["sql_cost_estimate"]["triggers_hitl"] is False
    assert result["statistical_validation"]["sample_size_depressed"] == 35
    assert result["statistical_validation"]["sample_size_control"] == 35
    assert result["statistical_validation"]["observation_count_depressed"] == 35
    assert result["statistical_validation"]["observation_count_control"] == 35
    assert result["statistical_validation"]["aggregation_method"] == "subject_mean"
    assert result["statistical_validation"]["assessment_status"] == "assessed"
    assert result["statistical_validation"]["is_significant"] is True
    assert result["statistical_validation"]["is_underpowered"] is False

    # Verify the critic prompt receives the mandatory guardrail instructions and verdict.
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in result["critic_system_prompt"]
    assert "ALLOWED:" in result["critic_system_prompt"]
    assert len(critic.recorded_invocations) == 1
    critic_system_msg = critic.recorded_invocations[0][0].content
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in critic_system_msg
    assert "sample_size_depressed: 35" in critic_system_msg


def test_ast_validation_rejection_triggers_retry_without_touching_executor() -> None:
    settings = Settings(_env_file=None, max_query_retries=2)
    router = _router(
        {
            "user_intent": "ad_hoc_sql",
            "answer_from_manifest": False,
            "sql_query": "DROP TABLE silver_posts",
        },
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT subject_id, label, night_post_ratio "
                "FROM gold_user_longitudinal_features"
            ),
            "metric_name": "night_post_ratio",
        },
    )
    critic = _critic(
        "Recovered after AST fix "
        "(depressed=30, control=30, p = 2.41976e-11, Cohen's d = 15.6425)."
    )
    executor_calls: list[str] = []

    def fake_executor(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        executor_calls.append(query)
        return _cohort_rows(30)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
    )
    result = graph.invoke(make_initial_state("Analyze night post ratio."))

    assert len(executor_calls) == 1
    assert result["retry_count"] == 1
    assert result["status"] == "completed"
    assert "DISALLOWED_STATEMENT" in str(router.recorded_invocations[1][1].content)


def test_retry_loop_stops_after_max_query_retries_on_safe_sql_error() -> None:
    settings = Settings(_env_file=None, max_query_retries=2)
    router = _router(
        {
            "user_intent": "ad_hoc_sql",
            "answer_from_manifest": False,
            "sql_query": "SELECT bad_col FROM gold_user_longitudinal_features",
        },
        {
            "user_intent": "ad_hoc_sql",
            "answer_from_manifest": False,
            "sql_query": "SELECT bad_col_2 FROM gold_user_longitudinal_features",
        },
        {
            "user_intent": "ad_hoc_sql",
            "answer_from_manifest": False,
            "sql_query": "SELECT bad_col_3 FROM gold_user_longitudinal_features",
        },
    )
    critic = _critic("unused")
    executor_calls: list[str] = []

    def failing_executor(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        executor_calls.append(query)
        raise SafeSQLError("SPARK_ANALYSIS_ERROR", "UNRESOLVED_COLUMN: bad_col", query=query)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=failing_executor,
    )
    result = graph.invoke(make_initial_state("Show bad_col across users."))

    # 1 initial attempt + 2 retries = 3 total attempts; planner re-entered twice.
    assert len(executor_calls) == 1 + settings.max_query_retries == 3
    assert len(router.recorded_invocations) == 3
    assert len(critic.recorded_invocations) == 0
    assert result["retry_count"] == 2
    assert result["status"] == "failed"
    assert "SPARK_ANALYSIS_ERROR" in result["final_response"]
    assert "3 attempt(s)" in result["final_response"]
    # Structured error was fed back to planner on retry.
    assert "SPARK_ANALYSIS_ERROR" in str(router.recorded_invocations[1][1].content)


def test_successful_retry_on_second_attempt_stops_loop() -> None:
    settings = Settings(_env_file=None, max_query_retries=2)
    router = _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": "SELECT wrong_col FROM gold_user_longitudinal_features",
        },
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT subject_id, label, night_post_ratio "
                "FROM gold_user_longitudinal_features"
            ),
            "metric_name": "night_post_ratio",
        },
    )
    completed_critique = (
        "Completed on retry (depressed=30, control=30, p = 2.41976e-11, Cohen's d = 15.6425)."
    )
    critic = _critic(completed_critique)
    attempts: list[str] = []

    def flaky_executor(query: str, _cfg: Settings) -> Sequence[dict[str, Any]]:
        attempts.append(query)
        if len(attempts) == 1:
            raise SafeSQLError("SPARK_ANALYSIS_ERROR", "Column wrong_col not found", query=query)
        return _cohort_rows(30)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=flaky_executor,  # type: ignore[arg-type]
    )
    result = graph.invoke(make_initial_state("Analyze night posting ratio."))

    assert len(attempts) == 2
    assert result["retry_count"] == 1
    assert result["status"] == "completed"
    assert result["final_response"] == completed_critique


def test_empty_result_triggers_retry_and_recovers() -> None:
    settings = Settings(_env_file=None, max_query_retries=2)
    router = _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT subject_id, label, night_post_ratio "
                "FROM gold_user_longitudinal_features WHERE label = 99"
            ),
        },
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT subject_id, label, night_post_ratio "
                "FROM gold_user_longitudinal_features"
            ),
            "metric_name": "night_post_ratio",
        },
    )
    critic = _critic(
        "Recovered after empty result "
        "(depressed=30, control=30, p = 2.41976e-11, Cohen's d = 15.6425)."
    )
    attempts: list[str] = []

    def executor_empty_then_valid(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        attempts.append(query)
        if len(attempts) == 1:
            return []
        return _cohort_rows(30)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=executor_empty_then_valid,
    )
    result = graph.invoke(make_initial_state("Compare night post ratio."))

    assert len(attempts) == 2
    assert result["retry_count"] == 1
    assert result["status"] == "completed"
    assert "EMPTY_RESULT" in str(router.recorded_invocations[1][1].content)


def test_hitl_interrupt_persists_across_sqlite_checkpointer_and_executes_on_approval(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "hitl_checkpoints.sqlite"
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    expensive_sql = "SELECT subject_id, label, text FROM silver_posts"

    router = _router(
        {
            "user_intent": "ad_hoc_sql",
            "answer_from_manifest": False,
            "sql_query": expensive_sql,
        }
    )
    approved_critique = (
        "Approved expensive query completed "
        "(sample_size_depressed=0, sample_size_control=0, p = 1.0, Cohen's d = 0.0)."
    )
    critic = _critic(approved_critique)
    executor_calls: list[str] = []

    def fake_executor(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        executor_calls.append(query)
        return [{"subject_id": "s1", "label": 1, "text": "example post"}]

    config = {"configurable": {"thread_id": "hitl-thread-approve"}}

    with open_sqlite_checkpointer(db_path) as saver1:
        graph1 = build_agent_graph(
            settings=settings,
            router_model=router,
            critic_model=critic,
            sql_executor=fake_executor,
            checkpointer=saver1,
        )
        interrupted = graph1.invoke(make_initial_state("Scan raw silver posts."), config=config)

    assert "__interrupt__" in interrupted
    assert len(interrupted["__interrupt__"]) == 1
    assert interrupted["__interrupt__"][0].value["estimated_cost"] > 1.0
    assert executor_calls == []

    # Re-open SQLite checkpointer and recompile the graph to verify checkpoint survival.
    with open_sqlite_checkpointer(db_path) as saver2:
        graph2 = build_agent_graph(
            settings=settings,
            router_model=router,
            critic_model=critic,
            sql_executor=fake_executor,
            checkpointer=saver2,
        )
        snapshot = graph2.get_state(config)
        assert snapshot.next == ("hitl_gate",)
        assert "silver_posts" in snapshot.values["generated_query"]

        resumed = graph2.invoke(Command(resume={"approved": True}), config=config)

    assert len(executor_calls) == 1
    assert resumed["hitl_decision"] == "approved"
    assert resumed["status"] == "completed"
    assert resumed["final_response"] == approved_critique


def test_hitl_refusal_never_executes_and_returns_cancelled_state(tmp_path: Path) -> None:
    db_path = tmp_path / "hitl_reject.sqlite"
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    router = _router(
        {
            "user_intent": "ad_hoc_sql",
            "answer_from_manifest": False,
            "sql_query": "SELECT * FROM silver_posts",
        }
    )
    critic = _critic("Should not run")
    executor_calls: list[str] = []

    def fake_executor(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        executor_calls.append(query)
        return []

    config = {"configurable": {"thread_id": "hitl-thread-reject"}}

    with open_sqlite_checkpointer(db_path) as saver:
        graph = build_agent_graph(
            settings=settings,
            router_model=router,
            critic_model=critic,
            sql_executor=fake_executor,
            checkpointer=saver,
        )
        interrupted = graph.invoke(make_initial_state("Dump all silver posts."), config=config)
        assert "__interrupt__" in interrupted
        assert executor_calls == []

        cancelled = graph.invoke(Command(resume={"approved": False}), config=config)

    assert executor_calls == []
    assert cancelled["hitl_decision"] == "rejected"
    assert cancelled["status"] == "cancelled"
    assert "cancelled" in cancelled["final_response"].lower()


def test_missing_subject_id_column_records_cannot_assess_verdict_and_forbids_claims() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    router = _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": "SELECT label, night_post_ratio FROM gold_user_longitudinal_features",
            "metric_name": "night_post_ratio",
        }
    )
    critic = _critic(
        "Impossibile valutare la significatività statistica senza dimensione subject_id."
    )

    def fake_executor(_query: str, _cfg: Settings) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in range(40):
            rows.append({"label": 1, "night_post_ratio": 0.60 + (i % 5) * 0.02})
            rows.append({"label": 0, "night_post_ratio": 0.10 + (i % 5) * 0.02})
        return rows

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
    )
    result = graph.invoke(make_initial_state("Compare night post ratio by cohort."))

    assert result["status"] == "completed"
    guard = result["statistical_validation"]
    assert guard["assessment_status"] == "cannot_assess"
    assert guard["p_value"] is None
    assert guard["effect_size_cohens_d"] is None
    assert guard["cannot_assess_reason"] is not None
    assert "cannot assess: no subject dimension" in guard["cannot_assess_reason"].lower()
    assert guard["statistical_confidence"] == "cannot_assess"
    assert guard["is_significant"] is False
    assert guard["sample_size_depressed"] == 0
    assert guard["sample_size_control"] == 0
    assert guard["observation_count_depressed"] == 40
    assert guard["observation_count_control"] == 40
    assert guard["aggregation_method"] == "none"

    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in result["critic_system_prompt"]
    assert "FORBIDDEN:" in result["critic_system_prompt"]
    assert "cannot assess: no subject dimension" in result["critic_system_prompt"].lower()
    critic_system_msg = critic.recorded_invocations[0][0].content
    assert "cannot assess: no subject dimension" in critic_system_msg.lower()


def test_pseudo_replicated_single_subject_rows_are_caught_as_underpowered_in_graph() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    router = _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT subject_id, label, night_post_ratio "
                "FROM gold_user_longitudinal_features"
            ),
            "metric_name": "night_post_ratio",
        }
    )
    critic = _critic(
        "Underpowered: depressed cohort has only 1 unique subject "
        "(depressed=1, control=35, p = 1.0, Cohen's d = 0.0)."
    )

    def fake_executor(_query: str, _cfg: Settings) -> list[dict[str, Any]]:
        rows = [
            {"subject_id": "dep_only", "label": 1, "night_post_ratio": 0.85 + (i % 4) * 0.02}
            for i in range(40)
        ]
        rows.extend(
            {"subject_id": f"ctrl_{i}", "label": 0, "night_post_ratio": 0.10 + (i % 5) * 0.02}
            for i in range(35)
        )
        return rows

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
    )
    result = graph.invoke(make_initial_state("Compare night post ratio."))

    guard = result["statistical_validation"]
    assert guard["sample_size_depressed"] == 1
    assert guard["observation_count_depressed"] == 40
    assert guard["sample_size_control"] == 35
    assert guard["observation_count_control"] == 35
    assert guard["aggregation_method"] == "subject_mean"
    assert guard["is_underpowered"] is True
    assert guard["is_significant"] is False
    assert "FORBIDDEN:" in result["critic_system_prompt"]


def test_topic_model_job_interrupts_at_hitl_and_executes_once_on_approval(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "hitl_topic_model.sqlite"
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    router = _router(
        {
            "user_intent": "lexical_analysis",
            "answer_from_manifest": False,
            "spark_job": "topic_model",
            "job_params": {"k": 8, "max_iter": 20},
            "sql_query": None,
            "metric_name": "mean_topic_probability",
        }
    )
    topic_critique = (
        "LDA topic recomputation completed after approval "
        "(cannot assess subject-level significance: depressed=0, control=0, p = 1.0, Cohen's d = 0.0)."
    )
    critic = _critic(topic_critique)
    sql_calls: list[str] = []
    job_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_sql(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        sql_calls.append(query)
        return []

    def fake_job(job_name: str, params: dict[str, Any], _cfg: Settings) -> list[dict[str, Any]]:
        job_calls.append((job_name, params))
        return [
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
        ]

    config = {"configurable": {"thread_id": "hitl-topic-model-approve"}}

    with open_sqlite_checkpointer(db_path) as saver1:
        graph1 = build_agent_graph(
            settings=settings,
            router_model=router,
            critic_model=critic,
            sql_executor=fake_sql,
            job_executor=fake_job,
            checkpointer=saver1,
        )
        interrupted = graph1.invoke(
            make_initial_state("Run a full LDA topic model recomputation across cohorts."),
            config=config,
        )

    assert "__interrupt__" in interrupted
    assert len(interrupted["__interrupt__"]) == 1
    assert interrupted["__interrupt__"][0].value["estimated_cost"] > 1.0
    assert job_calls == []
    assert sql_calls == []

    with open_sqlite_checkpointer(db_path) as saver2:
        graph2 = build_agent_graph(
            settings=settings,
            router_model=router,
            critic_model=critic,
            sql_executor=fake_sql,
            job_executor=fake_job,
            checkpointer=saver2,
        )
        snapshot = graph2.get_state(config)
        assert snapshot.next == ("hitl_gate",)
        assert "topic_model" in snapshot.values["generated_query"]

        resumed = graph2.invoke(Command(resume={"approved": True}), config=config)

    assert job_calls == [("topic_model", {"k": 8, "max_iter": 20})]
    assert sql_calls == []
    assert resumed["hitl_decision"] == "approved"
    assert resumed["status"] == "completed"
    assert resumed["final_response"] == topic_critique


def test_circadian_features_job_dispatch_runs_offline_and_guards_subjects() -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=2.0, min_subjects_per_class=30)
    router = _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "spark_job": "circadian_features",
            "job_params": {"window_days": 30},
            "sql_query": None,
            "metric_name": "night_post_ratio",
        }
    )
    critic = _critic(
        "Circadian job completed with significant cohort shift "
        "(depressed=35, control=35, p = 5.04483e-13, Cohen's d = 15.681)."
    )
    sql_calls: list[str] = []
    job_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_sql(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        sql_calls.append(query)
        return []

    def fake_job(job_name: str, params: dict[str, Any], _cfg: Settings) -> list[dict[str, Any]]:
        job_calls.append((job_name, params))
        return _cohort_rows(35)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_sql,
        job_executor=fake_job,
    )
    result = graph.invoke(
        make_initial_state("Recompute circadian window features for all subjects.")
    )

    assert sql_calls == []
    assert job_calls == [("circadian_features", {"window_days": 30})]
    assert result["status"] == "completed"
    assert result["sql_cost_estimate"]["estimated_cost"] > 1.0
    assert result["statistical_validation"]["assessment_status"] == "assessed"
    assert result["statistical_validation"]["is_significant"] is True


def test_unknown_spark_job_triggers_retry_and_recovers() -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=3.0, max_query_retries=2)
    router = _router(
        {
            "user_intent": "lexical_analysis",
            "answer_from_manifest": False,
            "spark_job": "invalid_job_name",
            "sql_query": None,
        },
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "spark_job": "circadian_features",
            "sql_query": None,
            "metric_name": "night_post_ratio",
        },
    )
    critic = _critic(
        "Recovered to valid job "
        "(depressed=30, control=30, p = 2.41976e-11, Cohen's d = 15.6425)."
    )
    job_calls: list[str] = []

    def fake_job(job_name: str, _params: dict[str, Any], _cfg: Settings) -> list[dict[str, Any]]:
        job_calls.append(job_name)
        return _cohort_rows(30)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        job_executor=fake_job,
    )
    result = graph.invoke(make_initial_state("Compute circadian features."))

    assert job_calls == ["circadian_features"]
    assert result["retry_count"] == 1
    assert result["status"] == "completed"
    assert "UNKNOWN_SPARK_JOB" in str(router.recorded_invocations[1][1].content)


def _cohort_router_model(*, include_subject_id: bool = True) -> RecordingFakeChatModel:
    cols = (
        "subject_id, label, night_post_ratio" if include_subject_id else "label, night_post_ratio"
    )
    return _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": f"SELECT {cols} FROM gold_user_longitudinal_features",
            "metric_name": "night_post_ratio",
        }
    )


def test_critic_compliant_first_response_makes_exactly_one_call() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    compliant = (
        "Depressed cohort (35 subjects) vs control cohort (35 subjects) showed a statistically "
        "significant difference (p = 5.04483e-13, Cohen's d = 15.681)."
    )
    critic = _critic(compliant)

    graph = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(),
        critic_model=critic,
        sql_executor=lambda _q, _cfg: _cohort_rows(35),
    )
    result = graph.invoke(make_initial_state("Compare night posting ratio."))

    assert len(critic.recorded_invocations) == 1
    assert result["critic_call_count"] == 1
    assert result["critic_retry_count"] == 0
    assert result["critic_output_check_failed"] is None
    assert result["final_response"] == compliant


def test_critic_omitting_p_value_triggers_single_corrective_retry() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30, max_query_retries=0)
    first_bad = "Significant difference observed (depressed=35, control=35, Cohen's d = 15.681)."
    second_good = (
        "Significant difference observed (depressed=35, control=35, "
        "p = 5.04483e-13, Cohen's d = 15.681)."
    )
    critic = _critic(first_bad, second_good)

    graph = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(),
        critic_model=critic,
        sql_executor=lambda _q, _cfg: _cohort_rows(35),
    )
    result = graph.invoke(make_initial_state("Compare night posting ratio."))

    assert len(critic.recorded_invocations) == 2
    assert result["critic_call_count"] == 2
    assert result["critic_retry_count"] == 1
    assert result["retry_count"] == 0
    assert result["critic_output_check_failed"] is None
    assert result["final_response"] == second_good
    correction_msg = str(critic.recorded_invocations[1][-1].content)
    assert "p_value" in correction_msg


def test_critic_never_complying_falls_back_to_deterministic_inconclusive_text() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    omits_d_1 = "Observed difference (depressed=35, control=35, p = 5.04483e-13)."
    omits_d_2 = "Still omitting effect size (depressed=35, control=35, p = 5.04483e-13)."
    critic = _critic(omits_d_1, omits_d_2, "Third call must never happen.")

    graph = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(),
        critic_model=critic,
        sql_executor=lambda _q, _cfg: _cohort_rows(35),
    )
    result = graph.invoke(make_initial_state("Compare night posting ratio."))

    assert len(critic.recorded_invocations) == 2
    assert result["critic_call_count"] == 2
    assert result["critic_retry_count"] == 1
    assert result["status"] == "completed"
    failure = result["critic_output_check_failed"]
    assert failure is not None
    assert failure["failed"] is True
    assert failure["retry_count"] == 1
    assert "cohens_d" in failure["missing_items"]

    final_text = result["final_response"]
    assert "non conclusivo" not in final_text.lower()
    assert "non viene affermata alcuna differenza" not in final_text.lower()
    assert "35" in final_text
    assert "5.04483e-13" in final_text
    assert "15.681" in final_text
    assert result["critic_critique"] == final_text

    guard = StatisticalGuardResult.model_validate(result["statistical_validation"])
    assert validate_critic_output(final_text, guard).passed is True

    # Non-significant guard failing citation check still falls back to inconclusive no-claim text.
    ns_critic = _critic(omits_d_1, omits_d_2, "Third call must never happen.")
    ns_result = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(),
        critic_model=ns_critic,
        sql_executor=lambda _q, _cfg: _cohort_rows(35, depressed_offset=-0.20),
    ).invoke(make_initial_state("Compare night posting ratio."))
    ns_final = ns_result["final_response"]
    assert "risultato non conclusivo" in ns_final.lower()
    assert "non viene affermata alcuna differenza" in ns_final.lower()
    ns_guard = StatisticalGuardResult.model_validate(ns_result["statistical_validation"])
    assert validate_critic_output(ns_final, ns_guard).passed is True


def test_nonsignificant_underpowered_and_cannot_assess_never_ship_significance_claims() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    rogue_claim = (
        "Depressed users post at night significantly more often "
        "(sample_size_depressed=35, sample_size_control=35, p = 0.248336, Cohen's d = 0.0000)."
    )

    # 1. Powered non-significant: rogue claim on call 1, corrected on call 2.
    compliant_ns = (
        "No statistically significant difference was found "
        "(sample_size_depressed=35, sample_size_control=35, p = 0.248336, Cohen's d = 0.0000)."
    )
    critic_ns = _critic(rogue_claim, compliant_ns)
    res_ns = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(),
        critic_model=critic_ns,
        sql_executor=lambda _q, _cfg: _cohort_rows(35, depressed_offset=-0.20),
    ).invoke(make_initial_state("Compare night post ratio."))
    guard_ns = StatisticalGuardResult.model_validate(res_ns["statistical_validation"])
    assert guard_ns.is_underpowered is False
    assert guard_ns.is_significant is False
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in res_ns["critic_system_prompt"]
    assert "FORBIDDEN:" in res_ns["critic_system_prompt"]
    assert "FORBIDDEN:" in str(critic_ns.recorded_invocations[0][0].content)
    assert "is_significant: False" in str(critic_ns.recorded_invocations[0][0].content)
    assert res_ns["critic_call_count"] == 2
    assert res_ns["critic_output_check_failed"] is None
    check_ns = validate_critic_output(res_ns["final_response"], guard_ns)
    assert check_ns.passed is True
    assert check_ns.forbidden_claims == []

    # 2. Underpowered: rogue claim on both calls -> deterministic fallback without significance claim.
    rogue_underpowered = (
        "Depressed users post at night significantly more often "
        "(depressed=5, control=5, p = 1.67857e-10, Cohen's d = 25.2982)."
    )
    critic_up = _critic(rogue_underpowered, rogue_underpowered)
    res_up = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(),
        critic_model=critic_up,
        sql_executor=lambda _q, _cfg: _cohort_rows(5, depressed_offset=0.6),
    ).invoke(make_initial_state("Compare night post ratio."))
    guard_up = StatisticalGuardResult.model_validate(res_up["statistical_validation"])
    assert guard_up.is_underpowered is True
    assert guard_up.is_significant is False
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in res_up["critic_system_prompt"]
    assert "FORBIDDEN:" in res_up["critic_system_prompt"]
    assert "Underpowered sample" in res_up["critic_system_prompt"]
    assert "FORBIDDEN:" in str(critic_up.recorded_invocations[0][0].content)
    assert "sample_size_depressed: 5" in str(critic_up.recorded_invocations[0][0].content)
    assert "sample_size_control: 5" in str(critic_up.recorded_invocations[0][0].content)
    assert res_up["critic_call_count"] == 2
    assert res_up["critic_output_check_failed"] is not None
    assert res_up["critic_output_check_failed"]["forbidden_claims"]
    check_up = validate_critic_output(res_up["final_response"], guard_up)
    assert check_up.passed is True
    assert check_up.forbidden_claims == []

    # 3. cannot_assess: rogue claim on both calls -> Italian fallback without fabricated p/d.
    rogue_ca = (
        "There is a statistically significant difference between depressed and control users."
    )
    critic_ca = _critic(rogue_ca, rogue_ca)
    res_ca = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(include_subject_id=False),
        critic_model=critic_ca,
        sql_executor=lambda _q, _cfg: [
            {"label": 1, "night_post_ratio": 0.6},
            {"label": 0, "night_post_ratio": 0.1},
        ],
    ).invoke(make_initial_state("Confronta il rapporto di post notturni per coorte."))
    guard_ca = StatisticalGuardResult.model_validate(res_ca["statistical_validation"])
    assert guard_ca.assessment_status == "cannot_assess"
    assert guard_ca.p_value is None
    assert guard_ca.effect_size_cohens_d is None
    assert res_ca["critic_call_count"] == 2
    assert res_ca["critic_output_check_failed"] is not None
    assert "impossibile valutare" in res_ca["final_response"].lower()
    assert "non viene affermata alcuna differenza" in res_ca["final_response"].lower()
    assert "p_value =" not in res_ca["final_response"]
    assert "cohen's d =" not in res_ca["final_response"].lower()
    check_ca = validate_critic_output(res_ca["final_response"], guard_ca)
    assert check_ca.passed is True
    assert check_ca.forbidden_claims == []


def test_descriptive_results_present_findings_and_block_significance_claims() -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)

    # 1. N-gram frequency table (gold_cohort_lexical_matrix)
    ngram_router = _router(
        {
            "user_intent": "lexical_analysis",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT label, ngram_type, ngram, frequency, tfidf_weight, tfidf_rank "
                "FROM gold_cohort_lexical_matrix ORDER BY label, tfidf_rank"
            ),
            "metric_name": "tfidf_weight",
        }
    )
    ngram_answer = (
        "I bigrammi con peso TF-IDF maggiore nella coorte label=1 sono "
        "'feel alone' (tfidf=0.14, freq=42) e 'cant sleep' (tfidf=0.11, freq=31). "
        "Trattandosi di una tabella descrittiva, non indica differenze statisticamente "
        "significative o segnali clinici."
    )
    ngram_critic = _critic(ngram_answer)
    ngram_res = build_agent_graph(
        settings=settings,
        router_model=ngram_router,
        critic_model=ngram_critic,
        sql_executor=lambda _q, _cfg: [
            {
                "label": 1,
                "ngram_type": "bigram",
                "ngram": "feel alone",
                "frequency": 42,
                "tfidf_weight": 0.14,
                "tfidf_rank": 1,
            },
            {
                "label": 0,
                "ngram_type": "bigram",
                "ngram": "weekend hike",
                "frequency": 35,
                "tfidf_weight": 0.11,
                "tfidf_rank": 1,
            },
        ],
    ).invoke(make_initial_state("Quali parole e bigrammi caratterizzano i soggetti depressi?"))

    ngram_guard = StatisticalGuardResult.model_validate(ngram_res["statistical_validation"])
    assert ngram_guard.assessment_status == "descriptive"
    assert ngram_guard.p_value is None
    assert ngram_guard.effect_size_cohens_d is None
    assert "DESCRIPTIVE_ONLY:" in ngram_res["critic_system_prompt"]
    assert "DO NOT refuse" in ngram_res["critic_system_prompt"]
    assert "SAME language" in ngram_res["critic_system_prompt"]
    assert ngram_res["final_response"] == ngram_answer
    assert ngram_res["critic_output_check_failed"] is None

    # 2. Topic table (gold_cohort_topic_matrix)
    topic_router = _router(
        {
            "user_intent": "lexical_analysis",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT label, topic, mean_topic_probability, document_count, top_terms "
                "FROM gold_cohort_topic_matrix"
            ),
            "metric_name": "mean_topic_probability",
        }
    )
    topic_answer = (
        "Topic 0 presenta i termini 'feel alone, cant sleep' "
        "(mean_topic_probability=0.42 in label=1; statistiche puramente descrittive)."
    )
    topic_res = build_agent_graph(
        settings=settings,
        router_model=topic_router,
        critic_model=_critic(topic_answer),
        sql_executor=lambda _q, _cfg: [
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
    ).invoke(make_initial_state("Mostra i topic LDA principali per coorte."))

    topic_guard = StatisticalGuardResult.model_validate(topic_res["statistical_validation"])
    assert topic_guard.assessment_status == "descriptive"
    assert topic_guard.p_value is None
    assert topic_guard.effect_size_cohens_d is None
    assert topic_res["final_response"] == topic_answer

    # 3. Single count query (silver_posts WHERE year = 2014)
    count_router = _router(
        {
            "user_intent": "ad_hoc_sql",
            "answer_from_manifest": False,
            "sql_query": "SELECT COUNT(*) AS post_count FROM silver_posts WHERE year = 2014",
        }
    )
    count_answer = "Nel 2014 sono presenti 1250 post in `silver_posts`."
    count_res = build_agent_graph(
        settings=settings,
        router_model=count_router,
        critic_model=_critic(count_answer),
        sql_executor=lambda _q, _cfg: [{"post_count": 1250}],
    ).invoke(make_initial_state("Quanti post ci sono nel 2014?"))

    count_guard = StatisticalGuardResult.model_validate(count_res["statistical_validation"])
    assert count_guard.assessment_status == "descriptive"
    assert count_guard.p_value is None
    assert count_guard.effect_size_cohens_d is None
    assert count_res["sql_cost_estimate"]["triggers_hitl"] is False
    assert count_res["final_response"] == count_answer

    # 4. Descriptive result making a forbidden significance claim is caught and replaced
    rogue_desc = "C'è una differenza statisticamente significativa nei bigrammi tra le coorti."
    desc_rogue_res = build_agent_graph(
        settings=settings,
        router_model=ngram_router,
        critic_model=_critic(rogue_desc, rogue_desc),
        sql_executor=lambda _q, _cfg: [
            {"label": 1, "ngram_type": "bigram", "ngram": "feel alone", "tfidf_weight": 0.14},
            {"label": 0, "ngram_type": "bigram", "ngram": "weekend hike", "tfidf_weight": 0.11},
        ],
    ).invoke(make_initial_state("Quali bigrammi caratterizzano i depressi?"))

    desc_rogue_guard = StatisticalGuardResult.model_validate(
        desc_rogue_res["statistical_validation"]
    )
    assert desc_rogue_guard.assessment_status == "descriptive"
    assert desc_rogue_res["critic_call_count"] == 2
    assert desc_rogue_res["critic_output_check_failed"] is not None
    assert "risultato descrittivo" in desc_rogue_res["final_response"].lower()
    assert validate_critic_output(desc_rogue_res["final_response"], desc_rogue_guard).passed is True


def test_hitl_pruned_silver_posts_skips_gate_while_unfiltered_triggers_gate(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)

    pruned_res = build_agent_graph(
        settings=settings,
        router_model=_router(
            {
                "user_intent": "ad_hoc_sql",
                "answer_from_manifest": False,
                "sql_query": "SELECT COUNT(*) FROM silver_posts WHERE year = 2014",
            }
        ),
        critic_model=_critic("Sono presenti 420 post nel 2014."),
        sql_executor=lambda _q, _cfg: [{"count(1)": 420}],
    ).invoke(make_initial_state("Quanti post ci sono nel 2014?"))

    assert pruned_res["status"] == "completed"
    assert pruned_res["sql_cost_estimate"]["triggers_hitl"] is False
    assert pruned_res["sql_cost_estimate"]["estimated_cost"] < 1.0

    with open_sqlite_checkpointer(tmp_path / "hitl_unfiltered.sqlite") as saver:
        unfiltered_graph = build_agent_graph(
            settings=settings,
            router_model=_router(
                {
                    "user_intent": "ad_hoc_sql",
                    "answer_from_manifest": False,
                    "sql_query": "SELECT * FROM silver_posts",
                }
            ),
            critic_model=_critic("unused"),
            sql_executor=lambda _q, _cfg: [],
            checkpointer=saver,
        )
        interrupted = unfiltered_graph.invoke(
            make_initial_state("Mostra tutti i post di silver_posts."),
            config={"configurable": {"thread_id": "unfiltered-silver-test"}},
        )

    assert "__interrupt__" in interrupted
    assert interrupted["__interrupt__"][0].value["estimated_cost"] > 1.0


def test_spark_job_plan_never_reports_ad_hoc_sql_intent() -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=5.0)

    lda_res = build_agent_graph(
        settings=settings,
        router_model=_router(
            {
                "user_intent": "ad_hoc_sql",
                "answer_from_manifest": False,
                "spark_job": "topic_model",
                "job_params": {"k": 8},
                "sql_query": None,
            }
        ),
        critic_model=_critic("Topic estratti (statistiche descrittive)."),
        job_executor=lambda _name, _params, _cfg: [
            {
                "label": 1,
                "topic": 0,
                "mean_topic_probability": 0.4,
                "document_count": 10,
                "top_terms": "feel alone",
            },
            {
                "label": 0,
                "topic": 0,
                "mean_topic_probability": 0.1,
                "document_count": 10,
                "top_terms": "morning coffee",
            },
        ],
    ).invoke(make_initial_state("Ricalcola il modello LDA sui post Silver."))

    assert lda_res["status"] == "completed"
    assert lda_res["user_intent"] == "topic_model"
    assert lda_res["user_intent"] != "ad_hoc_sql"
    assert lda_res["statistical_validation"]["assessment_status"] == "descriptive"

    inline_res = build_agent_graph(
        settings=settings,
        router_model=_router(
            {
                "user_intent": "ad_hoc_sql",
                "answer_from_manifest": False,
                "sql_query": 'SPARK_JOB:{"job": "circadian_features", "params": {"window_days": 30}}',
                "metric_name": "night_post_ratio",
            }
        ),
        critic_model=_critic(
            "Significant shift (depressed=30, control=30, p = 2.41976e-11, Cohen's d = 15.6425)."
        ),
        job_executor=lambda _name, _params, _cfg: _cohort_rows(30),
    ).invoke(make_initial_state("Ricalcola le feature circadiane."))

    assert inline_res["status"] == "completed"
    assert inline_res["user_intent"] == "circadian_features"
    assert inline_res["user_intent"] != "ad_hoc_sql"


@pytest.mark.spark
def test_cost_estimation_registers_lakehouse_views_before_explaining(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    """EXPLAIN resolves relations, so the estimate must register the views too.

    Regression: a plan naming `gold_user_longitudinal_features` raised
    TABLE_NOT_FOUND here, before the executor (which does register views) could
    run, and the retry loop then repeated the same failure until it gave up.
    """
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


@pytest.mark.spark
def test_full_cohort_819_subjects_analyzed_while_state_rows_capped_at_500(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    """819 subjects must all reach the guard while state/context rows stay <= sql_row_limit."""
    base_settings, paths, _ = lakehouse_env
    settings = Settings(
        _env_file=None,
        spark_storage_uri=base_settings.spark_storage_uri,
        min_subjects_per_class=30,
        sql_row_limit=500,
        sql_fetch_limit=50_000,
    )

    # 100 depressed (label=1) + 719 control (label=0) = 819 synthetic subjects,
    # mimicking the asymmetric class balance of the real eRISK corpus.
    synthetic_rows = [
        (f"dep_{i:04d}", 1, 0.55 + (i % 7) * 0.02) for i in range(100)
    ] + [
        (f"ctrl_{i:04d}", 0, 0.15 + (i % 7) * 0.02) for i in range(719)
    ]
    spark.createDataFrame(
        synthetic_rows,
        "subject_id STRING, label INT, night_post_ratio DOUBLE",
    ).write.mode("overwrite").parquet(str(paths.gold_user_longitudinal_features))

    critic = _critic(
        "Full cohort comparison across depressed=100 and control=719 subjects "
        "(p = 5.84328e-60, Cohen's d = 9.96672)."
    )
    sql_executor = make_spark_sql_executor(spark, settings=settings, paths=paths)
    graph = build_agent_graph(
        settings=settings,
        router_model=_cohort_router_model(),
        critic_model=critic,
        sql_executor=sql_executor,
        spark=spark,
    )

    result = graph.invoke(make_initial_state("Compare night_post_ratio across cohorts."))

    assert result["status"] == "completed"
    assert result["critic_call_count"] == 1
    assert result["critic_output_check_failed"] is None
    assert "LIMIT 50000" in result["generated_query"].upper()

    guard = result["statistical_validation"]
    assert guard["sample_size_depressed"] == 100
    assert guard["sample_size_control"] == 719
    assert guard["sample_size_depressed"] + guard["sample_size_control"] == 819

    qr = result["query_result"]
    assert qr["rows_returned"] == 819
    assert qr["rows_shown"] == 500
    assert len(qr["rows"]) == 500
    assert qr["truncated"] is False
    assert result["analysis_rows"] is None

    critic_user_msg = str(critic.recorded_invocations[0][1].content)
    assert '"rows_returned": 819' in critic_user_msg
    assert '"rows_shown": 500' in critic_user_msg
    assert '"truncated": false' in critic_user_msg


@pytest.mark.spark
def test_fetch_limit_cap_marks_truncated_and_demands_critic_disclosure(
    spark: SparkSession,
    lakehouse_env: tuple,
) -> None:
    """When query output hits sql_fetch_limit, truncated=True and critic prompt requires disclosure."""
    base_settings, paths, _ = lakehouse_env
    settings = Settings(
        _env_file=None,
        spark_storage_uri=base_settings.spark_storage_uri,
        min_subjects_per_class=5,
        sql_row_limit=500,
        sql_fetch_limit=20,
    )

    synthetic_rows = [
        (f"s_{i:03d}", i % 2, 0.2 + (i % 5) * 0.05) for i in range(60)
    ]
    spark.createDataFrame(
        synthetic_rows,
        "subject_id STRING, label INT, night_post_ratio DOUBLE",
    ).write.mode("overwrite").parquet(str(paths.gold_user_longitudinal_features))

    ordered_router = _router(
        {
            "user_intent": "circadian_study",
            "answer_from_manifest": False,
            "sql_query": (
                "SELECT subject_id, label, night_post_ratio "
                "FROM gold_user_longitudinal_features ORDER BY subject_id"
            ),
            "metric_name": "night_post_ratio",
        }
    )
    critic = _critic(
        "Capped sample analysis (truncated at 20 rows): "
        "depressed=10, control=10, p = 1.0, Cohen's d = 0.0."
    )
    sql_executor = make_spark_sql_executor(spark, settings=settings, paths=paths)
    graph = build_agent_graph(
        settings=settings,
        router_model=ordered_router,
        critic_model=critic,
        sql_executor=sql_executor,
        spark=spark,
    )

    result = graph.invoke(make_initial_state("Compare night_post_ratio across cohorts."))

    assert result["status"] == "completed"
    assert result["critic_call_count"] == 1
    assert result["critic_output_check_failed"] is None
    assert "LIMIT 20" in result["generated_query"].upper()

    qr = result["query_result"]
    assert qr["rows_returned"] == 20
    assert qr["rows_shown"] == 20
    assert qr["truncated"] is True
    assert result["statistical_validation"]["truncated"] is True

    critic_sys_msg = str(critic.recorded_invocations[0][0].content)
    assert "truncated=true" in critic_sys_msg.lower()
    assert "capped" in critic_sys_msg.lower()
    assert "sql_fetch_limit=20" in critic_sys_msg

    critic_user_msg = str(critic.recorded_invocations[0][1].content)
    assert '"truncated": true' in critic_user_msg


def test_hitl_cost_scale_separates_a_pruned_partition_from_a_full_scan() -> None:
    """One cost unit must mean "worth a human's time" for THIS dataset.

    Calibrated on the real corpus: at 50 MiB per unit a single year partition
    (57 MiB) tripped the gate on a query Spark answers in half a second, so a
    simple count needed manual approval. A pruned scan must stay below the
    threshold and an unfiltered full scan above it.
    """
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    pruned = estimate_sql_cost(
        "SELECT COUNT(*) AS post_count FROM silver_posts WHERE year = 2014",
        settings=settings,
    )
    full = estimate_sql_cost("SELECT * FROM silver_posts", settings=settings)

    assert pruned["triggers_hitl"] is False
    assert full["triggers_hitl"] is True
