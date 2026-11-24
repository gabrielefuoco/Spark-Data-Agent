"""Offline tests for the cyclic LangGraph Data Agent (fake LLMs + fake executor)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.types import Command
from src.agent.graph import build_agent_graph, open_sqlite_checkpointer
from src.agent.guards import CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS
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


def test_manifest_only_question_never_invokes_sql_executor() -> None:
    settings = Settings(_env_file=None)
    router = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "catalog_metadata",
                        "answer_from_manifest": True,
                        "sql_query": None,
                        "metric_name": None,
                        "direct_response": None,
                    }
                )
            )
        ]
    )
    critic = GenericFakeChatModel(messages=iter(["Critic must not be called for manifest queries."]))
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
    settings = Settings(_env_file=None, min_subjects_per_class=30, sql_row_limit=100)
    router = RecordingFakeChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
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
            )
        ]
    )
    critic = RecordingFakeChatModel(
        responses=[AIMessage(content="Significant difference observed (p < 0.001, d > 0.8).")]
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
    assert "LIMIT 100" in executor_calls[0].upper()
    assert result["status"] == "completed"
    assert result["sql_cost_estimate"]["triggers_hitl"] is False
    assert result["statistical_validation"]["sample_size_depressed"] == 35
    assert result["statistical_validation"]["sample_size_control"] == 35
    assert result["statistical_validation"]["is_significant"] is True
    assert result["statistical_validation"]["is_underpowered"] is False

    # Verify the critic prompt receives the mandatory guardrail instructions and verdict.
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in result["critic_system_prompt"]
    assert "ALLOWED:" in result["critic_system_prompt"]
    assert len(critic.recorded_invocations) == 1
    critic_system_msg = critic.recorded_invocations[0][0].content
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in critic_system_msg
    assert "sample_size_depressed: 35" in critic_system_msg


def test_underpowered_or_nonsignificant_guard_forbids_claims_in_critic_prompt() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    router = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
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
            )
        ]
    )
    critic = RecordingFakeChatModel(
        responses=[AIMessage(content="Sample is underpowered; no correlation claimed.")]
    )

    def fake_executor(_query: str, _cfg: Settings) -> list[dict[str, Any]]:
        # Only 5 subjects per class -> underpowered (min=30) even with huge separation.
        return _cohort_rows(5, depressed_offset=0.6)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
    )
    result = graph.invoke(make_initial_state("Check night post ratio."))

    guard = result["statistical_validation"]
    assert guard["is_underpowered"] is True
    assert guard["is_significant"] is False
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in result["critic_system_prompt"]
    assert "FORBIDDEN:" in result["critic_system_prompt"]
    assert "Underpowered sample" in result["critic_system_prompt"]

    critic_system_msg = critic.recorded_invocations[0][0].content
    assert "FORBIDDEN:" in critic_system_msg
    assert "sample_size_depressed: 5" in critic_system_msg
    assert "sample_size_control: 5" in critic_system_msg


def test_powered_nonsignificant_result_forbids_claims_in_critic_prompt() -> None:
    settings = Settings(_env_file=None, min_subjects_per_class=30)
    router = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
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
            )
        ]
    )
    critic = RecordingFakeChatModel(
        responses=[AIMessage(content="No statistically significant difference detected.")]
    )

    def fake_executor(_query: str, _cfg: Settings) -> list[dict[str, Any]]:
        # Powered cohorts (35 per class) with identical distributions -> p = 1.0, is_significant = False.
        return _cohort_rows(35, depressed_offset=-0.20)

    graph = build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
    )
    result = graph.invoke(make_initial_state("Check night post ratio."))

    guard = result["statistical_validation"]
    assert guard["is_underpowered"] is False
    assert guard["is_significant"] is False
    assert CRITIC_STATISTICAL_GUARDRAIL_INSTRUCTIONS in result["critic_system_prompt"]
    assert "FORBIDDEN:" in result["critic_system_prompt"]
    critic_system_msg = critic.recorded_invocations[0][0].content
    assert "FORBIDDEN:" in critic_system_msg
    assert "is_significant: False" in critic_system_msg


def test_ast_validation_rejection_triggers_retry_without_touching_executor() -> None:
    settings = Settings(_env_file=None, max_query_retries=2)
    router = RecordingFakeChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "ad_hoc_sql",
                        "answer_from_manifest": False,
                        "sql_query": "DROP TABLE silver_posts",
                    }
                )
            ),
            AIMessage(
                content=json.dumps(
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
            ),
        ]
    )
    critic = RecordingFakeChatModel(responses=[AIMessage(content="Recovered after AST fix.")])
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
    router = RecordingFakeChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "ad_hoc_sql",
                        "answer_from_manifest": False,
                        "sql_query": "SELECT bad_col FROM gold_user_longitudinal_features",
                    }
                )
            ),
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "ad_hoc_sql",
                        "answer_from_manifest": False,
                        "sql_query": "SELECT bad_col_2 FROM gold_user_longitudinal_features",
                    }
                )
            ),
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "ad_hoc_sql",
                        "answer_from_manifest": False,
                        "sql_query": "SELECT bad_col_3 FROM gold_user_longitudinal_features",
                    }
                )
            ),
        ]
    )
    critic = RecordingFakeChatModel(responses=[AIMessage(content="unused")])
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
    router = RecordingFakeChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "circadian_study",
                        "answer_from_manifest": False,
                        "sql_query": "SELECT wrong_col FROM gold_user_longitudinal_features",
                    }
                )
            ),
            AIMessage(
                content=json.dumps(
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
            ),
        ]
    )
    critic = RecordingFakeChatModel(responses=[AIMessage(content="Completed on retry.")])
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
    assert result["final_response"] == "Completed on retry."


def test_empty_result_triggers_retry_and_recovers() -> None:
    settings = Settings(_env_file=None, max_query_retries=2)
    router = RecordingFakeChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "circadian_study",
                        "answer_from_manifest": False,
                        "sql_query": (
                            "SELECT subject_id, label, night_post_ratio "
                            "FROM gold_user_longitudinal_features WHERE label = 99"
                        ),
                    }
                )
            ),
            AIMessage(
                content=json.dumps(
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
            ),
        ]
    )
    critic = RecordingFakeChatModel(responses=[AIMessage(content="Recovered after empty result.")])
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

    router = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "ad_hoc_sql",
                        "answer_from_manifest": False,
                        "sql_query": expensive_sql,
                    }
                )
            )
        ]
    )
    critic = FakeMessagesListChatModel(
        responses=[AIMessage(content="Approved expensive query completed.")]
    )
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
    assert resumed["final_response"] == "Approved expensive query completed."


def test_hitl_refusal_never_executes_and_returns_cancelled_state(tmp_path: Path) -> None:
    db_path = tmp_path / "hitl_reject.sqlite"
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    router = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "ad_hoc_sql",
                        "answer_from_manifest": False,
                        "sql_query": "SELECT * FROM silver_posts",
                    }
                )
            )
        ]
    )
    critic = FakeMessagesListChatModel(responses=[AIMessage(content="Should not run")])
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
