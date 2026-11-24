"""Tests for the Streamlit UI and Plotly chart helpers without Spark or LLMs."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from src.agent.graph import build_agent_graph, create_sqlite_checkpointer
from src.config import PROJECT_ROOT, Settings
from src.ui.app import build_result_figure
from streamlit.testing.v1 import AppTest

APP_PATH = PROJECT_ROOT / "src" / "ui" / "app.py"


def test_build_result_figure_handles_cohort_and_empty_results() -> None:
    assert build_result_figure(None) is None
    assert build_result_figure({"rows": []}) is None

    cohort_result = {
        "columns": ["subject_id", "label", "night_post_ratio"],
        "rows": [
            {"subject_id": f"s{i}", "label": i % 2, "night_post_ratio": 0.1 + 0.05 * i}
            for i in range(10)
        ],
    }
    fig = build_result_figure(cohort_result, {"metric_name": "night_post_ratio"})
    assert fig is not None
    assert len(fig.data) >= 1


def _build_hitl_test_graph(executor_calls: list[str]) -> Any:
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)
    router = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content=json.dumps(
                    {
                        "user_intent": "ad_hoc_sql",
                        "answer_from_manifest": False,
                        "sql_query": "SELECT subject_id, label, text FROM silver_posts",
                    }
                )
            )
        ]
    )
    critic = FakeMessagesListChatModel(
        responses=[AIMessage(content="Analisi completata dopo autorizzazione HITL.")]
    )

    def fake_executor(query: str, _cfg: Settings) -> list[dict[str, Any]]:
        executor_calls.append(query)
        return [
            {"subject_id": "s1", "label": 1, "text": "post 1"},
            {"subject_id": "s2", "label": 0, "text": "post 2"},
        ]

    saver = create_sqlite_checkpointer(":memory:")
    return build_agent_graph(
        settings=settings,
        router_model=router,
        critic_model=critic,
        sql_executor=fake_executor,
        checkpointer=saver,
    )


def test_ui_hitl_approve_flow_resumes_same_thread() -> None:
    executor_calls: list[str] = []
    graph = _build_hitl_test_graph(executor_calls)

    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.session_state["agent_graph"] = graph
    at.session_state["thread_id"] = "ui-test-approve"
    at.run()

    at.chat_input[0].set_value("Analizza i post raw in silver_posts").run()

    # Interrupted at HITL gate: executor has not run yet, warning + buttons are visible.
    assert executor_calls == []
    assert not at.exception
    assert len(at.warning) >= 1
    assert any("Autorizza" in btn.label for btn in at.button)
    assert any("Annulla" in btn.label for btn in at.button)

    approve_btn = next(btn for btn in at.button if btn.key == "hitl_approve")
    approve_btn.click().run()

    assert not at.exception
    assert len(executor_calls) == 1
    assert at.session_state["pending_interrupt"] is None
    assert any(
        "Analisi completata dopo autorizzazione HITL." in md.value
        for md in at.markdown
    )


def test_ui_hitl_reject_flow_never_executes_query() -> None:
    executor_calls: list[str] = []
    graph = _build_hitl_test_graph(executor_calls)

    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.session_state["agent_graph"] = graph
    at.session_state["thread_id"] = "ui-test-reject"
    at.run()

    at.chat_input[0].set_value("Scansiona silver_posts").run()
    assert executor_calls == []

    reject_btn = next(btn for btn in at.button if btn.key == "hitl_reject")
    reject_btn.click().run()

    assert not at.exception
    assert executor_calls == []
    assert at.session_state["pending_interrupt"] is None
    assert any("cancelled" in md.value.lower() for md in at.markdown)
