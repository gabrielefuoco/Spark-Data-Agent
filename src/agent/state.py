"""Typed state schema for the cyclic LangGraph Data Agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class DataAgentState(TypedDict):
    """Shared state persisted across LangGraph nodes and SQLite checkpoints."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_intent: str
    generated_query: str | None
    analysis_metric: str | None
    sql_cost_estimate: dict[str, Any] | None
    query_result: dict[str, Any] | None
    analysis_rows: list[dict[str, Any]] | None
    statistical_validation: dict[str, Any] | None
    retry_count: int
    last_error: dict[str, Any] | None
    planner_notes: str | None
    hitl_decision: str | None
    critic_system_prompt: str | None
    critic_critique: str | None
    critic_call_count: int
    critic_retry_count: int
    critic_output_check_failed: dict[str, Any] | None
    final_response: str | None
    status: str


def make_initial_state(question: str) -> DataAgentState:
    """Create a fresh ``DataAgentState`` seeded with a single user question."""
    return DataAgentState(
        messages=[HumanMessage(content=question)],
        user_intent="",
        generated_query=None,
        analysis_metric=None,
        sql_cost_estimate=None,
        query_result=None,
        analysis_rows=None,
        statistical_validation=None,
        retry_count=0,
        last_error=None,
        planner_notes=None,
        hitl_decision=None,
        critic_system_prompt=None,
        critic_critique=None,
        critic_call_count=0,
        critic_retry_count=0,
        critic_output_check_failed=None,
        final_response=None,
        status="running",
    )
