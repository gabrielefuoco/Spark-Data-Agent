"""Typed state schema for the cyclic LangGraph Data Agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

UserIntent = Literal[
    "catalog_metadata",
    "lexical_analysis",
    "circadian_study",
    "ad_hoc_sql",
    "general_conversation",
]
AgentStatus = Literal[
    "running",
    "awaiting_approval",
    "completed",
    "cancelled",
    "failed",
]


class CostEstimate(TypedDict):
    """Computational cost summary attached before the HITL approval gate."""

    estimated_cost: float
    threshold: float
    triggers_hitl: bool
    method: Literal["spark_explain", "ast_heuristic", "injected"]
    reasons: list[str]
    explain_plan: str | None


class DataAgentState(TypedDict):
    """Shared state persisted across LangGraph nodes and SQLite checkpoints."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_intent: str
    generated_query: str | None
    analysis_metric: str | None
    sql_cost_estimate: dict[str, Any] | None
    query_result: dict[str, Any] | None
    statistical_validation: dict[str, Any] | None
    retry_count: int
    last_error: dict[str, Any] | None
    planner_notes: str | None
    hitl_decision: str | None
    critic_system_prompt: str | None
    critic_critique: str | None
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
        statistical_validation=None,
        retry_count=0,
        last_error=None,
        planner_notes=None,
        hitl_decision=None,
        critic_system_prompt=None,
        critic_critique=None,
        final_response=None,
        status="running",
    )
