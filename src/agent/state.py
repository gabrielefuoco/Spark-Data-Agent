"""Typed state schema for the tool-calling LangGraph Data Agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class ToolTraceEntry(TypedDict):
    """Auditable record for a single tool call in the agent loop."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    rows_returned: int
    guard_payload: dict[str, Any]
    status: str
    cost_estimate: dict[str, Any] | None
    error: dict[str, Any] | None


class ToolResultEntry(TypedDict):
    """Executed tool result paired with its deterministic statistical guard payload."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    query: str | None
    cost_estimate: dict[str, Any] | None
    result: dict[str, Any]
    guard_payload: dict[str, Any]


class DataAgentState(TypedDict):
    """Shared state persisted across LangGraph nodes and SQLite checkpoints."""

    messages: Annotated[list[AnyMessage], add_messages]
    tool_trace: list[ToolTraceEntry]
    tool_results: list[ToolResultEntry]
    tool_call_count: int
    last_tool_signature: str | None
    consecutive_repeat_count: int
    repeat_cap_triggered: bool
    repeat_loop_halted: bool
    budget_exhausted: bool
    manifest_digest: str | None
    agent_draft_response: str | None
    generated_query: str | None
    query_result: dict[str, Any] | None
    statistical_validation: dict[str, Any] | None
    hitl_decision: str | None
    critic_output_check_failed: dict[str, Any] | None
    final_response: str | None
    status: str


def make_initial_state(question: str) -> DataAgentState:
    """Create a fresh ``DataAgentState`` seeded with a single user question."""
    return DataAgentState(
        messages=[HumanMessage(content=question)],
        tool_trace=[],
        tool_results=[],
        tool_call_count=0,
        last_tool_signature=None,
        consecutive_repeat_count=0,
        repeat_cap_triggered=False,
        repeat_loop_halted=False,
        budget_exhausted=False,
        manifest_digest=None,
        agent_draft_response=None,
        generated_query=None,
        query_result=None,
        statistical_validation=None,
        hitl_decision=None,
        critic_output_check_failed=None,
        final_response=None,
        status="running",
    )
