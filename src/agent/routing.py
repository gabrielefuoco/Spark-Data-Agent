"""Conditional-edge routing functions for the tool-calling LangGraph Data Agent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END

from src.agent.state import DataAgentState


def find_pending_tool_calls(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Return unanswered tool calls from the latest ``AIMessage`` that requested tools."""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            answered_ids = {
                m.tool_call_id
                for m in messages[idx + 1 :]
                if isinstance(m, ToolMessage) and m.tool_call_id
            }
            return [dict(tc) for tc in msg.tool_calls if tc.get("id") not in answered_ids]
    return []


def route_after_agent(state: DataAgentState) -> str:
    """Route after ``agent``: fail if stuck, run tools if requested, critique results, or end."""
    if state.get("status") == "failing":
        return "handle_failure"
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "execute_tools"
    if state.get("tool_results"):
        return "scientific_critic"
    return END


def route_after_tools(state: DataAgentState) -> str:
    """Route after ``execute_tools``: halt repeat loops, drain pending tool calls, or return to ``agent``."""
    if state.get("repeat_loop_halted"):
        return "handle_failure"
    if find_pending_tool_calls(state.get("messages", [])):
        return "execute_tools"
    return "agent"
