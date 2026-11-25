"""Conditional-edge routing functions for the tool-calling LangGraph Data Agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

from src.agent.state import DataAgentState


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
    messages = state.get("messages", [])
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            answered_ids = {
                m.tool_call_id
                for m in messages[idx + 1 :]
                if isinstance(m, ToolMessage) and m.tool_call_id
            }
            if any(tc.get("id") not in answered_ids for tc in msg.tool_calls):
                return "execute_tools"
            break
    return "agent"
