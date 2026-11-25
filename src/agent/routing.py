"""Conditional-edge routing functions for the cyclic LangGraph Data Agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langgraph.graph import END

from src.agent.nodes import decode_spark_job
from src.agent.state import DataAgentState


def _route_error_or_retry(state: DataAgentState, max_query_retries: int) -> str | None:
    """Return ``query_rewriter`` or ``handle_failure`` when ``last_error`` is present."""
    if state.get("last_error") is None:
        return None
    if state.get("retry_count", 0) < max_query_retries:
        return "query_rewriter"
    return "handle_failure"


def route_after_planner(state: DataAgentState, max_query_retries: int) -> str:
    """Route after planning: finish direct replies, retry on error, or validate."""
    if state.get("status") in {"completed", "failed"} and state.get("final_response"):
        return END
    if (error_route := _route_error_or_retry(state, max_query_retries)) is not None:
        return error_route
    return "ast_validator_and_cost"


def route_after_validator(state: DataAgentState, max_query_retries: int) -> str:
    """Route after AST/job validation: retry on error, gate costly runs, or execute."""
    if (error_route := _route_error_or_retry(state, max_query_retries)) is not None:
        return error_route
    cost_info = state.get("sql_cost_estimate") or {}
    if cost_info.get("triggers_hitl"):
        return "hitl_gate"
    if decode_spark_job(state.get("generated_query")) is not None:
        return "execute_job"
    return "execute_sql"


def route_after_hitl(state: DataAgentState) -> str:
    """Route after human approval: stop on rejection or dispatch SQL/Spark execution."""
    if state.get("hitl_decision") == "rejected" or state.get("status") == "cancelled":
        return END
    if decode_spark_job(state.get("generated_query")) is not None:
        return "execute_job"
    return "execute_sql"


def route_after_execute(state: DataAgentState, max_query_retries: int) -> str:
    """Route after SQL/job execution: retry on failure or evaluate statistical guard."""
    if (error_route := _route_error_or_retry(state, max_query_retries)) is not None:
        return error_route
    return "statistical_guard"


@dataclass(frozen=True)
class AgentRouters:
    """Bound conditional-edge router callables for ``StateGraph`` assembly."""

    after_planner: Callable[[DataAgentState], str]
    after_validator: Callable[[DataAgentState], str]
    after_hitl: Callable[[DataAgentState], str]
    after_execute: Callable[[DataAgentState], str]


def _bind_retry_router(
    fn: Callable[[DataAgentState, int], str],
    max_query_retries: int,
) -> Callable[[DataAgentState], str]:
    def _router(state: DataAgentState) -> str:
        return fn(state, max_query_retries)

    _router.__name__ = fn.__name__
    return _router


def build_agent_routers(max_query_retries: int) -> AgentRouters:
    """Create bound router callables parameterized by ``max_query_retries``."""
    return AgentRouters(
        after_planner=_bind_retry_router(route_after_planner, max_query_retries),
        after_validator=_bind_retry_router(route_after_validator, max_query_retries),
        after_hitl=route_after_hitl,
        after_execute=_bind_retry_router(route_after_execute, max_query_retries),
    )
