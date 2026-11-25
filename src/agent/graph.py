"""Tool-calling LangGraph StateGraph assembly and SQLite checkpointer helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pyspark.sql import SparkSession

from src.agent.nodes import (
    AgentRuntimeDeps,
    JobExecutor,
    ManifestLoader,
    SqlExecutor,
    build_agent_nodes,
)
from src.agent.routing import route_after_agent, route_after_tools
from src.agent.state import DataAgentState
from src.config import Settings, get_settings


def create_sqlite_checkpointer(db_path: str | Path = ":memory:") -> SqliteSaver:
    """Create a persistent ``SqliteSaver`` suitable for long-lived UI sessions."""
    target = str(db_path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_agent_graph(
    *,
    settings: Settings | None = None,
    router_model: BaseChatModel | None = None,
    critic_model: BaseChatModel | None = None,
    sql_executor: SqlExecutor | None = None,
    job_executor: JobExecutor | None = None,
    spark: SparkSession | None = None,
    manifest_loader: ManifestLoader | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the tool-calling LangGraph ``StateGraph`` for lakehouse analytics.

    Dependencies (chat models, Spark/SQL executor, Spark job executor,
    manifest loader, SQLite checkpointer) are injected so offline tests never
    touch networked LLMs or start a JVM SparkSession.
    """
    cfg = settings or get_settings()
    deps = AgentRuntimeDeps(
        settings=cfg,
        router_model=router_model,
        critic_model=critic_model,
        sql_executor=sql_executor,
        job_executor=job_executor,
        spark=spark,
        manifest_loader=manifest_loader,
    )
    nodes = build_agent_nodes(deps)

    builder = StateGraph(DataAgentState)
    for name, node_fn in nodes.items():
        builder.add_node(name, node_fn)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "execute_tools": "execute_tools",
            "scientific_critic": "scientific_critic",
            "handle_failure": "handle_failure",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "execute_tools": "execute_tools",
            "agent": "agent",
            "handle_failure": "handle_failure",
        },
    )
    builder.add_edge("scientific_critic", END)
    builder.add_edge("handle_failure", END)

    compiled = builder.compile(checkpointer=checkpointer)
    return compiled.with_config({"recursion_limit": max(25, cfg.max_tool_calls * 3 + 15)})
