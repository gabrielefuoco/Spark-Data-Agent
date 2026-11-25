"""Cyclic LangGraph StateGraph assembly and SQLite checkpointer helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pyspark.sql import SparkSession

from src.agent.nodes import (
    SUPPORTED_SPARK_JOBS,
    AgentRuntimeDeps,
    JobExecutor,
    ManifestLoader,
    SqlExecutor,
    build_agent_nodes,
    decode_spark_job,
    encode_spark_job,
    make_spark_job_executor,
    make_spark_sql_executor,
)
from src.agent.routing import build_agent_routers
from src.agent.state import DataAgentState
from src.config import Settings, get_settings

__all__ = [
    "SUPPORTED_SPARK_JOBS",
    "JobExecutor",
    "ManifestLoader",
    "SqlExecutor",
    "build_agent_graph",
    "create_sqlite_checkpointer",
    "decode_spark_job",
    "encode_spark_job",
    "make_spark_job_executor",
    "make_spark_sql_executor",
    "open_sqlite_checkpointer",
]


@contextmanager
def open_sqlite_checkpointer(db_path: str | Path = ":memory:") -> Iterator[SqliteSaver]:
    """Open a ``SqliteSaver`` checkpointer as a managed context."""
    target = str(db_path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(target) as saver:
        yield saver


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
    """Compile the cyclic LangGraph ``StateGraph`` for lakehouse analytics.

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
    routers = build_agent_routers(cfg.max_query_retries)

    builder = StateGraph(DataAgentState)
    for name, node_fn in nodes.items():
        builder.add_node(name, node_fn)

    builder.add_edge(START, "intent_router_planner")
    builder.add_conditional_edges(
        "intent_router_planner",
        routers.after_planner,
        {
            "ast_validator_and_cost": "ast_validator_and_cost",
            "query_rewriter": "query_rewriter",
            "handle_failure": "handle_failure",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "ast_validator_and_cost",
        routers.after_validator,
        {
            "hitl_gate": "hitl_gate",
            "execute_sql": "execute_sql",
            "execute_job": "execute_job",
            "query_rewriter": "query_rewriter",
            "handle_failure": "handle_failure",
        },
    )
    builder.add_conditional_edges(
        "hitl_gate",
        routers.after_hitl,
        {
            "execute_sql": "execute_sql",
            "execute_job": "execute_job",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "execute_sql",
        routers.after_execute,
        {
            "statistical_guard": "statistical_guard",
            "query_rewriter": "query_rewriter",
            "handle_failure": "handle_failure",
        },
    )
    builder.add_conditional_edges(
        "execute_job",
        routers.after_execute,
        {
            "statistical_guard": "statistical_guard",
            "query_rewriter": "query_rewriter",
            "handle_failure": "handle_failure",
        },
    )
    builder.add_edge("query_rewriter", "intent_router_planner")
    builder.add_edge("statistical_guard", "scientific_critic")
    builder.add_edge("scientific_critic", END)
    builder.add_edge("handle_failure", END)

    return builder.compile(checkpointer=checkpointer)
