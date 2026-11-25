"""Streamlit conversational UI for the Spark Data Agent lakehouse.

Run inside the dev container bound to ``0.0.0.0:8501``:
``streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 8501``
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langgraph.types import Command

from src.agent.graph import build_agent_graph, create_sqlite_checkpointer
from src.agent.guards import _as_finite_float
from src.agent.state import make_initial_state
from src.config import PROJECT_ROOT, Settings, get_settings
from src.lakehouse.manifest import DatasetManifest, load_manifest
from src.spark import build_spark_session


def build_result_figure(
    query_result: dict[str, Any] | None,
    statistical_validation: dict[str, Any] | None = None,
) -> go.Figure | None:
    """Create a Plotly visualization from Spark SQL query rows when plottable."""
    if not query_result:
        return None
    rows = query_result.get("rows") or []
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"label", "year"}
        and any(_as_finite_float(v) is not None for v in df[col].tolist())
    ]

    if "label" in df.columns and numeric_cols:
        preferred = (statistical_validation or {}).get("metric_name")
        metric_col = preferred if preferred in numeric_cols else numeric_cols[0]
        plot_df = df.copy()
        plot_df["coorte"] = plot_df["label"].map(
            lambda val: "Depressi (1)" if str(val) == "1" else "Controllo (0)"
        )
        if len(plot_df) > 4:
            return px.box(
                plot_df,
                x="coorte",
                y=metric_col,
                color="coorte",
                points="all",
                title=f"Distribuzione per coorte: {metric_col}",
            )
        return px.bar(
            plot_df,
            x="coorte",
            y=metric_col,
            color="coorte",
            title=f"Confronto per coorte: {metric_col}",
        )

    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    if categorical_cols and numeric_cols:
        return px.bar(
            df.head(30),
            x=categorical_cols[0],
            y=numeric_cols[0],
            title=f"{numeric_cols[0]} per {categorical_cols[0]}",
        )

    if numeric_cols:
        return px.histogram(
            df,
            x=numeric_cols[0],
            title=f"Distribuzione: {numeric_cols[0]}",
        )

    return None


@st.cache_resource
def _get_cached_runtime(
    storage_uri: str,
    llm_backend: str,
    checkpoint_db_path: str,
) -> Any:
    """Lazily initialize SparkSession, SQLite checkpointer, and compiled StateGraph."""
    settings = get_settings()
    spark = build_spark_session(settings)
    checkpointer = create_sqlite_checkpointer(checkpoint_db_path)
    return build_agent_graph(
        settings=settings,
        spark=spark,
        checkpointer=checkpointer,
    )


def _get_graph(settings: Settings) -> Any:
    """Return an injected test graph from session_state or build the live runtime graph."""
    injected = st.session_state.get("agent_graph")
    if injected is not None:
        return injected

    # SQLite requires a local filesystem for file locking and cannot reside on s3a://.
    checkpoint_path = str(PROJECT_ROOT / "data" / "agent_checkpoints.sqlite")
    return _get_cached_runtime(
        settings.spark_storage_uri,
        settings.llm_backend,
        checkpoint_path,
    )


def _load_manifest_safe(settings: Settings) -> DatasetManifest | None:
    try:
        return load_manifest(settings=settings)
    except FileNotFoundError:
        return None


def _extract_interrupt_payload(interrupt_obj: Any) -> dict[str, Any]:
    raw = getattr(interrupt_obj, "value", interrupt_obj)
    if isinstance(raw, dict):
        return dict(raw)
    return {"message": str(raw), "query": "", "estimated_cost": 0.0, "threshold": 1.0}


def _record_assistant_outcome(outcome: dict[str, Any]) -> None:
    content = (
        outcome.get("final_response") or outcome.get("critic_critique") or "Analisi completata."
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "generated_query": outcome.get("generated_query"),
            "sql_cost_estimate": outcome.get("sql_cost_estimate"),
            "query_result": outcome.get("query_result"),
            "statistical_validation": outcome.get("statistical_validation"),
            "status": outcome.get("status"),
        }
    )


def _invoke_or_interrupt(graph: Any, payload: Any, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(payload, config=config)
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts:
        st.session_state["pending_interrupt"] = _extract_interrupt_payload(interrupts[0])
        return

    st.session_state["pending_interrupt"] = None
    _record_assistant_outcome(
        result if isinstance(result, dict) else {"final_response": str(result)}
    )


def _render_statistical_card(stats_payload: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N Depressi (1)", stats_payload.get("sample_size_depressed", 0))
    col2.metric("N Controllo (0)", stats_payload.get("sample_size_control", 0))
    p_val = stats_payload.get("p_value")
    col3.metric("p-value", f"{p_val:.4g}" if isinstance(p_val, (int, float)) else "N/A")
    cohens_d = stats_payload.get("effect_size_cohens_d")
    col4.metric("Cohen's d", f"{cohens_d:.3f}" if isinstance(cohens_d, (int, float)) else "N/A")

    if stats_payload.get("assessment_status") == "descriptive":
        st.info("Analisi descrittiva: nessun test di significatività statistica applicato.")
    elif stats_payload.get("assessment_status") == "cannot_assess":        st.warning(
            stats_payload.get("cannot_assess_reason")
            or "Impossibile valutare la significatività statistica senza dimensione `subject_id`."
        )
    elif stats_payload.get("is_underpowered"):
        st.warning(
            stats_payload.get("underpowered_reason") or "Campione sottodimensionato (underpowered)."
        )
    elif not stats_payload.get("is_significant"):
        st.info("Nessuna differenza statisticamente significativa rilevata (p >= 0.05).")
    else:
        st.success(
            f"Test: {stats_payload.get('test_used')} | Confidenza: {stats_payload.get('statistical_confidence')}"
        )

    balance_ratio = stats_payload.get("balance_ratio")
    if stats_payload.get("is_imbalanced") and isinstance(balance_ratio, (int, float)):
        st.warning(
            f"Coorti sbilanciate {balance_ratio:.1f}:1 ({stats_payload.get('sample_size_depressed')} "
            f"depressi vs {stats_payload.get('sample_size_control')} controlli): la coorte più "
            "piccola determina la precisione della stima."
        )


def _render_assistant_message(msg: dict[str, Any]) -> None:
    st.markdown(msg["content"])

    query = msg.get("generated_query")
    if query:
        with st.expander("Query Spark SQL eseguita", expanded=False):
            st.code(query, language="sql")

    stats_payload = msg.get("statistical_validation")
    if isinstance(stats_payload, dict):
        _render_statistical_card(stats_payload)

    query_result = msg.get("query_result")
    if isinstance(query_result, dict) and query_result.get("rows"):
        fig = build_result_figure(query_result, stats_payload)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        with st.expander(
            f"Dati estratti ({query_result.get('row_count', 0)} righe)", expanded=False
        ):
            st.dataframe(pd.DataFrame(query_result["rows"]), width="stretch")


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None


def main() -> None:
    """Render the Streamlit conversational analytics interface."""
    st.set_page_config(
        page_title="Spark Data Agent — Lakehouse eRISK",
        page_icon="🔬",
        layout="wide",
    )
    _init_session_state()
    settings = get_settings()

    st.title("🔬 Spark Data Agent — Lakehouse eRISK")
    st.caption(
        "Analisi conversazionale su PySpark Medallion Lakehouse con validazione AST, "
        "Human-in-the-Loop e guardia di significatività statistica."
    )

    with st.sidebar:
        st.header("Configurazione Runtime")
        model_label = (
            settings.ollama_model
            if settings.llm_backend == "ollama"
            else f"{settings.remote_router_model} / {settings.remote_reasoning_model}"
        )
        st.markdown(f"- **Backend LLM**: `{settings.llm_backend}`")
        st.markdown(f"- **Modelli**: `{model_label}`")
        st.markdown(f"- **Soglia HITL**: `{settings.hitl_cost_threshold}`")
        st.markdown(f"- **Max Retry SQL**: `{settings.max_query_retries}`")
        st.markdown(f"- **Thread ID**: `{st.session_state.thread_id}`")

        if st.button("Nuova Conversazione"):
            st.session_state.messages = []
            st.session_state.pending_interrupt = None
            st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
            st.rerun()

        manifest = _load_manifest_safe(settings)
        if manifest is not None:
            st.divider()
            st.subheader("Catalogo Gold (Manifest)")
            st.metric("Post totali", manifest.total_posts)
            st.metric("Soggetti totali", manifest.total_subjects)
            st.caption(f"Intervallo: {manifest.min_date or 'N/A'} -> {manifest.max_date or 'N/A'}")

    if prompt := st.chat_input("Fai una domanda sul corpus eRISK (es. post notturni, metadati)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        graph = _get_graph(settings)
        _invoke_or_interrupt(graph, make_initial_state(prompt), st.session_state.thread_id)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(message)
            else:
                st.markdown(message["content"])

    pending = st.session_state.get("pending_interrupt")
    if pending is not None:
        st.warning(
            "Approvazione Richiesta (Human-in-the-Loop): la query supera la soglia di costo computazionale."
        )
        col_cost, col_thresh = st.columns(2)
        est_cost = pending.get("estimated_cost", 0.0)
        threshold = pending.get("threshold", settings.hitl_cost_threshold)
        col_cost.metric(
            "Costo Stimato",
            f"{est_cost:.2f}" if isinstance(est_cost, (int, float)) else str(est_cost),
        )
        col_thresh.metric(
            "Soglia HITL",
            f"{threshold:.2f}" if isinstance(threshold, (int, float)) else str(threshold),
        )

        if pending.get("query"):
            st.code(pending["query"], language="sql")
        if pending.get("reasons"):
            st.caption("Fattori di costo: " + ", ".join(str(r) for r in pending["reasons"]))

        btn_approve, btn_reject = st.columns(2)
        if btn_approve.button("Autorizza", key="hitl_approve", type="primary"):
            graph = _get_graph(settings)
            _invoke_or_interrupt(
                graph,
                Command(resume={"approved": True}),
                st.session_state.thread_id,
            )
            st.rerun()

        if btn_reject.button("Annulla", key="hitl_reject"):
            graph = _get_graph(settings)
            _invoke_or_interrupt(
                graph,
                Command(resume={"approved": False}),
                st.session_state.thread_id,
            )
            st.rerun()


if __name__ == "__main__":
    main()
