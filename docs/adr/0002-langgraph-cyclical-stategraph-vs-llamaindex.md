# ADR 0002: LangGraph Cyclical StateGraph vs. LlamaIndex AgentWorkflow

## Status

Accepted

## Context

The previous agent (deleted in the rewrite; `git show e2976dc:legacy/agent_handler.py`) used LlamaIndex `AgentWorkflow` as a thin pass-through over a single LLM: one prompt went in, one tool ran, and an unverified answer came out. It lacked:

- **SQL injection / safety guarantees**: LLM-generated queries were not AST-checked or row-bounded before reaching Spark.
- **Self-correction**: Spark analysis errors or empty results terminated in raw tracebacks instead of structured retries.
- **Cost control**: Expensive full-table scans on raw posts ran without user confirmation.
- **Statistical discipline**: The LLM was free to over-interpret small samples (`N < 30`) or non-significant differences as clinical depression signals.

## Decision

Replace LlamaIndex `AgentWorkflow` with a deterministic **cyclical LangGraph `StateGraph`** (`src/agent/graph.py`) backed by `SqliteSaver` checkpointing:

1. **Deterministic routing & retry cycle**: `intent_router_planner` routes catalog questions directly to `gold_dataset_manifest.json` (zero Spark jobs) and analytical questions to `ast_validator_and_cost` (`sqlglot` AST allowlist + `LIMIT <= 500` enforcement + `EXPLAIN COST` / AST heuristic scoring). Errors (`SafeSQLError` or `EMPTY_RESULT`) route through `query_rewriter` back to the planner up to `Settings.max_query_retries` (default `2`) before terminal `handle_failure`.
2. **Runtime `interrupt()` + `SqliteSaver` HITL gate**: Rather than LangGraph 0.x compile-time `interrupt_before` (which pauses unconditionally before a node regardless of query cost), `hitl_gate` calls LangGraph 1.x's dynamic `langgraph.types.interrupt(...)` *only* when `sql_cost_estimate["triggers_hitl"]` is true (`estimated_cost > Settings.hitl_cost_threshold`). `SqliteSaver` persists the checkpoint across process/UI restarts until the Streamlit UI resumes the same `thread_id` with `Command(resume={"approved": True | False})`.
3. **Non-LLM statistical significance guard**: Every query result passes through `statistical_guard` (`src/agent/guards.py`), which enforces `N >= 30` per cohort, selects Welch's t-test or Mann-Whitney U via normality screening, computes Cohen's *d*, and injects mandatory guardrail constraints into `scientific_critic`'s system prompt.

## Consequences

- **Positive**: Execution order, SQL safety, retry bounds, HITL pausing, and statistical bounds are enforced deterministically in graph edges and Python code rather than prompt wishes.
- **Positive**: All nodes accept injected dependencies (`router_model`, `critic_model`, `sql_executor`, `checkpointer`), making the full cycle and HITL resume flow unit-testable offline without LLMs or Spark.
- **Negative**: Explicit state schemas (`DataAgentState`), JSON-safe serialization for SQLite checkpoints, and conditional edge wiring add more code than a one-file `AgentWorkflow`, and require keeping `langgraph >= 1.0` with `langgraph-checkpoint-sqlite >= 2.0`.
