# Architecture Decision Records

- [ADR 0001: Medallion Lakehouse vs. Flat Parquet Tables](0001-medallion-lakehouse-vs-flat-parquet.md) — Separate Bronze, Silver (atomic posts + dead-letter quarantine), and Gold ZSTD Parquet layers instead of flat `collect_list` aggregations, backed by local disk or local MinIO (`s3a://`).
- [ADR 0002: LangGraph Cyclical StateGraph vs. LlamaIndex AgentWorkflow](0002-langgraph-cyclical-stategraph-vs-llamaindex.md) — Deterministic cyclic `StateGraph` with AST validation, `interrupt()`/`SqliteSaver` cost gate, retry loop, and statistical guard replacing single-shot LlamaIndex orchestration.
- [ADR 0003: Dual-Backend Multi-Model Routing (Ollama and OpenRouter)](0003-openrouter-multi-model-routing.md) — One OpenAI-compatible `ChatOpenAI` factory supporting free local Ollama for development and role-split OpenRouter DeepSeek models (`deepseek-chat` router vs. `deepseek-r1` critic) for remote execution.
