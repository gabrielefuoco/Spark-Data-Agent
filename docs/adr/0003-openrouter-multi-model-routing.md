# ADR 0003: Dual-Backend Multi-Model Routing (Ollama and OpenRouter)

## Status

Accepted

## Context

The legacy implementation hardcoded Gemini 2.5 Flash with credentials loaded manually from `src/api.env`. Hardcoding a single remote commercial model creates two problems:

1. **Development friction and cost**: Every local test, prompt tweak, or interactive UI session burns paid API tokens and fails offline.
2. **Role mismatch**: Intent classification and SQL generation need fast, cheap, deterministic JSON output (`temperature=0.0`), whereas scientific critique over statistical guard outputs benefits from stronger reasoning models—yet forcing a heavy reasoning model on every routing retry wastes latency and budget.

## Decision

Standardize on a **dual-backend, role-split model factory** (`src/agent/models.py`) driven by `Settings` (`src/config.py`):

- **Single OpenAI-compatible client**: Both local Ollama (`http://localhost:11434/v1`, or `http://host.docker.internal:11434/v1` inside Docker) and OpenRouter (`https://openrouter.ai/api/v1`) expose OpenAI-compatible `/v1` chat endpoints, so one `langchain_openai.ChatOpenAI` wrapper (`create_chat_model`) handles both backends without separate SDKs.
- **Free local default (`llm_backend="ollama"`)**: Development and CI/offline runs default to local Ollama (`ollama_model="qwen2.5:7b"`) with zero token cost, and model construction is lazy until a graph node actually invokes the model.
- **Role-split remote routing (`llm_backend="openrouter"`)**: Remote execution separates the `"router"` role (`remote_router_model="deepseek/deepseek-chat"`) for low-cost, low-latency SQL planning and retries from the `"critic"` role (`remote_reasoning_model="deepseek/deepseek-r1"`) for interpreting `StatisticalGuardResult` verdicts. Both roles are configurable via environment variables without code changes.

## Consequences

- **Positive**: Zero-cost local development and deterministic offline unit tests (with `FakeMessagesListChatModel` injection), plus one-env-var switch (`LLM_BACKEND=openrouter`) for stronger remote models.
- **Positive**: Eliminates vendor-specific client sprawl (`llama-index-llms-gemini` removed) and keeps `OPENROUTER_API_KEY` encapsulated as `pydantic.SecretStr`.
- **Negative**: Assumes both backends support OpenAI-style chat completions reliably enough for the router's JSON output contract (mitigated by `_parse_router_output` fallback extraction and the `query_rewriter` retry loop).
