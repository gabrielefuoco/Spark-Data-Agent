"""Lazy dual-backend chat model factory for Ollama and OpenRouter.

Both backends expose OpenAI-compatible ``/v1`` chat endpoints, so one
``ChatOpenAI`` client covers offline local development (Ollama) and remote
multi-model routing/reasoning (OpenRouter). Construction is deferred until a
graph node actually invokes a model, keeping imports and tests completely free
of network or API-key requirements.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import Settings, get_settings

ModelRole = Literal["router", "critic"]


def resolve_model_config(
    role: ModelRole = "router",
    *,
    settings: Settings | None = None,
) -> tuple[str, str, SecretStr]:
    """Return ``(model_name, base_url, api_key)`` for the configured backend.

    Raises ``ValueError`` when ``llm_backend='openrouter'`` without an API key,
    or when an unknown backend is specified.
    """
    cfg = settings or get_settings()

    if cfg.llm_backend == "ollama":
        # The OpenAI client requires a non-empty key string even for local Ollama.
        return cfg.ollama_model, cfg.ollama_base_url, SecretStr("ollama")

    if cfg.llm_backend == "openrouter":
        if cfg.openrouter_api_key is None or not cfg.openrouter_api_key.get_secret_value().strip():
            raise ValueError(
                "OPENROUTER_API_KEY must be set when LLM_BACKEND='openrouter'."
            )
        model_name = (
            cfg.remote_router_model if role == "router" else cfg.remote_reasoning_model
        )
        return model_name, cfg.openrouter_base_url, cfg.openrouter_api_key

    raise ValueError(f"Unsupported llm_backend: {cfg.llm_backend!r}")


def create_chat_model(
    role: ModelRole = "router",
    *,
    settings: Settings | None = None,
    temperature: float | None = None,
    timeout: float = 60.0,
) -> ChatOpenAI:
    """Instantiate a ``ChatOpenAI`` client lazily for ``router`` or ``critic``."""
    cfg = settings or get_settings()
    model_name, base_url, api_key = resolve_model_config(role, settings=cfg)
    resolved_temperature = 0.0 if temperature is None else temperature

    headers = (
        {
            "HTTP-Referer": "https://github.com/spark-data-agent",
            "X-Title": "spark-data-agent",
        }
        if cfg.llm_backend == "openrouter"
        else None
    )

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=resolved_temperature,
        timeout=timeout,
        default_headers=headers,
    )


def build_model_factory(
    role: ModelRole,
    *,
    settings: Settings | None = None,
    temperature: float | None = None,
) -> Callable[[], BaseChatModel]:
    """Return a zero-arg factory that builds the chat model only on first call."""

    def _factory() -> BaseChatModel:
        return create_chat_model(role, settings=settings, temperature=temperature)

    return _factory
