"""Central configuration.

Every external dependency is read here and nowhere else: the LLM backend, the
Spark storage root, the credentials. Components receive a `Settings` instance
instead of reaching for environment variables themselves, so a test can inject
one without touching the process environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMBackend = Literal["ollama", "openrouter"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, read from the environment or from a `.env` file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM backend -------------------------------------------------------
    # Local by default: development and tests must not cost tokens.
    llm_backend: LLMBackend = "ollama"

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Cheap, deterministic model for routing and planning.
    remote_router_model: str = "deepseek/deepseek-chat"
    # Reasoning model for the scientific critique.
    remote_reasoning_model: str = "deepseek/deepseek-r1"

    # ---- Storage -----------------------------------------------------------
    # Local path or an S3-compatible URI (s3a://bucket/prefix with MinIO).
    spark_storage_uri: str = Field(default=str(PROJECT_ROOT / "data"))
    spark_app_name: str = "spark-data-agent"
    spark_master: str = "local[*]"

    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    s3_endpoint_url: str | None = None

    # ---- Agent behaviour ---------------------------------------------------
    # Below this many subjects per class the guard marks the sample underpowered.
    min_subjects_per_class: int = 30
    # Hard ceiling on rows shown to the LLM/critic and UI from an ad-hoc query.
    sql_row_limit: int = 500
    # Hard ceiling on rows fetched into the driver for statistical analysis.
    sql_fetch_limit: int = 50_000
    # Queries estimated above this cost stop at the human-in-the-loop gate.
    hitl_cost_threshold: float = 1.0
    max_query_retries: int = 2
    # The underlying model offers 1M of context and we deliberately use 800k
    # for conversation history so system prompts and model output retain headroom.
    conversation_token_budget: int = 800_000

    @field_validator(
        "openrouter_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "s3_endpoint_url",
        mode="before",
    )
    @classmethod
    def blank_means_unset(cls, value: object) -> object:
        """Treat an empty variable as absent.

        `.env.example` ships every credential as `KEY=`, and a compose file can
        pass one through as an empty string. Without this, `Optional` fields
        become `SecretStr('')` and "is the key configured?" answers yes.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings, instantiated once."""
    return Settings()
