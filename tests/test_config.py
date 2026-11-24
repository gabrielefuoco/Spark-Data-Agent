"""Tests for the configuration layer.

These run without Spark: `Settings` is the only component that reads the
environment, which is exactly why it is worth testing.
"""

from __future__ import annotations

from src.config import Settings, get_settings


def test_defaults_are_local_and_free(monkeypatch) -> None:
    # The container sets OLLAMA_BASE_URL to the host bridge; the default under
    # test is the one that applies when nothing overrides it.
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_backend == "ollama"
    assert settings.openrouter_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.spark_master == "local[*]"


def test_environment_overrides_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    settings = Settings(_env_file=None)

    assert settings.llm_backend == "openrouter"
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-test"


def test_guard_thresholds_are_explicit() -> None:
    settings = Settings(_env_file=None)

    assert settings.min_subjects_per_class == 30
    assert settings.sql_row_limit == 500
    assert settings.max_query_retries == 2


def test_settings_are_cached(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("SQL_ROW_LIMIT", "42")

    assert get_settings().sql_row_limit == 42
    assert get_settings() is get_settings()

    get_settings.cache_clear()
