"""Unit tests for lazy dual-backend model factory and AST SQL cost estimation."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from src.agent.cost import estimate_ast_cost, estimate_sql_cost
from src.agent.models import create_chat_model, resolve_model_config
from src.config import Settings


def test_ollama_backend_resolves_local_openai_compatible_config() -> None:
    settings = Settings(
        _env_file=None,
        llm_backend="ollama",
        ollama_base_url="http://host.docker.internal:11434/v1",
        ollama_model="qwen2.5:7b",
    )

    model_name, base_url, api_key = resolve_model_config("router", settings=settings)
    assert model_name == "qwen2.5:7b"
    assert base_url == "http://host.docker.internal:11434/v1"
    assert api_key.get_secret_value() == "ollama"

    client = create_chat_model("router", settings=settings)
    assert isinstance(client, ChatOpenAI)
    assert client.model_name == "qwen2.5:7b"


def test_openrouter_backend_requires_key_and_selects_role_models() -> None:
    missing_key_settings = Settings(
        _env_file=None,
        llm_backend="openrouter",
        openrouter_api_key=None,
    )
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        resolve_model_config("router", settings=missing_key_settings)

    configured = Settings(
        _env_file=None,
        llm_backend="openrouter",
        openrouter_api_key=SecretStr("sk-or-test-key"),
        remote_router_model="deepseek/deepseek-chat",
        remote_reasoning_model="deepseek/deepseek-r1",
    )

    router_name, router_url, router_key = resolve_model_config("router", settings=configured)
    critic_name, critic_url, critic_key = resolve_model_config("critic", settings=configured)

    assert router_name == "deepseek/deepseek-chat"
    assert critic_name == "deepseek/deepseek-r1"
    assert router_url == "https://openrouter.ai/api/v1"
    assert critic_url == "https://openrouter.ai/api/v1"
    assert router_key.get_secret_value() == "sk-or-test-key"
    assert critic_key.get_secret_value() == "sk-or-test-key"


def test_ast_cost_heuristic_distinguishes_gold_cube_from_silver_full_scan() -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)

    gold_cost = estimate_sql_cost(
        "SELECT subject_id, label, night_post_ratio FROM gold_user_longitudinal_features LIMIT 100",
        settings=settings,
    )
    assert gold_cost["method"] == "ast_heuristic"
    assert gold_cost["estimated_cost"] <= 1.0
    assert gold_cost["triggers_hitl"] is False

    silver_cost = estimate_sql_cost(
        "SELECT * FROM silver_posts LIMIT 100",
        settings=settings,
    )
    assert silver_cost["estimated_cost"] > 1.0
    assert silver_cost["triggers_hitl"] is True

    join_score, join_reasons = estimate_ast_cost(
        "SELECT a.subject_id FROM silver_posts a JOIN silver_posts b ON a.subject_id = b.subject_id"
    )
    assert join_score > 1.5
    assert any("join_operation" in reason for reason in join_reasons)


def test_hitl_cost_calibration_separates_partition_pruned_and_unfiltered_silver_posts() -> None:
    settings = Settings(_env_file=None, hitl_cost_threshold=1.0)

    pruned_year_count = estimate_sql_cost(
        "SELECT COUNT(*) FROM silver_posts WHERE year = 2014",
        settings=settings,
    )
    assert pruned_year_count["estimated_cost"] < 1.0
    assert pruned_year_count["triggers_hitl"] is False

    pruned_both = estimate_sql_cost(
        "SELECT subject_id, text FROM silver_posts WHERE label = 1 AND year = 2014",
        settings=settings,
    )
    assert pruned_both["estimated_cost"] < 1.0
    assert pruned_both["triggers_hitl"] is False

    unfiltered_star = estimate_sql_cost(
        "SELECT * FROM silver_posts",
        settings=settings,
    )
    assert unfiltered_star["estimated_cost"] > 1.0
    assert unfiltered_star["triggers_hitl"] is True

    unfiltered_count = estimate_sql_cost(
        "SELECT COUNT(*) FROM silver_posts",
        settings=settings,
    )
    assert unfiltered_count["estimated_cost"] > 1.0
    assert unfiltered_count["triggers_hitl"] is True
