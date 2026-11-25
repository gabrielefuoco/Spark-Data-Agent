"""Tests for token-budgeted conversation memory and prompt windowing."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from src.agent.memory import TRUNCATION_MARKER, build_conversation_window, count_tokens
from src.agent.prompts import build_planner_user_prompt


def test_count_tokens_monotonic_and_empty() -> None:
    assert count_tokens("") == 0
    short = "How many posts exist in 2014?"
    medium = short + " Compare depressed and control cohorts across active days."
    long = medium + (" Additional corpus context sentence." * 10)
    assert count_tokens(short) > 0
    assert count_tokens(short) <= count_tokens(medium) <= count_tokens(long)


def test_window_keeps_newest_and_reports_dropped_under_small_budget() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="Turn 0 question about 2011 cohort post counts."),
        AIMessage(
            content="Turn 0 answer with detailed statistical summary.",
            additional_kwargs={"generated_query": "SELECT COUNT(*) FROM silver_posts WHERE year = 2011"},
        ),
        HumanMessage(content="Turn 1 question about 2012 cohort post counts."),
        AIMessage(
            content="Turn 1 answer with detailed statistical summary.",
            additional_kwargs={"generated_query": "SELECT COUNT(*) FROM silver_posts WHERE year = 2012"},
        ),
        HumanMessage(content="Turn 2 question about 2013 cohort post counts."),
        AIMessage(
            content="Turn 2 answer with detailed statistical summary.",
            additional_kwargs={"generated_query": "SELECT COUNT(*) FROM silver_posts WHERE year = 2013"},
        ),
        HumanMessage(content="Turn 3 question about 2014 cohort post counts."),
        AIMessage(
            content="Turn 3 answer with detailed statistical summary.",
            additional_kwargs={"generated_query": "SELECT COUNT(*) FROM silver_posts WHERE year = 2014"},
        ),
    ]

    # Budget is above 4x the largest single message (no truncation), below all 8 messages (~132 tokens).
    budget = 100
    window = build_conversation_window(messages, token_budget=budget)

    assert window.budget == budget
    assert window.dropped_messages > 0
    assert window.used_tokens <= budget
    assert len(window.messages) + window.dropped_messages == len(messages)
    assert window.messages[-1].content == messages[-1].content
    assert window.messages[0].content != messages[0].content

    prompt = build_planner_user_prompt("e per il 2015?", history=window)
    assert f"[{window.dropped_messages} earlier " in prompt
    assert "omitted due to token budget" in prompt
    assert "Turn 3 answer" in prompt
    assert "Turn 0 question" not in prompt


def test_five_turn_conversation_under_generous_budget_keeps_oldest_question() -> None:
    history: list[BaseMessage] = []
    for turn in range(5):
        history.append(HumanMessage(content=f"Turn {turn} unique question about metric_{turn}"))
        history.append(
            AIMessage(
                content=f"Turn {turn} full analytical answer for metric_{turn}.",
                additional_kwargs={"generated_query": f"SELECT metric_{turn} FROM gold_user_longitudinal_features"},
            )
        )

    window = build_conversation_window(history, token_budget=800_000)
    assert window.dropped_messages == 0
    assert len(window.messages) == 10
    assert 0 < window.used_tokens < 800_000

    prompt = build_planner_user_prompt("e per il 2015?", history=window)
    assert "Turn 0 unique question about metric_0" in prompt
    assert "SELECT metric_0 FROM gold_user_longitudinal_features" in prompt
    assert "omitted due to token budget" not in prompt


def test_oversized_message_truncated_to_share_and_newest_turn_survives() -> None:
    huge_answer = " ".join(f"word{i}" for i in range(400))
    messages: list[BaseMessage] = [
        HumanMessage(content="Earlier normal question."),
        AIMessage(
            content=huge_answer,
            additional_kwargs={"generated_query": "SELECT subject_id FROM gold_user_longitudinal_features"},
        ),
        HumanMessage(content="Follow-up question on the newest turn."),
        AIMessage(
            content="Newest turn concise answer.",
            additional_kwargs={"generated_query": "SELECT COUNT(*) FROM silver_posts"},
        ),
    ]

    budget = 240
    window = build_conversation_window(messages, token_budget=budget)

    assert len(window.messages) == 4
    assert window.dropped_messages == 0
    assert window.used_tokens <= budget

    truncated_ai = window.messages[1]
    assert isinstance(truncated_ai, AIMessage)
    assert TRUNCATION_MARKER in str(truncated_ai.content)
    assert truncated_ai.additional_kwargs["generated_query"] == (
        "SELECT subject_id FROM gold_user_longitudinal_features"
    )
    assert (
        count_tokens(str(truncated_ai.content))
        + count_tokens(str(truncated_ai.additional_kwargs["generated_query"]))
        <= int(budget * 0.25)
    )
    assert window.messages[-1].content == "Newest turn concise answer."


def test_newest_message_never_dropped_even_under_tiny_budget() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="Older question with enough detail to exceed four tokens."),
        AIMessage(content="Newest response " + ("extra " * 30)),
    ]
    window = build_conversation_window(messages, token_budget=4)

    assert len(window.messages) == 1
    assert window.dropped_messages == 1
    assert TRUNCATION_MARKER in str(window.messages[0].content)
