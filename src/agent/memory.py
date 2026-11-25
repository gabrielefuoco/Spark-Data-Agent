"""Token-budgeted conversation memory utilities over LangChain messages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage

TRUNCATION_MARKER = "[… truncated …]"
PER_MESSAGE_BUDGET_SHARE = 0.25

_CL100K_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return the cached ``cl100k_base`` tiktoken encoder."""
    global _CL100K_ENCODER
    if _CL100K_ENCODER is None:
        _CL100K_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _CL100K_ENCODER


def count_tokens(text: str) -> int:
    """Count BPE tokens in ``text`` using the cached ``cl100k_base`` encoding."""
    if not text:
        return 0
    return len(_get_encoder().encode(text, disallowed_special=()))


def extract_message_text(message: BaseMessage) -> str:
    """Extract plain text from a LangChain message across string or block payloads."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _extract_generated_query(message: BaseMessage) -> str:
    if not isinstance(message, AIMessage):
        return ""
    raw_query = message.additional_kwargs.get("generated_query")
    if isinstance(raw_query, str) and raw_query.strip():
        return raw_query.strip()
    return ""


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens, appending ``TRUNCATION_MARKER``."""
    enc = _get_encoder()
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text

    bare_marker_tokens = len(enc.encode(TRUNCATION_MARKER, disallowed_special=()))
    if len(tokens) <= bare_marker_tokens:
        return text

    marker_tokens = len(enc.encode(f"\n{TRUNCATION_MARKER}", disallowed_special=()))
    if max_tokens <= marker_tokens:
        return TRUNCATION_MARKER

    keep = max_tokens - marker_tokens
    prefix = enc.decode(tokens[:keep]).rstrip()
    if not prefix:
        return TRUNCATION_MARKER

    candidate = f"{prefix}\n{TRUNCATION_MARKER}"
    while keep > 0 and len(enc.encode(candidate, disallowed_special=())) > max_tokens:
        keep -= 1
        prefix = enc.decode(tokens[:keep]).rstrip()
        candidate = f"{prefix}\n{TRUNCATION_MARKER}" if prefix else TRUNCATION_MARKER
    return candidate


def _fit_message_to_share(
    message: BaseMessage,
    max_message_tokens: int,
) -> tuple[BaseMessage, int]:
    """Measure message tokens and truncate oversized messages to ``max_message_tokens``."""
    text = extract_message_text(message)
    query = _extract_generated_query(message)
    text_tokens = count_tokens(text)
    query_tokens = count_tokens(query)
    total_tokens = text_tokens + query_tokens

    if total_tokens <= max_message_tokens:
        return message, total_tokens

    if not query:
        truncated_text = _truncate_text_to_tokens(text, max_message_tokens)
        return message.model_copy(update={"content": truncated_text}), count_tokens(truncated_text)

    if query_tokens < max_message_tokens:
        remaining = max_message_tokens - query_tokens
        truncated_text = _truncate_text_to_tokens(text, remaining)
        return (
            message.model_copy(update={"content": truncated_text}),
            count_tokens(truncated_text) + query_tokens,
        )

    # Query alone exceeds the per-message share: preserve as much query as fits.
    if text:
        marker_tokens = count_tokens(TRUNCATION_MARKER)
        truncated_query = _truncate_text_to_tokens(query, max(0, max_message_tokens - marker_tokens))
        truncated_text = TRUNCATION_MARKER
    else:
        truncated_query = _truncate_text_to_tokens(query, max_message_tokens)
        truncated_text = ""

    updated_kwargs = {**message.additional_kwargs, "generated_query": truncated_query}
    updated_msg = message.model_copy(
        update={"content": truncated_text, "additional_kwargs": updated_kwargs}
    )
    return updated_msg, count_tokens(truncated_text) + count_tokens(truncated_query)


@dataclass(frozen=True)
class ConversationWindow:
    """Result of applying the token-budgeted sliding window to message history."""

    messages: list[BaseMessage]
    dropped_messages: int
    used_tokens: int
    budget: int


def build_conversation_window(
    messages: Sequence[BaseMessage],
    *,
    token_budget: int,
) -> ConversationWindow:
    """Keep the newest suffix of ``messages`` fitting within ``token_budget``.

    Walks from newest to oldest, truncating any single message larger than
    ``token_budget * 0.25`` with ``TRUNCATION_MARKER`` instead of evicting the
    rest of the conversation, and never drops the newest message.
    """
    if not messages:
        return ConversationWindow(
            messages=[],
            dropped_messages=0,
            used_tokens=0,
            budget=token_budget,
        )

    effective_budget = max(0, token_budget)
    max_message_tokens = int(effective_budget * PER_MESSAGE_BUDGET_SHARE)

    kept_reversed: list[BaseMessage] = []
    used_tokens = 0
    newest_idx = len(messages) - 1

    for idx in range(newest_idx, -1, -1):
        fitted_msg, msg_tokens = _fit_message_to_share(messages[idx], max_message_tokens)
        is_newest = idx == newest_idx
        if is_newest or used_tokens + msg_tokens <= effective_budget:
            kept_reversed.append(fitted_msg)
            used_tokens += msg_tokens
        else:
            break

    kept_messages = list(reversed(kept_reversed))
    return ConversationWindow(
        messages=kept_messages,
        dropped_messages=len(messages) - len(kept_messages),
        used_tokens=used_tokens,
        budget=token_budget,
    )
