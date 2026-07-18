"""Prompt helpers for AI Toolkit-style Differential Output Preservation."""

from __future__ import annotations

import re
from typing import List, Sequence


def split_caption_tokens(value: str, separator: str = ",") -> List[str]:
    """Split a caption field and discard empty, surrounding whitespace-only tokens."""

    separator = separator or ","
    return [token.strip() for token in str(value or "").split(separator) if token.strip()]


def normalize_trigger_match(value: str) -> str:
    """Normalize spelling variants that should be equivalent for trigger matching.

    Matching is case-insensitive, treats whitespace and underscores as equivalent,
    and treats escaped/unescaped parentheses as equivalent.
    """

    value = re.sub(r"\\([()])", r"\1", str(value or ""))
    value = re.sub(r"[\s_]+", "", value.casefold())
    return value


def contains_trigger_word(flex_token: str, trigger_words: Sequence[str]) -> bool:
    """Return whether a flex token contains any normalized trigger word."""

    normalized_flex = normalize_trigger_match(flex_token)
    return any(
        normalized_trigger and normalized_trigger in normalized_flex
        for normalized_trigger in (normalize_trigger_match(word) for word in trigger_words)
    )


def build_preservation_prompt(
    caption: str,
    keep_tokens: int,
    separator: str = ",",
    preservation_class: str = "",
) -> str:
    """Build a class or classless preservation prompt from a processed caption.

    The first ``keep_tokens`` caption tokens are treated as trigger words. Flex
    tokens containing a trigger spelling variant are removed. When a class is
    supplied it replaces the trigger section; otherwise only safe flex tokens
    remain.
    """

    tokens = split_caption_tokens(caption, separator)
    trigger_count = max(0, int(keep_tokens or 0))
    trigger_words = tokens[:trigger_count]
    flex_tokens = tokens[trigger_count:]
    safe_flex_tokens = [token for token in flex_tokens if not contains_trigger_word(token, trigger_words)]
    class_tokens = split_caption_tokens(preservation_class, separator) if preservation_class else []
    return f"{separator} ".join(class_tokens + safe_flex_tokens)
