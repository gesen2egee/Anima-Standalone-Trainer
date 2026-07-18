"""Class-word helpers for AI Toolkit-style Differential Output Preservation."""

from __future__ import annotations

import re
from typing import List


CLASS_TAG_PATTERN = re.compile(r"(?:1(?:girl|boy|other)|[2-6](?:boys|girls|others))", re.IGNORECASE)


def split_caption_tokens(value: str, separator: str = ",") -> List[str]:
    """Split a caption field and discard empty, surrounding whitespace-only tokens."""

    separator = separator or ","
    return [token.strip() for token in str(value or "").split(separator) if token.strip()]


def find_caption_class_tokens(caption: str, separator: str = ",") -> List[str]:
    """Return supported class tags found as complete caption tokens."""

    matches: List[str] = []
    seen = set()
    for token in split_caption_tokens(caption, separator):
        if not CLASS_TAG_PATTERN.fullmatch(token):
            continue
        normalized = token.casefold()
        if normalized not in seen:
            matches.append(normalized)
            seen.add(normalized)
    return matches


def build_preservation_prompt(
    caption: str,
    separator: str = ",",
    class_tokens: str = "",
) -> str:
    """Build a class-only prompt, falling back to an empty BPP-style prompt.

    An explicitly configured dataset class takes precedence. When it is empty,
    supported person-count tags are detected from the processed caption. Flex
    tokens are never included in a preservation prompt. If neither source has a
    class, the empty prompt is returned for unconditional preservation.
    """

    configured_tokens = split_caption_tokens(class_tokens, separator)
    resolved_tokens = configured_tokens or find_caption_class_tokens(caption, separator)
    return f"{separator} ".join(resolved_tokens)
