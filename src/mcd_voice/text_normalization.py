"""Shared text normalization helpers used across dialog and LLM layers."""

from __future__ import annotations

import re


def normalize_item_text(text: str) -> str:
    """Normalize menu/item text for fuzzy matching."""
    s = (text or "").lower()
    s = re.sub(r"[®™℠]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())
