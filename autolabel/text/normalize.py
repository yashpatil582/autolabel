"""Unicode normalization and script detection utilities."""

from __future__ import annotations

import re
import unicodedata

# Devanagari Unicode block: U+0900 to U+097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def normalize_text(text: str, form: str = "NFKC") -> str:
    """Normalize Unicode text.

    Args:
        text: Input text string.
        form: Unicode normalization form — one of NFC, NFKC, NFD, NFKD.
            NFKC is recommended for Devanagari as it normalizes combining
            characters and compatibility decompositions.

    Returns:
        Normalized text string.
    """
    return unicodedata.normalize(form, text)


def contains_devanagari(text: str) -> bool:
    """Check if text contains any Devanagari script characters."""
    return bool(_DEVANAGARI_RE.search(text))
