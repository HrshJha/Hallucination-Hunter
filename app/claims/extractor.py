"""
Claim Extraction Module (DEPLOY-SAFE VERSION)
--------------------------------------------
No spaCy. Uses simple sentence splitting.
"""

from __future__ import annotations
import re
from typing import List


# ── Filters ──────────────────────────────────────────

_META_PATTERNS = re.compile(
    r"^(note:|disclaimer:|as an ai|i cannot|i'?m not sure|"
    r"please note|it'?s important to|in summary|overall|"
    r"to summarize|in conclusion)",
    re.IGNORECASE,
)


def _is_question(sent: str) -> bool:
    return sent.strip().endswith("?")


def _is_meta(sent: str) -> bool:
    return bool(_META_PATTERNS.match(sent.strip()))


def _is_too_short(sent: str, min_words: int = 3) -> bool:
    return len(sent.strip().split()) < min_words


# ── Simple sentence splitter ──────────────────────────

def _split_sentences(text: str) -> List[str]:
    # basic split on punctuation
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


# ── Public API ───────────────────────────────────────

def extract_claims(
    text: str,
    filter_questions: bool = True,
    filter_meta: bool = True,
    min_words: int = 3,
    max_claims: int | None = None,
) -> List[str]:

    sentences = _split_sentences(text)

    claims: List[str] = []
    for s in sentences:
        if not s:
            continue
        if filter_questions and _is_question(s):
            continue
        if filter_meta and _is_meta(s):
            continue
        if _is_too_short(s, min_words):
            continue
        claims.append(s)

    if max_claims is not None:
        claims = claims[: max_claims]

    return claims


def extract_source_sentences(text: str) -> List[str]:
    return _split_sentences(text)