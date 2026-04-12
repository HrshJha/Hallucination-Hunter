"""
Claim Extraction Module (DEPLOY-SAFE VERSION)
----------------------------------------------
Uses spaCy for sentence segmentation when available,
falls back to simple regex splitting otherwise.
"""

from __future__ import annotations
import re
from typing import List

# ── Lazy spaCy loader ─────────────────────────────────────────

_nlp = None
_spacy_available: bool | None = None


def _get_nlp():
    """Load spaCy model lazily with auto-download fallback."""
    global _nlp, _spacy_available
    if _spacy_available is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            _spacy_available = True
        except Exception:
            _spacy_available = False
            _nlp = None
    return _nlp


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


# ── Sentence splitter ─────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy if available, else regex."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # Fallback: basic split on punctuation
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