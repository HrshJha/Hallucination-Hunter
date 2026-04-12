"""
Claim Extraction Module
-----------------------
Breaks an AI-generated response into atomic claims using spaCy
sentence segmentation, with optional filtering of questions and meta-text.
"""

from __future__ import annotations

import re
from typing import List

import spacy
from spacy.language import Language

from configs.settings import settings

# ── Lazy-loaded spaCy model ─────────────────────────────────────────

_nlp: Language | None = None


def _get_nlp() -> Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(settings.model.spacy_model)
        except OSError:
            # Auto-download if missing (handy for Colab / Docker first-run)
            from spacy.cli import download
            download(settings.model.spacy_model)
            _nlp = spacy.load(settings.model.spacy_model)
    return _nlp


# ── Filters ──────────────────────────────────────────────────────────

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


# ── Public API ───────────────────────────────────────────────────────

def extract_claims(
    text: str,
    filter_questions: bool = True,
    filter_meta: bool = True,
    min_words: int = 3,
    max_claims: int | None = None,
) -> List[str]:
    """
    Extract atomic claims from *text*.

    Parameters
    ----------
    text : str
        The AI-generated response.
    filter_questions : bool
        Drop sentences that end with '?'.
    filter_meta : bool
        Drop sentences that match common meta-text patterns.
    min_words : int
        Minimum word count to keep a sentence.
    max_claims : int | None
        Cap the number of returned claims. ``None`` → no cap.

    Returns
    -------
    list[str]
        Cleaned list of claim strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    claims: List[str] = []
    for sent in doc.sents:
        s = sent.text.strip()
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
    """Split the source passage into sentences (no filtering)."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
