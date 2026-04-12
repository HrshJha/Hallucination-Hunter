"""
NLI Engine (Core)
-----------------
Cross-encoder NLI using ``cross-encoder/nli-deberta-v3-small``.
Classifies each (claim, source) pair as entailment / contradiction / neutral.

All heavy imports (torch, transformers) are deferred to first use so that
FastAPI can bind the port immediately on Render without timing out.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from app.configs.settings import settings

# ── Label map (built lazily to avoid import-time side-effects) ───────

_LABEL_MAP: Dict[int, str] | None = None


def _get_label_map() -> Dict[int, str]:
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _LABEL_MAP = {
            settings.threshold.contradiction_idx: "contradiction",
            settings.threshold.entailment_idx: "entailment",
            settings.threshold.neutral_idx: "neutral",
        }
    return _LABEL_MAP


# ── Lazy-loaded model + tokenizer ────────────────────────────────────

_tokenizer = None
_model = None
_torch = None  # lazy reference to the torch module


def _load():
    global _tokenizer, _model, _torch
    if _tokenizer is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _torch = torch
        _tokenizer = AutoTokenizer.from_pretrained(settings.model.nli_model)
        _model = AutoModelForSequenceClassification.from_pretrained(settings.model.nli_model)
        _model.to(settings.inference.device)
        _model.eval()


# ── Public API ───────────────────────────────────────────────────────

def classify_pairs(
    pairs: List[Tuple[str, str]],
    batch_size: int | None = None,
) -> List[Dict]:
    """
    Run NLI on a batch of ``(premise, hypothesis)`` pairs.

    Parameters
    ----------
    pairs : list of (premise, hypothesis)
        *premise* = source text, *hypothesis* = claim.
    batch_size : int | None
        Override default batch size.

    Returns
    -------
    list[dict]
        Each dict: ``{"label": str, "scores": {str: float}}``.
    """
    _load()
    label_map = _get_label_map()

    if batch_size is None:
        batch_size = settings.inference.nli_batch_size

    results: List[Dict] = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start: start + batch_size]
        premises = [p for p, _ in batch]
        hypotheses = [h for _, h in batch]

        encoded = _tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(settings.inference.device)

        with _torch.no_grad():
            logits = _model(**encoded).logits
            probs = _torch.softmax(logits, dim=-1).cpu().numpy()

        for prob_row in probs:
            label_idx = int(np.argmax(prob_row))
            results.append({
                "label": label_map[label_idx],
                "scores": {
                    "contradiction": float(prob_row[settings.threshold.contradiction_idx]),
                    "entailment": float(prob_row[settings.threshold.entailment_idx]),
                    "neutral": float(prob_row[settings.threshold.neutral_idx]),
                },
            })

    return results


def classify_claims_against_source(
    source: str,
    claims: List[str],
) -> List[Dict]:
    """
    For each claim, pair it with the full source and run NLI.

    Returns
    -------
    list[dict]
        ``[{"text": claim, "label": ..., "score": ..., "entailment_prob": ...}, ...]``
    """
    pairs = [(source, claim) for claim in claims]
    raw = classify_pairs(pairs)

    out: List[Dict] = []
    for claim_text, result in zip(claims, raw):
        out.append({
            "text": claim_text,
            "label": result["label"],
            "score": result["scores"][result["label"]],
            "entailment_prob": result["scores"]["entailment"],
        })
    return out
