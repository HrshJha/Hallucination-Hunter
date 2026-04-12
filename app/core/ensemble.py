"""
Optional Ensemble Model
-----------------------
Combines similarity score, NLI outputs, and claim-level stats
via logistic regression or XGBoost for improved classification.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from claims.extractor import extract_claims, extract_source_sentences
from core.nli import classify_claims_against_source
from core.similarity import similarity_scores

# ── Feature extraction ───────────────────────────────────────────────


def extract_features(
    source: str,
    response: str,
) -> np.ndarray:
    """
    Extract a feature vector for the ensemble classifier.

    Features (8-dim):
        0: avg_similarity
        1: min_similarity
        2: n_claims
        3: entailment_fraction
        4: contradiction_fraction
        5: neutral_fraction
        6: max_contradiction_score
        7: mean_entailment_score
    """
    claims = extract_claims(response, max_claims=50)
    source_sents = extract_source_sentences(source) or [source]

    if not claims:
        return np.array([1.0, 1.0, 0, 1.0, 0.0, 0.0, 0.0, 1.0])

    # Similarity
    avg_sim, min_sim, _matrix = similarity_scores(source_sents, claims)

    # NLI
    nli_results = classify_claims_against_source(source, claims)

    n = len(nli_results)
    n_ent = sum(1 for r in nli_results if r["label"] == "entailment")
    n_con = sum(1 for r in nli_results if r["label"] == "contradiction")
    n_neu = sum(1 for r in nli_results if r["label"] == "neutral")

    # Per-claim contradiction/entailment scores (if available from raw NLI)
    max_con_score = max(
        (r["score"] for r in nli_results if r["label"] == "contradiction"),
        default=0.0,
    )
    mean_ent_score = float(np.mean(
        [r["score"] for r in nli_results if r["label"] == "entailment"] or [0.0]
    ))

    return np.array([
        avg_sim,
        min_sim,
        n,
        n_ent / n,
        n_con / n,
        n_neu / n,
        max_con_score,
        mean_ent_score,
    ])


# ── Train ────────────────────────────────────────────────────────────

def train_ensemble(
    features: np.ndarray,
    labels: np.ndarray,
    model_type: str = "logistic",
    save_path: Optional[str] = None,
):
    """
    Train a logistic regression or XGBoost ensemble.

    Parameters
    ----------
    features : np.ndarray, shape (N, 8)
    labels : np.ndarray, shape (N,)  — 0 = faithful, 1 = hallucinated
    model_type : str
        ``"logistic"`` or ``"xgboost"``
    save_path : str | None
        Path to save the trained model pickle.

    Returns
    -------
    model
    """
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(features, labels)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(clf, f)

    return clf


# ── Predict ──────────────────────────────────────────────────────────

def ensemble_predict(
    source: str,
    response: str,
    model_path: str,
) -> Tuple[str, float]:
    """
    Run the ensemble model.

    Returns
    -------
    label, confidence
    """
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    features = extract_features(source, response).reshape(1, -1)
    prob = clf.predict_proba(features)[0]

    # prob[0] = faithful, prob[1] = hallucinated
    if prob[1] >= 0.5:
        return "HALLUCINATED", float(prob[1])
    else:
        return "FAITHFUL", float(prob[0])
