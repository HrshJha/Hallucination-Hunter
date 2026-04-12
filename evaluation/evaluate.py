"""
Evaluation Script
-----------------
Evaluates the hallucination detection pipeline on test sets from
HaluEval and/or TRUE benchmark datasets.

Computes: balanced accuracy, precision, recall, F1 score.
Supports per-dataset breakdown when evaluating mixed datasets.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from core.pipeline import detect


# ── Core evaluation ──────────────────────────────────────────────────

def evaluate_pipeline(
    df: pd.DataFrame,
    max_samples: int | None = None,
) -> dict:
    """
    Run the detection pipeline on each sample and compute metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: source, response, label (0=faithful, 1=hallucinated)
        May optionally have: split_name (for per-dataset breakdown)
    max_samples : int | None

    Returns
    -------
    dict with overall metrics + per-dataset breakdown
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    y_conf: list[float] = []
    split_names: list[str] = []

    total = min(len(df), max_samples) if max_samples else len(df)

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=total, desc="Evaluating")):
        if max_samples and i >= max_samples:
            break
        try:
            result = detect(row["source"], row["response"])
            pred = 1 if result.label == "HALLUCINATED" else 0
            y_pred.append(pred)
            y_true.append(int(row["label"]))
            y_conf.append(result.confidence)
            split_names.append(row.get("split_name", "unknown"))
        except Exception as e:
            print(f"[WARN] Skipping sample {i}: {e}")
            continue

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_conf_arr = np.array(y_conf)

    # Overall metrics
    metrics = _compute_metrics(y_true_arr, y_pred_arr, y_conf_arr)
    metrics["n_samples"] = len(y_true)

    # Per-dataset breakdown
    if "split_name" in df.columns:
        per_dataset: Dict[str, dict] = {}
        unique_splits = set(split_names)

        for split in sorted(unique_splits):
            indices = [j for j, s in enumerate(split_names) if s == split]
            if len(indices) < 5:
                continue
            yt = y_true_arr[indices]
            yp = y_pred_arr[indices]
            yc = y_conf_arr[indices]
            per_dataset[split] = _compute_metrics(yt, yp, yc)
            per_dataset[split]["n_samples"] = len(indices)

        metrics["per_dataset"] = per_dataset

    # Classification report
    metrics["classification_report"] = classification_report(
        y_true_arr, y_pred_arr,
        target_names=["FAITHFUL", "HALLUCINATED"],
        zero_division=0,
    )

    return metrics


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
) -> dict:
    """Compute standard binary classification metrics."""
    unique_labels = set(y_true)

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_hallucinated": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_hallucinated": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_hallucinated": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_faithful": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_faithful": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_faithful": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # ROC AUC requires both classes present
    if len(unique_labels) >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_conf))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


# ── Pretty printing ──────────────────────────────────────────────────

def print_results(metrics: dict):
    """Format and print evaluation results."""
    print("\n" + "=" * 70)
    print("  HALLUCINATION HUNTER — EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n  Samples evaluated:       {metrics['n_samples']}")
    print(f"  Balanced Accuracy:       {metrics['balanced_accuracy']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC AUC:                 {metrics['roc_auc']:.4f}")

    print(f"\n  ┌─────────────────────┬──────────┬──────────┬──────────┐")
    print(f"  │       Class         │ Precision│  Recall  │    F1    │")
    print(f"  ├─────────────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ HALLUCINATED        │  {metrics['precision_hallucinated']:.4f}  │  {metrics['recall_hallucinated']:.4f}  │  {metrics['f1_hallucinated']:.4f}  │")
    print(f"  │ FAITHFUL            │  {metrics['precision_faithful']:.4f}  │  {metrics['recall_faithful']:.4f}  │  {metrics['f1_faithful']:.4f}  │")
    print(f"  └─────────────────────┴──────────┴──────────┴──────────┘")

    cm = np.array(metrics["confusion_matrix"])
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 FAITH  HALLUC")
    print(f"  Actual FAITH   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"  Actual HALLUC  {cm[1][0]:5d}   {cm[1][1]:5d}")

    # Per-dataset breakdown
    if "per_dataset" in metrics and metrics["per_dataset"]:
        print(f"\n  {'─' * 70}")
        print(f"  Per-Dataset Breakdown:")
        print(f"  {'─' * 70}")
        print(f"  {'Dataset':<30s} {'N':>5s}  {'BAcc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
        print(f"  {'─' * 70}")
        for name, m in sorted(metrics["per_dataset"].items()):
            print(
                f"  {name:<30s} {m['n_samples']:5d}"
                f"  {m['balanced_accuracy']:.4f}"
                f"  {m['precision_hallucinated']:.4f}"
                f"  {m['recall_hallucinated']:.4f}"
                f"  {m['f1_hallucinated']:.4f}"
            )

    print(f"\n{'=' * 70}\n")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detection pipeline on HaluEval / TRUE data"
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["halueval", "true", "combined"],
        help="If set, download fresh data instead of using cached parquet",
    )
    parser.add_argument(
        "--halueval-split", type=str, default="qa",
        choices=["qa", "dialogue", "summarization", "general", "all"],
    )
    parser.add_argument("--true-datasets", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save metrics JSON")
    args = parser.parse_args()

    # Load data
    data_path = os.path.join(args.data_dir, f"{args.split}.parquet")

    if args.source:
        # Download fresh
        sys.path.insert(0, PROJECT_ROOT)
        from training.data_pipeline import (
            load_halueval, load_halueval_all, load_true_datasets,
            load_combined, preprocess,
        )

        if args.source == "halueval":
            if args.halueval_split == "all":
                df = load_halueval_all(max_samples_per_split=args.max_samples)
            else:
                df = load_halueval(split=args.halueval_split, max_samples=args.max_samples)
        elif args.source == "true":
            true_ds = args.true_datasets.split(",") if args.true_datasets else None
            df = load_true_datasets(datasets=true_ds, max_per_dataset=args.max_samples)
        else:
            true_ds = args.true_datasets.split(",") if args.true_datasets else ["frank", "qags_cnndm"]
            df = load_combined(
                halueval_splits=[args.halueval_split] if args.halueval_split != "all" else None,
                true_datasets=true_ds,
                max_halueval_per_split=args.max_samples,
                max_true_per_dataset=args.max_samples,
            )
        df = preprocess(df)
    elif os.path.exists(data_path):
        print(f"Loading {data_path}...")
        df = pd.read_parquet(data_path)
    else:
        print(f"{data_path} not found.")
        print("Run: python -m training.data_pipeline --source combined")
        print("Or use: --source halueval / --source true to download fresh")
        sys.exit(1)

    print(f"\nEvaluating on {min(len(df), args.max_samples)} samples...\n")
    metrics = evaluate_pipeline(df, max_samples=args.max_samples)
    print_results(metrics)

    if args.output:
        # Remove non-serializable items
        save_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(save_metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
