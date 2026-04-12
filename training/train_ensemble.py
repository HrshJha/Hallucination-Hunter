"""
Training script for the ensemble classifier.
Extracts features from HaluEval + TRUE datasets and trains
a logistic regression or XGBoost model.
"""

from __future__ import annotations

import os
import sys

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.core.ensemble import extract_features, train_ensemble
from training.data_pipeline import (
    load_halueval,
    load_halueval_all,
    load_true_datasets,
    load_combined,
    preprocess,
    split_data,
    save_splits,
)


def extract_dataset_features(
    df: pd.DataFrame,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors for each sample in the DataFrame.

    Returns
    -------
    features : np.ndarray, shape (N, 8)
    labels : np.ndarray, shape (N,)
    """
    features_list = []
    labels_list = []

    rows = df.iterrows()
    total = min(len(df), max_samples) if max_samples else len(df)

    for i, (_, row) in enumerate(tqdm(rows, total=total, desc="Extracting features")):
        if max_samples and i >= max_samples:
            break
        try:
            feat = extract_features(row["source"], row["response"])
            features_list.append(feat)
            labels_list.append(row["label"])
        except Exception as e:
            print(f"[WARN] Skipping sample {i}: {e}")
            continue

    return np.array(features_list), np.array(labels_list)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train ensemble classifier on HaluEval + TRUE")
    parser.add_argument(
        "--source",
        type=str,
        default="halueval",
        choices=["halueval", "true", "combined"],
        help="Data source for training",
    )
    parser.add_argument(
        "--halueval-split", type=str, default="qa",
        choices=["qa", "dialogue", "summarization", "general", "all"],
    )
    parser.add_argument(
        "--true-datasets", type=str, default=None,
        help="Comma-separated TRUE dataset names (e.g. frank,qags_cnndm,begin)",
    )
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples for feature extraction")
    parser.add_argument("--model-type", type=str, default="logistic", choices=["logistic", "xgboost"])
    parser.add_argument("--output", type=str, default="models/ensemble.pkl")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    # Load data
    data_path = os.path.join(args.data_dir, "train.parquet")

    if os.path.exists(data_path):
        print(f"Loading cached training data from {data_path}")
        train_df = pd.read_parquet(data_path)
    else:
        print(f"No cached data found. Downloading from --source={args.source}...")

        if args.source == "halueval":
            if args.halueval_split == "all":
                df = load_halueval_all(max_samples_per_split=args.max_samples)
            else:
                df = load_halueval(split=args.halueval_split, max_samples=args.max_samples * 2)
        elif args.source == "true":
            true_ds = args.true_datasets.split(",") if args.true_datasets else None
            df = load_true_datasets(datasets=true_ds, max_per_dataset=args.max_samples)
        elif args.source == "combined":
            true_ds = args.true_datasets.split(",") if args.true_datasets else ["frank", "qags_cnndm", "begin"]
            halueval_splits = (
                ["qa", "dialogue", "summarization"]
                if args.halueval_split == "all"
                else [args.halueval_split]
            )
            df = load_combined(
                halueval_splits=halueval_splits,
                true_datasets=true_ds,
                max_halueval_per_split=args.max_samples,
                max_true_per_dataset=args.max_samples,
            )

        df = preprocess(df)
        train_df, val_df, test_df = split_data(df)
        save_splits(train_df, val_df, test_df, output_dir=args.data_dir)

    # Extract features
    n_extract = min(len(train_df), args.max_samples)
    print(f"\nExtracting features from {n_extract} samples...")
    X, y = extract_dataset_features(train_df, max_samples=args.max_samples)
    print(f"Feature matrix: {X.shape}, Labels: {y.shape}")
    print(f"Label distribution: faithful={sum(y == 0)}, hallucinated={sum(y == 1)}")

    # Train
    print(f"\nTraining {args.model_type} classifier...")
    clf = train_ensemble(X, y, model_type=args.model_type, save_path=args.output)
    print(f"Model saved to {args.output}")

    # Quick accuracy on training set
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = clf.predict(X)
    print(f"\nTraining accuracy: {accuracy_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred, target_names=["FAITHFUL", "HALLUCINATED"]))


if __name__ == "__main__":
    main()
