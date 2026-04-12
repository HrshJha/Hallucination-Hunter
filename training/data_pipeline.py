"""
Data Pipeline
-------------
Load and preprocess datasets for hallucination detection:

1. HaluEval  — from RUCAIBox/HaluEval GitHub repo (direct JSON download)
   - qa_data.json       : 10K QA samples          (knowledge, right_answer, hallucinated_answer)
   - dialogue_data.json : 10K dialogue samples     (knowledge, right_response, hallucinated_response)
   - summarization_data.json : 10K summarization    (document, right_summary, hallucinated_summary)
   - general_data.json  : 5K general ChatGPT       (user_query, chatgpt_response, hallucination_label)

2. TRUE benchmark — from google-research/true
   Standardized format: grounding, generated_text, label (binary: 1=faithful, 0=hallucinated)
   Datasets: FRANK, QAGS-CNNDM, QAGS-XSum, MNBM, BEGIN, Q², DialFact, VitaminC, PAWS, SummEval, FEVER

Both are unified into: source, response, label (0=faithful, 1=hallucinated)
"""

from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════
# 1.  HaluEval  (RUCAIBox/HaluEval)
# ═══════════════════════════════════════════════════════════════════════

HALUEVAL_BASE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data"

HALUEVAL_FILES = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json",
}

# Field mappings per split
HALUEVAL_FIELDS = {
    "qa": {
        "source_key": "knowledge",
        "right_key": "right_answer",
        "hallucinated_key": "hallucinated_answer",
    },
    "dialogue": {
        "source_key": "knowledge",
        "right_key": "right_response",
        "hallucinated_key": "hallucinated_response",
    },
    "summarization": {
        "source_key": "document",
        "right_key": "right_summary",
        "hallucinated_key": "hallucinated_summary",
    },
}


def load_halueval(
    split: str = "qa",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load HaluEval data from the official GitHub repository.

    Parameters
    ----------
    split : str
        One of "qa", "dialogue", "summarization", "general".
    max_samples : int | None
        Cap per-class sample count (total ≈ 2× for paired splits).
    cache_dir : str | None
        Directory to cache downloaded JSON files. Uses ``data/cache`` if None.

    Returns
    -------
    pd.DataFrame
        Columns: source, response, label (0 = faithful, 1 = hallucinated), split_name
    """
    if split not in HALUEVAL_FILES:
        raise ValueError(f"Unknown HaluEval split: {split}. Choose from {list(HALUEVAL_FILES.keys())}")

    filename = HALUEVAL_FILES[split]
    cache_dir = cache_dir or os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    local_path = os.path.join(cache_dir, filename)

    # Download if not cached
    if not os.path.exists(local_path):
        url = f"{HALUEVAL_BASE_URL}/{filename}"
        print(f"  Downloading {url} ...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved to {local_path}")
    else:
        print(f"  Using cached {local_path}")

    with open(local_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Handle the "general" split differently
    if split == "general":
        return _parse_halueval_general(raw_data, max_samples)

    return _parse_halueval_paired(raw_data, split, max_samples)


def _parse_halueval_paired(
    raw_data: list,
    split: str,
    max_samples: Optional[int],
) -> pd.DataFrame:
    """Parse paired HaluEval data (QA, dialogue, summarization)."""
    fields = HALUEVAL_FIELDS[split]
    records: List[Dict] = []

    for item in raw_data:
        if max_samples and len(records) >= max_samples * 2:
            break

        source = item.get(fields["source_key"], "")
        right = item.get(fields["right_key"], "")
        hallucinated = item.get(fields["hallucinated_key"], "")

        if source and right:
            records.append({
                "source": source,
                "response": right,
                "label": 0,
                "split_name": f"halueval_{split}",
            })

        if source and hallucinated:
            records.append({
                "source": source,
                "response": hallucinated,
                "label": 1,
                "split_name": f"halueval_{split}",
            })

    return pd.DataFrame(records)


def _parse_halueval_general(
    raw_data: list,
    max_samples: Optional[int],
) -> pd.DataFrame:
    """
    Parse HaluEval general_data.json.
    Fields: user_query, chatgpt_response, hallucination_label (Yes/No).
    No separate source — user_query serves as context.
    """
    records: List[Dict] = []

    for item in raw_data:
        if max_samples and len(records) >= max_samples:
            break

        query = item.get("user_query", "")
        response = item.get("chatgpt_response", "")
        label_str = item.get("hallucination_label", "").strip().lower()

        if not query or not response:
            continue

        label = 1 if label_str == "yes" else 0

        records.append({
            "source": query,
            "response": response,
            "label": label,
            "split_name": "halueval_general",
        })

    return pd.DataFrame(records)


def load_halueval_all(max_samples_per_split: Optional[int] = None) -> pd.DataFrame:
    """Load and concatenate all four HaluEval splits."""
    dfs = []
    for split in HALUEVAL_FILES:
        print(f"[HaluEval] Loading {split}...")
        df = load_halueval(split=split, max_samples=max_samples_per_split)
        dfs.append(df)
        print(f"  → {len(df)} samples")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[HaluEval] Total: {len(combined)} samples")
    return combined


# ═══════════════════════════════════════════════════════════════════════
# 2.  TRUE Benchmark  (google-research/true)
# ═══════════════════════════════════════════════════════════════════════

TRUE_BASE_URL = "https://raw.githubusercontent.com/google-research/true/main"


def load_true_frank(data_split: str = "valid") -> pd.DataFrame:
    """Download and parse the FRANK dataset (TRUE benchmark)."""
    url = "https://raw.githubusercontent.com/artidoro/frank/main/data/human_annotations_sentence.json"
    print(f"  Downloading FRANK ({data_split})...")
    resp = requests.get(url, timeout=120)
    dataset = json.loads(resp.text)

    records = []
    for example in dataset:
        if example["split"] != data_split:
            continue

        summary_annotations = example["summary_sentences_annotations"]
        label = 0  # faithful
        for sentence_annotation in summary_annotations:
            sentence_labels = [
                1 if annotation[0] == "NoE" else 0
                for annotation in sentence_annotation.values()
            ]
            majority = int(round(sum(sentence_labels) / len(sentence_labels)))
            if not majority:
                label = 1  # hallucinated
                break

        records.append({
            "source": example["article"],
            "response": example["summary"],
            "label": label,
            "split_name": f"true_frank_{data_split}",
        })

    return pd.DataFrame(records)


def load_true_qags(data_source: str = "cnndm") -> pd.DataFrame:
    """Download and parse QAGS dataset (TRUE benchmark)."""
    url = f"https://raw.githubusercontent.com/W4ngatang/qags/master/data/mturk_{data_source}.jsonl"
    print(f"  Downloading QAGS-{data_source}...")
    resp = requests.get(url, timeout=120)

    records = []
    for line in resp.iter_lines():
        example = json.loads(line.decode())
        sentences_text = []
        label = 0  # faithful

        for sentence in example["summary_sentences"]:
            sentences_text.append(sentence["sentence"])
            sentence_labels = [
                1 if ann["response"] == "yes" else 0
                for ann in sentence["responses"]
            ]
            majority = int(round(sum(sentence_labels) / len(sentence_labels)))
            if not majority:
                label = 1  # hallucinated

        records.append({
            "source": example["article"],
            "response": " ".join(sentences_text),
            "label": label,
            "split_name": f"true_qags_{data_source}",
        })

    return pd.DataFrame(records)


def load_true_begin(data_split: str = "dev") -> pd.DataFrame:
    """Download and parse BEGIN dataset (TRUE benchmark)."""
    url = f"https://raw.githubusercontent.com/google/BEGIN-dataset/6f881995bca5405918f3138b04f29fdd2a1b73ae/{data_split}_05_24_21.tsv"
    print(f"  Downloading BEGIN ({data_split})...")
    df = pd.read_csv(url, sep="\t")

    records = []
    for _, row in df.iterrows():
        label = 0 if row["gold label"] == "entailment" else 1
        records.append({
            "source": str(row.get("evidence", "")),
            "response": str(row.get("response", "")),
            "label": label,
            "split_name": f"true_begin_{data_split}",
        })

    return pd.DataFrame(records)


def load_true_q2() -> pd.DataFrame:
    """Download and parse Q² dataset (TRUE benchmark)."""
    url = "https://raw.githubusercontent.com/orhonovich/q-squared/main/third_party/data/cross_annotation.csv"
    print("  Downloading Q²...")
    df = pd.read_csv(url)

    records = []
    for _, row in df.iterrows():
        for model in ["dodeca", "memnet"]:
            # In TRUE, label=1 means consistent. q2 label=1 means hallucinated.
            # TRUE inverts: 1 - label. We want 1=hallucinated, so we keep original.
            records.append({
                "source": str(row.get("knowledge", "")),
                "response": str(row.get(f"{model}_response", "")),
                "label": int(row.get(f"{model}_label", 0)),
                "split_name": "true_q2",
            })

    return pd.DataFrame(records)


def load_true_dialfact(data_split: str = "valid") -> pd.DataFrame:
    """Download and parse DialFact dataset (TRUE benchmark)."""
    url = f"https://raw.githubusercontent.com/salesforce/DialFact/master/data/{data_split}_split.jsonl"
    print(f"  Downloading DialFact ({data_split})...")
    resp = requests.get(url, timeout=120)

    records = []
    for line in resp.iter_lines():
        example = json.loads(line.decode())
        if example.get("type_label") != "factual":
            continue

        evidence_list = [ev[2] for ev in example.get("evidence_list", []) if len(ev) > 2]
        evidence_joined = " ".join(evidence_list)
        if not evidence_joined:
            continue

        label = 0 if example.get("response_label") == "SUPPORTS" else 1

        records.append({
            "source": evidence_joined,
            "response": example.get("response", ""),
            "label": label,
            "split_name": f"true_dialfact_{data_split}",
        })

    return pd.DataFrame(records)


def load_true_vitc() -> pd.DataFrame:
    """Download and parse VitaminC dataset (TRUE benchmark)."""
    import zipfile

    url = "https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip"
    print("  Downloading VitaminC...")
    resp = requests.get(url, stream=True, timeout=300)

    records = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("vitaminc/dev.jsonl") as f:
            for line in f:
                example = json.loads(line.decode())
                label = 0 if example.get("label") == "SUPPORTS" else 1
                records.append({
                    "source": example.get("evidence", ""),
                    "response": example.get("claim", ""),
                    "label": label,
                    "split_name": "true_vitc",
                })

    return pd.DataFrame(records)


def load_true_paws() -> pd.DataFrame:
    """Download and parse PAWS dataset (TRUE benchmark)."""
    import tarfile

    url = "https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz"
    print("  Downloading PAWS...")
    resp = requests.get(url, stream=True, timeout=300)

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
        f = tf.extractfile("final/dev.tsv")
        df = pd.read_csv(f, sep="\t")

    records = []
    for _, row in df.iterrows():
        records.append({
            "source": str(row.get("sentence1", "")),
            "response": str(row.get("sentence2", "")),
            "label": 1 - int(row.get("label", 0)),  # PAWS: 1=paraphrase→faithful, invert
            "split_name": "true_paws",
        })

    return pd.DataFrame(records)


def load_true_datasets(
    datasets: Optional[List[str]] = None,
    max_per_dataset: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load multiple TRUE benchmark datasets and concatenate.

    Parameters
    ----------
    datasets : list[str] | None
        Subset of dataset names to load. None = all available.
        Available: frank, qags_cnndm, qags_xsum, begin, q2, dialfact, vitc, paws
    max_per_dataset : int | None
        Cap samples per dataset.

    Returns
    -------
    pd.DataFrame with columns: source, response, label, split_name
    """
    loaders = {
        "frank": lambda: load_true_frank("valid"),
        "qags_cnndm": lambda: load_true_qags("cnndm"),
        "qags_xsum": lambda: load_true_qags("xsum"),
        "begin": lambda: load_true_begin("dev"),
        "q2": load_true_q2,
        "dialfact": lambda: load_true_dialfact("valid"),
        "vitc": load_true_vitc,
        "paws": load_true_paws,
    }

    if datasets is None:
        # Default: load the lightweight ones that don't require large downloads
        datasets = ["frank", "qags_cnndm", "qags_xsum", "begin", "q2", "dialfact"]

    dfs = []
    for name in datasets:
        if name not in loaders:
            print(f"  [WARN] Unknown TRUE dataset: {name}, skipping")
            continue
        try:
            print(f"[TRUE] Loading {name}...")
            df = loaders[name]()
            if max_per_dataset and len(df) > max_per_dataset:
                df = df.sample(n=max_per_dataset, random_state=42).reset_index(drop=True)
            dfs.append(df)
            print(f"  → {len(df)} samples (faithful={sum(df['label']==0)}, hallucinated={sum(df['label']==1)})")
        except Exception as e:
            print(f"  [ERROR] Failed to load {name}: {e}")
            continue

    if not dfs:
        return pd.DataFrame(columns=["source", "response", "label", "split_name"])

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[TRUE] Total: {len(combined)} samples")
    return combined


# ═══════════════════════════════════════════════════════════════════════
# 3.  Combined Loader
# ═══════════════════════════════════════════════════════════════════════

def load_combined(
    halueval_splits: Optional[List[str]] = None,
    true_datasets: Optional[List[str]] = None,
    max_halueval_per_split: Optional[int] = None,
    max_true_per_dataset: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load HaluEval + TRUE data together.

    Parameters
    ----------
    halueval_splits : list[str] | None
        HaluEval splits to include. None = ["qa", "dialogue", "summarization"].
    true_datasets : list[str] | None
        TRUE datasets to include. None = default lightweight set.
    max_halueval_per_split : int | None
    max_true_per_dataset : int | None

    Returns
    -------
    pd.DataFrame with columns: source, response, label, split_name
    """
    dfs = []

    # HaluEval
    if halueval_splits is None:
        halueval_splits = ["qa", "dialogue", "summarization"]

    for split in halueval_splits:
        print(f"\n{'='*60}")
        print(f"[HaluEval] Loading {split}...")
        df = load_halueval(split=split, max_samples=max_halueval_per_split)
        dfs.append(df)
        print(f"  → {len(df)} samples")

    # TRUE
    if true_datasets is not None:
        print(f"\n{'='*60}")
        df_true = load_true_datasets(datasets=true_datasets, max_per_dataset=max_true_per_dataset)
        dfs.append(df_true)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"[COMBINED] Total: {len(combined)} samples")
    print(f"  Faithful:     {sum(combined['label'] == 0)}")
    print(f"  Hallucinated: {sum(combined['label'] == 1)}")
    return combined


# ═══════════════════════════════════════════════════════════════════════
# 4.  Preprocessing & Splitting
# ═══════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop nulls, strip whitespace, drop duplicates."""
    df = df.dropna(subset=["source", "response"])
    df["source"] = df["source"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    df = df[df["source"].str.len() > 0]
    df = df[df["response"].str.len() > 0]
    df = df.drop_duplicates(subset=["source", "response"]).reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test.

    Returns
    -------
    train_df, val_df, test_df
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    val_frac = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_frac, random_state=random_state, stratify=train_df["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data",
):
    """Save splits as parquet files."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    print(f"Saved splits to {output_dir}/")
    print(f"  train = {len(train_df)}  |  val = {len(val_df)}  |  test = {len(test_df)}")


# ═══════════════════════════════════════════════════════════════════════
# 5.  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download, preprocess, and split HaluEval + TRUE data"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="halueval",
        choices=["halueval", "true", "combined"],
        help="Data source to load",
    )
    parser.add_argument(
        "--halueval-split",
        type=str,
        default="qa",
        choices=["qa", "dialogue", "summarization", "general", "all"],
        help="HaluEval split (ignored if --source=true)",
    )
    parser.add_argument(
        "--true-datasets",
        type=str,
        default=None,
        help="Comma-separated TRUE dataset names (e.g. frank,qags_cnndm,begin)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    if args.source == "halueval":
        if args.halueval_split == "all":
            df = load_halueval_all(max_samples_per_split=args.max_samples)
        else:
            print(f"Loading HaluEval ({args.halueval_split})...")
            df = load_halueval(split=args.halueval_split, max_samples=args.max_samples)

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

    print(f"\nLoaded {len(df)} samples")
    df = preprocess(df)
    print(f"After preprocessing: {len(df)} samples")

    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df, output_dir=args.output_dir)
