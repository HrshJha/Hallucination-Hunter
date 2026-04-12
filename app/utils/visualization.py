"""
Visualization utilities for alignment matrices.
"""

from __future__ import annotations

import io
from typing import List, Optional

import numpy as np


def plot_alignment_matrix(
    matrix: List[List[float]],
    claim_labels: List[str],
    source_labels: List[str],
    title: str = "Claim ↔ Source Alignment Matrix",
    save_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    Plot the cosine similarity matrix as a heatmap.

    Parameters
    ----------
    matrix : list[list[float]]
        Shape (n_claims, n_source_sents).
    claim_labels : list[str]
        Claim text (truncated for display).
    source_labels : list[str]
        Source sentence text (truncated for display).
    title : str
    save_path : str | None
        If provided, saves the figure to this path.

    Returns
    -------
    bytes | None
        PNG image bytes if save_path is None.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    arr = np.array(matrix)

    # Truncate labels for readability
    max_len = 40
    y_labels = [c[:max_len] + "…" if len(c) > max_len else c for c in claim_labels]
    x_labels = [s[:max_len] + "…" if len(s) > max_len else s for s in source_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), max(5, len(y_labels) * 0.6)))

    sns.heatmap(
        arr,
        xticklabels=x_labels,
        yticklabels=y_labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Source Sentences", fontsize=11)
    ax.set_ylabel("Response Claims", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
