def plot_alignment_matrix(
    matrix,
    claim_labels,
    source_labels,
    title="Semantic Alignment Heatmap",
    save_path=None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        import logging
        logging.warning("Visualization libraries (matplotlib/seaborn) are not installed. Skipping plot.")
        return None

    import numpy as np
    import io

    arr = np.array(matrix)

    # 🔥 Truncate labels
    def truncate(text, n=35):
        return text[:n] + "…" if len(text) > n else text

    y_labels = [truncate(c) for c in claim_labels]
    x_labels = [truncate(s) for s in source_labels]

    # 🔥 DARK THEME
    plt.style.use("dark_background")

    fig, ax = plt.subplots(
        figsize=(max(10, len(x_labels) * 1.5), max(6, len(y_labels) * 0.8))
    )

    # 🔥 CRAZY COLOR MAP (neon gradient)
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    heatmap = sns.heatmap(
        arr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="#1f2937",
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar=True,
        annot_kws={"size": 9, "weight": "bold"},
        ax=ax,
    )

    # 🔥 TITLE + LABELS
    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        color="#00F5D4",
        pad=20
    )

    ax.set_xlabel("Source Sentences", fontsize=11, color="#9CA3AF")
    ax.set_ylabel("Extracted Claims", fontsize=11, color="#9CA3AF")

    # 🔥 TICKS STYLE
    plt.xticks(rotation=45, ha="right", fontsize=8, color="#D1D5DB")
    plt.yticks(fontsize=8, color="#D1D5DB")

    # 🔥 COLORBAR STYLE
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(colors="#D1D5DB")
    cbar.set_label("Similarity Score", color="#D1D5DB")

    # 🔥 GLOW EFFECT
    for spine in ax.spines.values():
        spine.set_edgecolor("#00F5D4")
        spine.set_linewidth(1.2)

    # 🔥 GRID BACKGROUND
    ax.set_facecolor("#020617")
    fig.patch.set_facecolor("#020617")

    plt.tight_layout()

    # 🔥 OUTPUT
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()