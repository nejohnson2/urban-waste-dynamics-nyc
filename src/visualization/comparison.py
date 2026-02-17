"""
Model and cluster comparison visualization module.

Produces comparison charts across clustering methods and prediction models.

Usage:
    python -m src.visualization.comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from src.config import FIGURES_DIR, RESULTS_DIR, setup_logging

logger = setup_logging("viz_comparison")

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150


def plot_silhouette_comparison():
    """Bar chart comparing silhouette scores across methods and levels."""
    logger.info("Plotting silhouette comparison...")

    metrics_path = RESULTS_DIR / "cluster_metrics.csv"
    if not metrics_path.exists():
        logger.warning("No cluster metrics found")
        return

    metrics = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    bar_width = 0.6
    colors = sns.color_palette("Set2", len(metrics))

    bars = ax.bar(x, metrics["silhouette"], bar_width, color=colors)

    # Labels
    labels = [f"{row['method']}\n({row['level']})" for _, row in metrics.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Clustering Method Comparison — Silhouette Score")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, metrics["silhouette"]):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "silhouette_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_curves():
    """Silhouette score vs k for each method."""
    logger.info("Plotting silhouette curves...")

    import glob as glob_mod
    sil_files = list(RESULTS_DIR.glob("silhouette_*.csv"))
    if not sil_files:
        logger.warning("No silhouette curve files found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, level in zip(axes, ["district", "section"]):
        level_files = [f for f in sil_files if level in f.name]
        for f in level_files:
            data = pd.read_csv(f)
            method = data["method"].iloc[0] if "method" in data.columns else f.stem
            ax.plot(data["k"], data["silhouette"], marker="o", markersize=4, label=method)

        ax.set_xlabel("k (Number of Clusters)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title(f"{level.title()} Level")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Silhouette Score vs. k", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "silhouette_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(level: str = "district"):
    """Dendrogram from hierarchical clustering."""
    logger.info(f"Plotting dendrogram ({level})...")

    linkage_path = RESULTS_DIR / f"linkage_{level}.npy"
    ids_path = RESULTS_DIR / f"feature_ids_{level}.csv"

    if not linkage_path.exists():
        logger.warning(f"No linkage matrix for {level}")
        return

    Z = np.load(linkage_path)
    ids = pd.read_csv(ids_path).iloc[:, 0].tolist()

    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(
        Z,
        labels=ids,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        color_threshold=0,  # color by default clusters
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram — {level.title()} Level", fontsize=14)
    ax.set_xlabel(level.title())
    ax.set_ylabel("Ward Distance")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"dendrogram_{level}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison():
    """Model performance comparison — RMSE box plots by district."""
    logger.info("Plotting prediction comparison...")

    results_path = RESULTS_DIR / "prediction_results.parquet"
    if not results_path.exists():
        logger.warning("No prediction results found")
        return

    results = pd.read_parquet(results_path)
    results["abs_error"] = np.abs(results["y_true"] - results["y_pred"])

    # Compute RMSE per district per model
    rmse_by = (
        results.groupby(["model", "District_Code"])
        .apply(lambda g: np.sqrt((g["abs_error"] ** 2).mean()), include_groups=False)
        .reset_index(name="rmse")
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot of RMSE by model
    sns.boxplot(data=rmse_by, x="model", y="rmse", ax=axes[0], palette="Set2")
    axes[0].set_title("RMSE Distribution by Model")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("RMSE (Tons/Week)")

    # Overall comparison bar chart
    comp_path = RESULTS_DIR / "prediction_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path, index_col=0)

        metrics_to_plot = ["rmse", "mae", "mape"]
        available = [m for m in metrics_to_plot if m in comp.columns]

        x = np.arange(len(comp))
        width = 0.25
        for i, metric in enumerate(available):
            axes[1].bar(x + i * width, comp[metric], width, label=metric.upper())

        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(comp.index, rotation=45)
        axes[1].set_title("Overall Model Metrics")
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_stability():
    """Cluster size and composition across methods."""
    logger.info("Plotting cluster stability...")

    for level in ["district", "section"]:
        assign_path = RESULTS_DIR / f"cluster_assignments_{level}.parquet"
        if not assign_path.exists():
            continue

        assignments = pd.read_parquet(assign_path)
        cluster_cols = [c for c in assignments.columns if c.startswith("cluster_")]

        if not cluster_cols:
            continue

        fig, axes = plt.subplots(1, len(cluster_cols), figsize=(5 * len(cluster_cols), 5))
        if len(cluster_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, cluster_cols):
            counts = assignments[col].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette("Set2"))
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            method_name = col.replace("cluster_", "").replace("_", " ").title()
            ax.set_title(method_name)

        plt.suptitle(f"Cluster Sizes — {level.title()} Level", fontsize=14)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"cluster_sizes_{level}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def main():
    logger.info("=" * 60)
    logger.info("Generating comparison visualizations")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_silhouette_comparison()
    plot_silhouette_curves()

    for level in ["district", "section"]:
        plot_dendrogram(level)

    plot_prediction_comparison()
    plot_cluster_stability()

    logger.info("Comparison visualization complete")


if __name__ == "__main__":
    main()
