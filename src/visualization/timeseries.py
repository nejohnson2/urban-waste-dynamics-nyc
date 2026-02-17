"""
Time series visualization module.

All functions read saved results/data and produce publication-ready figures.
Separated from analysis per pipeline design.

Usage:
    python -m src.visualization.timeseries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import STL

from src.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    RESULTS_DIR,
    STREAM_COL_NAMES,
    setup_logging,
)

logger = setup_logging("viz_timeseries")

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
FIGSIZE = (14, 6)
DPI = 150


def plot_citywide_totals(df: pd.DataFrame):
    """Weekly citywide tonnage by waste stream."""
    logger.info("Plotting citywide totals...")
    totals = df.groupby("week_start")[list(STREAM_COL_NAMES.values()) + ["tons_total"]].sum()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Total
    axes[0].plot(totals.index, totals["tons_total"], linewidth=0.8, color="black")
    axes[0].set_ylabel("Tons / Week")
    axes[0].set_title("NYC Total Waste Collection — Weekly Tonnage")

    # By stream
    for col, label in [
        ("tons_refuse", "Refuse"),
        ("tons_paper", "Paper"),
        ("tons_mgp", "MGP"),
    ]:
        axes[1].plot(totals.index, totals[col], linewidth=0.8, label=label)
    axes[1].set_ylabel("Tons / Week")
    axes[1].set_title("By Waste Stream")
    axes[1].legend()
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "citywide_totals.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_borough_comparison(df: pd.DataFrame):
    """Faceted time series by borough."""
    logger.info("Plotting borough comparison...")

    # Need borough info — derive from district code
    from src.config import extract_borough
    df = df.copy()
    df["borough"] = df["District_Code"].apply(extract_borough)

    borough_weekly = (
        df.groupby(["borough", "week_start"])["tons_refuse"]
        .sum()
        .reset_index()
    )
    boroughs = sorted(borough_weekly["borough"].unique())
    n_boroughs = len(boroughs)

    fig, axes = plt.subplots(n_boroughs, 1, figsize=(14, 3 * n_boroughs), sharex=True)
    if n_boroughs == 1:
        axes = [axes]

    for ax, borough in zip(axes, boroughs):
        data = borough_weekly[borough_weekly["borough"] == borough]
        ax.plot(data["week_start"], data["tons_refuse"], linewidth=0.6)
        ax.set_ylabel("Tons")
        ax.set_title(borough)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.suptitle("Weekly Refuse Tonnage by Borough", y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "borough_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_decomposition(df: pd.DataFrame, n_districts: int = 4):
    """STL decomposition for representative districts."""
    logger.info("Plotting seasonal decomposition...")
    districts = df["District_Code"].unique()
    # Pick evenly spaced districts
    selected = districts[np.linspace(0, len(districts) - 1, n_districts, dtype=int)]

    fig, axes = plt.subplots(n_districts, 4, figsize=(20, 4 * n_districts))

    for i, district in enumerate(selected):
        data = df[df["District_Code"] == district].sort_values("week_start")
        ts = data.set_index("week_start")["tons_refuse"]
        ts = ts.asfreq("W-MON")
        ts = ts.interpolate()

        if len(ts) < 104:
            continue

        stl = STL(ts, period=52, robust=True)
        result = stl.fit()

        for j, (component, label) in enumerate([
            (ts, "Observed"),
            (result.trend, "Trend"),
            (result.seasonal, "Seasonal"),
            (result.resid, "Residual"),
        ]):
            axes[i, j].plot(component.index, component.values, linewidth=0.5)
            if i == 0:
                axes[i, j].set_title(label, fontsize=12)
            if j == 0:
                axes[i, j].set_ylabel(district, fontsize=10, rotation=0, labelpad=60)

    plt.suptitle("STL Decomposition — Representative Districts", y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "seasonal_decomposition.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_centers(level: str = "district"):
    """Plot cluster center time series with member trajectories."""
    logger.info(f"Plotting cluster centers ({level})...")

    assignments = pd.read_parquet(RESULTS_DIR / f"cluster_assignments_{level}.parquet")
    X = np.load(RESULTS_DIR / f"feature_matrix_{level}.npy")
    ids = pd.read_csv(RESULTS_DIR / f"feature_ids_{level}.csv").iloc[:, 0].tolist()

    # Use K-Means clusters
    cluster_col = "cluster_kmeans"
    if cluster_col not in assignments.columns:
        logger.warning(f"No {cluster_col} in assignments")
        return

    labels = assignments[cluster_col].values
    n_clusters = len(set(labels))

    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for k in range(n_clusters):
        row, col = divmod(k, n_cols)
        ax = axes[row, col]
        mask = labels == k
        members = X[mask]

        # Plot individual members (transparent)
        for ts in members:
            ax.plot(ts, color="steelblue", alpha=0.15, linewidth=0.5)

        # Plot center (mean)
        center = members.mean(axis=0)
        ax.plot(center, color="darkred", linewidth=2)
        ax.set_title(f"Cluster {k} (n={mask.sum()})")
        ax.set_xlabel("Week")
        ax.set_ylabel("Z-score")

    # Hide unused subplots
    for k in range(n_clusters, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle(f"K-Means Cluster Centers — {level.title()} Level", y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"cluster_centers_{level}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_heatmap(level: str = "district"):
    """Weekly pattern heatmap by cluster (week_of_year x cluster)."""
    logger.info(f"Plotting cluster heatmaps ({level})...")

    group_col = "District_Code" if level == "district" else "Section_Code"
    df = pd.read_parquet(DATA_PROCESSED / f"weekly_{level}.parquet")
    assignments = pd.read_parquet(RESULTS_DIR / f"cluster_assignments_{level}.parquet")

    cluster_col = "cluster_kmeans"
    if cluster_col not in assignments.columns:
        return

    # Merge clusters
    merged = df.merge(assignments[[group_col, cluster_col]], on=group_col)
    merged["week_of_year"] = merged["week_start"].dt.isocalendar().week.astype(int)

    # Average tonnage by cluster × week_of_year
    heatmap_data = (
        merged.groupby([cluster_col, "week_of_year"])["tons_refuse"]
        .mean()
        .reset_index()
        .pivot(index=cluster_col, columns="week_of_year", values="tons_refuse")
    )

    fig, ax = plt.subplots(figsize=(16, max(4, len(heatmap_data) * 0.8)))
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean Tons"})
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Seasonal Pattern by Cluster — {level.title()} Level")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"cluster_heatmap_{level}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_comparison():
    """Actual vs predicted for each model."""
    logger.info("Plotting forecast comparison...")

    results_path = RESULTS_DIR / "prediction_results.parquet"
    if not results_path.exists():
        logger.warning("No prediction results found")
        return

    results = pd.read_parquet(results_path)
    models = results["model"].unique()

    # Pick a representative district (highest tonnage)
    district_totals = results.groupby("District_Code")["y_true"].sum()
    top_district = district_totals.idxmax()

    fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        subset = results[
            (results["model"] == model) & (results["District_Code"] == top_district)
        ].sort_values("week_start")

        ax.plot(subset["week_start"], subset["y_true"], label="Actual", linewidth=1.2)
        ax.plot(subset["week_start"], subset["y_pred"], label=model, linewidth=1.2, linestyle="--")
        ax.set_ylabel("Tons")
        ax.set_title(f"{model} — {top_district}")
        ax.legend()

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.suptitle("Forecast Comparison — Top District by Volume", y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_year_over_year(df: pd.DataFrame, n_districts: int = 4):
    """Year-over-year overlay for selected districts."""
    logger.info("Plotting year-over-year overlays...")
    districts = df["District_Code"].unique()
    selected = districts[np.linspace(0, len(districts) - 1, n_districts, dtype=int)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cmap = plt.cm.viridis
    for ax, district in zip(axes, selected):
        data = df[df["District_Code"] == district].copy()
        data["week_of_year"] = data["week_start"].dt.isocalendar().week.astype(int)
        data["year"] = data["week_start"].dt.year

        years = sorted(data["year"].unique())
        colors = cmap(np.linspace(0.2, 0.9, len(years)))

        for year, color in zip(years, colors):
            yr_data = data[data["year"] == year].sort_values("week_of_year")
            ax.plot(yr_data["week_of_year"], yr_data["tons_refuse"],
                    alpha=0.5, linewidth=0.7, color=color)

        ax.set_title(district)
        ax.set_xlabel("Week of Year")
        ax.set_ylabel("Tons Refuse")

    plt.suptitle("Year-over-Year Weekly Patterns", y=1.01, fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "year_over_year.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    logger.info("=" * 60)
    logger.info("Generating time series visualizations")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    district_path = DATA_PROCESSED / "weekly_district.parquet"
    if not district_path.exists():
        logger.error(f"Data not found: {district_path}")
        return

    df = pd.read_parquet(district_path)

    plot_citywide_totals(df)
    plot_borough_comparison(df)
    plot_seasonal_decomposition(df)
    plot_year_over_year(df)

    # Cluster visualizations (depend on clustering results)
    for level in ["district", "section"]:
        if (RESULTS_DIR / f"cluster_assignments_{level}.parquet").exists():
            plot_cluster_centers(level)
            plot_cluster_heatmap(level)

    # Forecast visualizations
    plot_forecast_comparison()

    logger.info("Time series visualization complete")


if __name__ == "__main__":
    main()
