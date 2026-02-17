"""
Temporal clustering pipeline.

Runs multiple clustering algorithms on weekly waste time series at both
district and section levels. Compares K-Means, DTW K-Means, Hierarchical,
and DBSCAN.

Usage:
    python -m src.analysis.clustering
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw as tslearn_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from src.analysis.evaluation import compute_cluster_metrics, silhouette_by_k
from src.config import (
    DATA_PROCESSED,
    RANDOM_SEED,
    RESULTS_DIR,
    STREAM_COL_NAMES,
    set_seeds,
    setup_logging,
)

logger = setup_logging("clustering")

K_RANGE = range(2, 16)


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    group_col: str,
    stream_col: str = "tons_refuse",
) -> tuple[np.ndarray, list[str]]:
    """
    Build a feature matrix where each row is a district/section and each
    column is a weekly tonnage value.

    Parameters
    ----------
    df : weekly aggregated DataFrame
    group_col : 'District_Code' or 'Section_Code'
    stream_col : which waste stream column to use

    Returns
    -------
    X : (n_groups, n_weeks) array, z-score normalized per row
    group_ids : list of group identifiers in row order
    """
    # Pivot: rows=groups, columns=weeks
    pivot = df.pivot_table(
        index=group_col,
        columns="week_start",
        values=stream_col,
        aggfunc="sum",
        fill_value=0,
    )

    # Drop groups with too many missing weeks (< 80% coverage)
    min_weeks = int(pivot.shape[1] * 0.80)
    valid_mask = (pivot > 0).sum(axis=1) >= min_weeks
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"  Dropping {n_dropped} groups with <80% weekly coverage")
    pivot = pivot[valid_mask]

    group_ids = pivot.index.tolist()
    X = pivot.values.astype(float)

    # Z-score normalize each row (each district's time series)
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T  # normalize per row

    return X, group_ids


def build_multistream_features(
    df: pd.DataFrame,
    group_col: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a feature matrix concatenating all three waste streams.

    Returns (n_groups, n_weeks * 3) array.
    """
    matrices = []
    for stream_col in STREAM_COL_NAMES.values():
        X, group_ids = build_feature_matrix(df, group_col, stream_col)
        matrices.append(X)

    # Concatenate along feature axis
    X_combined = np.hstack(matrices)
    return X_combined, group_ids


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, k_range: range = K_RANGE) -> dict:
    """Standard K-Means with PCA initialization and silhouette analysis."""
    logger.info("  Running K-Means...")

    # PCA for initialization (retain 95% variance)
    pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    n_components = X_pca.shape[1]
    logger.info(f"    PCA: {X.shape[1]} features → {n_components} components (95% var)")

    # Evaluate k range
    sil_results = silhouette_by_k(X_pca, k_range)
    best_k = sil_results.loc[sil_results["silhouette"].idxmax(), "k"]
    logger.info(f"    Best k={best_k} (silhouette={sil_results['silhouette'].max():.3f})")

    # Final model
    km = KMeans(n_clusters=int(best_k), n_init=10, random_state=RANDOM_SEED)
    labels = km.fit_predict(X_pca)
    metrics = compute_cluster_metrics(X_pca, labels)
    metrics["best_k"] = int(best_k)
    metrics["method"] = "KMeans"

    return {
        "labels": labels,
        "metrics": metrics,
        "sil_by_k": sil_results,
        "centers": km.cluster_centers_,
        "X_pca": X_pca,
        "pca": pca,
    }


def run_dtw_kmeans(X: np.ndarray, k_range: range = K_RANGE) -> dict:
    """Time Series K-Means with DTW distance (tslearn)."""
    logger.info("  Running DTW K-Means...")

    # Reshape for tslearn: (n_samples, n_timesteps, 1)
    X_ts = X.reshape(X.shape[0], X.shape[1], 1)

    # Scale
    scaler = TimeSeriesScalerMeanVariance()
    X_ts = scaler.fit_transform(X_ts)

    # Evaluate k range (subset for speed — DTW is expensive)
    best_sil = -1
    best_k = 2
    sil_results = []

    for k in tqdm(k_range, desc="    DTW k-search"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = TimeSeriesKMeans(
                n_clusters=k,
                metric="dtw",
                max_iter=30,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            labels = model.fit_predict(X_ts)

        if len(set(labels)) >= 2:
            sil = compute_cluster_metrics(X_ts.reshape(X_ts.shape[0], -1), labels)["silhouette"]
        else:
            sil = np.nan

        sil_results.append({"k": k, "silhouette": sil})
        if not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    logger.info(f"    Best k={best_k} (silhouette={best_sil:.3f})")

    # Final model at best k
    model = TimeSeriesKMeans(
        n_clusters=best_k,
        metric="dtw",
        max_iter=50,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    labels = model.fit_predict(X_ts)
    metrics = compute_cluster_metrics(X_ts.reshape(X_ts.shape[0], -1), labels)
    metrics["best_k"] = best_k
    metrics["method"] = "DTW_KMeans"

    return {
        "labels": labels,
        "metrics": metrics,
        "sil_by_k": pd.DataFrame(sil_results),
        "centers": model.cluster_centers_,
    }


def run_hierarchical(X: np.ndarray, k_range: range = K_RANGE) -> dict:
    """Agglomerative hierarchical clustering with Ward linkage."""
    logger.info("  Running Hierarchical clustering...")

    Z = linkage(X, method="ward")

    # Evaluate k range
    best_sil = -1
    best_k = 2
    sil_results = []

    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = compute_cluster_metrics(X, labels)["silhouette"]
        sil_results.append({"k": k, "silhouette": sil})
        if not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    logger.info(f"    Best k={best_k} (silhouette={best_sil:.3f})")

    labels = fcluster(Z, t=best_k, criterion="maxclust")
    metrics = compute_cluster_metrics(X, labels)
    metrics["best_k"] = best_k
    metrics["method"] = "Hierarchical"

    return {
        "labels": labels,
        "metrics": metrics,
        "sil_by_k": pd.DataFrame(sil_results),
        "linkage": Z,
    }


def run_dbscan(X: np.ndarray) -> dict:
    """DBSCAN with automatic epsilon via k-distance plot."""
    logger.info("  Running DBSCAN...")

    # Find optimal eps using k-nearest neighbors
    k = min(5, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dists = np.sort(distances[:, -1])

    # Knee detection: find point of maximum curvature
    diffs = np.diff(k_dists)
    diffs2 = np.diff(diffs)
    knee_idx = np.argmax(diffs2) + 1 if len(diffs2) > 0 else len(k_dists) // 2
    eps = k_dists[knee_idx]
    logger.info(f"    Auto eps={eps:.3f} (knee at index {knee_idx})")

    db = DBSCAN(eps=eps, min_samples=3)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    logger.info(f"    Found {n_clusters} clusters, {n_noise} noise points")

    metrics = compute_cluster_metrics(X, labels)
    metrics["eps"] = eps
    metrics["n_noise"] = int(n_noise)
    metrics["method"] = "DBSCAN"

    return {
        "labels": labels,
        "metrics": metrics,
        "k_distances": k_dists,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def cluster_level(
    df: pd.DataFrame,
    group_col: str,
    level_name: str,
) -> pd.DataFrame:
    """
    Run all clustering methods for a single aggregation level.

    Returns DataFrame with cluster assignments for each method.
    """
    logger.info(f"\n{'='*40}")
    logger.info(f"Clustering at {level_name} level ({group_col})")

    # Build feature matrix (refuse only for primary clustering)
    X, group_ids = build_feature_matrix(df, group_col, "tons_refuse")
    logger.info(f"Feature matrix: {X.shape} (groups × weeks)")

    if X.shape[0] < 5:
        logger.warning(f"Too few groups ({X.shape[0]}) for meaningful clustering")
        return pd.DataFrame()

    # Run all methods
    results = {}
    results["kmeans"] = run_kmeans(X)
    results["dtw_kmeans"] = run_dtw_kmeans(X)
    results["hierarchical"] = run_hierarchical(X)
    results["dbscan"] = run_dbscan(X)

    # Build assignments DataFrame
    assignments = pd.DataFrame({group_col: group_ids})
    for method_name, result in results.items():
        assignments[f"cluster_{method_name}"] = result["labels"]

    # Build metrics comparison
    metrics_rows = []
    for method_name, result in results.items():
        row = result["metrics"].copy()
        row["level"] = level_name
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    logger.info(f"\n{level_name} clustering comparison:")
    logger.info(f"\n{metrics_df.to_string(index=False)}")

    # Save silhouette-by-k curves
    for method_name in ["kmeans", "dtw_kmeans", "hierarchical"]:
        if "sil_by_k" in results[method_name]:
            sil_df = results[method_name]["sil_by_k"]
            sil_df["method"] = method_name
            sil_df["level"] = level_name
            sil_path = RESULTS_DIR / f"silhouette_{level_name}_{method_name}.csv"
            sil_df.to_csv(sil_path, index=False)

    # Save linkage matrix for dendrogram visualization
    if "linkage" in results["hierarchical"]:
        linkage_path = RESULTS_DIR / f"linkage_{level_name}.npy"
        np.save(linkage_path, results["hierarchical"]["linkage"])

    return assignments, metrics_df


def main():
    set_seeds()
    logger.info("=" * 60)
    logger.info("Starting temporal clustering pipeline")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load aggregated data
    district_path = DATA_PROCESSED / "weekly_district.parquet"
    section_path = DATA_PROCESSED / "weekly_section.parquet"

    if not district_path.exists():
        logger.error(f"District data not found: {district_path}. Run aggregate.py first.")
        sys.exit(1)

    df_district = pd.read_parquet(district_path)
    df_section = pd.read_parquet(section_path)
    logger.info(f"Loaded district data: {df_district.shape}")
    logger.info(f"Loaded section data: {df_section.shape}")

    # Cluster at district level
    dist_assignments, dist_metrics = cluster_level(
        df_district, "District_Code", "district"
    )

    # Cluster at section level
    sect_assignments, sect_metrics = cluster_level(
        df_section, "Section_Code", "section"
    )

    # Save results
    dist_assignments.to_parquet(RESULTS_DIR / "cluster_assignments_district.parquet", index=False)
    sect_assignments.to_parquet(RESULTS_DIR / "cluster_assignments_section.parquet", index=False)

    all_metrics = pd.concat([dist_metrics, sect_metrics], ignore_index=True)
    all_metrics.to_csv(RESULTS_DIR / "cluster_metrics.csv", index=False)
    logger.info(f"\nAll metrics saved to {RESULTS_DIR / 'cluster_metrics.csv'}")

    # Also save the feature matrices for visualization
    for level, df_level, group_col in [
        ("district", df_district, "District_Code"),
        ("section", df_section, "Section_Code"),
    ]:
        X, ids = build_feature_matrix(df_level, group_col, "tons_refuse")
        np.save(RESULTS_DIR / f"feature_matrix_{level}.npy", X)
        pd.Series(ids).to_csv(RESULTS_DIR / f"feature_ids_{level}.csv", index=False)

    logger.info("Clustering pipeline complete")


if __name__ == "__main__":
    main()
