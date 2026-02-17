"""
Shared evaluation metrics for clustering and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def compute_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute standard clustering evaluation metrics.

    Parameters
    ----------
    X : feature matrix (n_samples, n_features)
    labels : cluster assignments

    Returns
    -------
    Dict with silhouette, calinski_harabasz, davies_bouldin scores.
    """
    n_clusters = len(set(labels) - {-1})
    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "n_clusters": n_clusters,
        }

    # Filter out noise points (label == -1) for metrics
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(set(labels_clean)) < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            "n_clusters": n_clusters,
        }

    return {
        "silhouette": silhouette_score(X_clean, labels_clean),
        "calinski_harabasz": calinski_harabasz_score(X_clean, labels_clean),
        "davies_bouldin": davies_bouldin_score(X_clean, labels_clean),
        "n_clusters": n_clusters,
    }


def silhouette_by_k(X: np.ndarray, k_range: range, method: str = "kmeans") -> pd.DataFrame:
    """
    Compute silhouette scores across a range of k values.

    Returns DataFrame with columns: k, silhouette, inertia (for kmeans).
    """
    from sklearn.cluster import KMeans

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else np.nan
        results.append({
            "k": k,
            "silhouette": sil,
            "inertia": km.inertia_,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------

def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression/forecasting metrics.

    Returns dict with RMSE, MAE, MAPE, R².
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]

    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE — exclude zeros in denominator
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero])
    else:
        mape = np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def build_comparison_table(
    results: dict[str, dict],
) -> pd.DataFrame:
    """
    Build a comparison table from a dict of {model_name: metrics_dict}.

    Returns a DataFrame with models as rows and metrics as columns.
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    return df.round(4)
