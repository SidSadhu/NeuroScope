"""Clustering algorithms (K-Means, DBSCAN, Hierarchical) for NeuroScope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    algorithm: str
    labels: np.ndarray
    n_clusters: int  # effective number of clusters (excluding noise)
    silhouette: float
    noise_fraction: float  # only meaningful for DBSCAN


# ── Internal helpers ─────────────────────────────────────────────────────────


def _auto_eps(X_scaled: np.ndarray, k: int = 5) -> float:
    """Estimate DBSCAN eps via the 90th-percentile of k-NN distances."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    return float(np.percentile(distances[:, -1], 90))


def _safe_silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score, returning 0.0 on failure."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2:
        return 0.0
    try:
        mask = labels >= 0
        if mask.sum() <= len(unique):
            return 0.0
        return float(silhouette_score(X_scaled[mask], labels[mask]))
    except Exception:
        return 0.0


# ── Public API ───────────────────────────────────────────────────────────────


def run_clustering(
    X: pd.DataFrame,
    n_clusters: int = 3,
) -> dict[str, ClusterResult]:
    """
    Run K-Means, DBSCAN, and Hierarchical Agglomerative clustering.

    Parameters
    ----------
    X : feature matrix (unscaled)
    n_clusters : target k for K-Means and Hierarchical

    Returns
    -------
    dict mapping algorithm name → ClusterResult
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results: dict[str, ClusterResult] = {}

    # ── K-Means ──────────────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    results["K-Means"] = ClusterResult(
        algorithm="K-Means",
        labels=km_labels,
        n_clusters=n_clusters,
        silhouette=_safe_silhouette(X_scaled, km_labels),
        noise_fraction=0.0,
    )

    # ── DBSCAN ───────────────────────────────────────────────────────────
    eps = _auto_eps(X_scaled)
    db_labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)
    n_noise = int((db_labels == -1).sum())
    n_db = int(len(np.unique(db_labels[db_labels >= 0])))
    results["DBSCAN"] = ClusterResult(
        algorithm="DBSCAN",
        labels=db_labels,
        n_clusters=max(n_db, 1),
        silhouette=_safe_silhouette(X_scaled, db_labels),
        noise_fraction=n_noise / len(db_labels),
    )

    # ── Hierarchical (Ward linkage) ───────────────────────────────────────
    hc_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X_scaled)
    results["Hierarchical"] = ClusterResult(
        algorithm="Hierarchical",
        labels=hc_labels,
        n_clusters=n_clusters,
        silhouette=_safe_silhouette(X_scaled, hc_labels),
        noise_fraction=0.0,
    )

    return results


@st.cache_data(show_spinner=False)
def compute_elbow_data(_X: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
    """Compute inertia + silhouette for K=2..max_k (cached)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(_X)

    rows = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": _safe_silhouette(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def reduce_to_3d(X: pd.DataFrame) -> np.ndarray:
    """PCA reduction to 3 components for 3D scatter visualisation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = min(3, X.shape[1])
    return PCA(n_components=n_components, random_state=42).fit_transform(X_scaled)


def get_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Mean feature values per cluster (excludes noise label -1)."""
    tmp = df[feature_cols].copy()
    tmp["_cluster"] = labels
    tmp = tmp[tmp["_cluster"] >= 0]
    return tmp.groupby("_cluster")[feature_cols].mean().round(2)
