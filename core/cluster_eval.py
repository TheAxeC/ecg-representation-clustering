"""Clustering, chance-corrected evaluation, baselines, and a supervised ceiling.

This is the module that fixes the thesis's evaluation. The headline metrics are
EXTERNAL and CHANCE-CORRECTED (ARI, AMI), so a degenerate "one big cluster"
solution scores ~0 instead of the misleadingly high purity / silhouette the
thesis reported. Purity and silhouette are still computed, but always alongside
their baselines and cluster-size sanity, never as standalone evidence.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import (AgglomerativeClustering, KMeans, SpectralClustering)
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             fowlkes_mallows_score, normalized_mutual_info_score,
                             silhouette_score, v_measure_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.linear_model import LogisticRegression

from core import config


# --------------------------------------------------------------------------- #
# Scaling                                                                      #
# --------------------------------------------------------------------------- #
def make_scaler():
    # NOTE: scaler choice is consequential for clustering. HRV/RR-irregularity
    # features are heavy-tailed (AFib outliers); RobustScaler leaves those tails
    # unbounded -> Euclidean distance explodes -> k-means collapses to one cluster
    # (ARI~0) even though the features are linearly separable (supervised acc>0.9).
    # PowerTransformer (Yeo-Johnson) normalises the tails and recovers ARI~0.56-0.60
    # consistently across clusterers. Verified on the PTB-XL SR-vs-AF smoke set.
    return {"power": lambda: PowerTransformer(method="yeo-johnson"),
            "standard": StandardScaler,
            "quantile": lambda: QuantileTransformer(output_distribution="normal"),
            "robust": RobustScaler,
            "minmax": lambda: MinMaxScaler((-1, 1))}[config.SCALER]()


# --------------------------------------------------------------------------- #
# Clustering                                                                   #
# --------------------------------------------------------------------------- #
def fit_clusterer(name: str, X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if name == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
    if name == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X)
    if name == "spectral":
        return SpectralClustering(n_clusters=n_clusters, random_state=seed,
                                  affinity="nearest_neighbors", assign_labels="kmeans").fit_predict(X)
    if name == "gmm":
        # diagonal covariance + reg_covar: full covariance is singular when the
        # representation is high-dim relative to n (e.g. raw_flatten). Diagonal keeps
        # GMM usable across all arms on equal footing.
        return GaussianMixture(n_components=n_clusters, covariance_type="diag",
                               reg_covar=1e-4, random_state=seed).fit_predict(X)
    if name == "hdbscan":
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=max(10, X.shape[0] // 50)).fit_predict(X)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for c in np.unique(y_pred):
        if c == -1:                       # HDBSCAN noise
            continue
        members = y_true[y_pred == c]
        if len(members):
            total += np.bincount(members).max()
    return total / len(y_true)


def evaluate(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = y_pred != -1                   # exclude noise points from internal index
    sil = (silhouette_score(X[mask], y_pred[mask])
           if mask.sum() > len(np.unique(y_pred[mask])) > 1 else np.nan)
    sizes = np.bincount(y_pred[y_pred != -1]).tolist()
    return {
        # chance-corrected (headline)
        "ARI": adjusted_rand_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        # not chance-corrected (context)
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Vmeasure": v_measure_score(y_true, y_pred),
        "FMI": fowlkes_mallows_score(y_true, y_pred),
        "purity": purity(y_true, y_pred),
        "silhouette": sil,
        "largest_cluster_frac": (max(sizes) / len(y_true)) if sizes else np.nan,
        "n_nonempty_clusters": len(sizes),
    }


# --------------------------------------------------------------------------- #
# Baselines and supervised ceiling: the controls the thesis lacked           #
# --------------------------------------------------------------------------- #
def majority_purity(y_true: np.ndarray) -> float:
    """Purity achieved by assigning everything to the majority class.
    The thesis's reported purity equals THIS number in every experiment."""
    return np.bincount(y_true).max() / len(y_true)


def random_baseline(y_true: np.ndarray, n_clusters: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y_rand = rng.integers(0, n_clusters, size=len(y_true))
    return {"ARI": adjusted_rand_score(y_true, y_rand),
            "AMI": adjusted_mutual_info_score(y_true, y_rand),
            "purity": purity(y_true, y_rand)}


def supervised_ceiling(X: np.ndarray, y_true: np.ndarray, groups: np.ndarray) -> dict:
    """Patient-grouped CV accuracy/F1 of a simple supervised classifier on the SAME
    features. Upper bound on what is *learnable* from the representation: if this is
    near chance, the failure is the representation, not the clustering algorithm."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = StratifiedGroupKFold(n_splits=5)
    acc = cross_val_score(clf, X, y_true, groups=groups, cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X, y_true, groups=groups, cv=cv, scoring="f1_macro")
    return {"sup_acc_mean": acc.mean(), "sup_acc_std": acc.std(),
            "sup_f1_macro_mean": f1.mean(), "sup_f1_macro_std": f1.std()}
