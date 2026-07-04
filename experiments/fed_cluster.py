"""Federated-clustering robustness of ECG representations (modest 2nd contribution).

We do NOT invent a federated method. We take STANDARD federated k-means
(FedAvg-style centroid aggregation) and ask a benchmark question: when the 2542
PTB-XL records are siloed across simulated hospitals (so raw data cannot be
pooled), which representation still recovers the SR-vs-AF structure, and how much
does it degrade vs centralized clustering, under IID vs non-IID client splits?

Representations (cached, no recompute): raw_flatten, rhythm_morph, hubert_ecg,
ecgfounder. Conditioning: global power transform (representation-level), consistent
with the main benchmark; the federation concerns the CLUSTERING step.
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from core import config

N_CLIENTS = 10
N_ROUNDS = 15
LOCAL_ITERS = 3
SEEDS = list(range(10))
K = 2


def load_reps():
    f = np.load(config.PTBXL_DIR / "_feats_2542_lr.npz")
    recs = pickle.load(open(config.PTBXL_DIR / "_cohort_0_lr.pkl", "rb"))
    y = np.array([r.label for r in recs])
    groups = np.array([r.patient_id for r in recs])
    reps = {
        "raw_flatten": f["raw"],
        "rhythm_morph": f["rhythm"],
        "hubert_ecg": np.load(config.PTBXL_DIR / "_hubert_emb.npy"),
        "ecgfounder": np.load(config.PTBXL_DIR / "_ecgfounder_emb_up100.npy"),
    }
    return reps, y, groups


def partition(y, groups, mode, seed):
    """Patient-grouped partition into N_CLIENTS. IID = random; non-IID = class-skewed
    via Dirichlet(0.3) so each client sees very different SR/AF proportions."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # one dominant label per patient (records are single-class here anyway)
    plabel = {g: int(round(y[groups == g].mean())) for g in uniq}
    clients = [[] for _ in range(N_CLIENTS)]
    if mode == "iid":
        for g in rng.permutation(uniq):
            clients[rng.integers(N_CLIENTS)].append(g)
    else:  # non-IID: per-class Dirichlet allocation of patients to clients
        for c in (0, 1):
            pg = rng.permutation([g for g in uniq if plabel[g] == c])
            props = rng.dirichlet(np.full(N_CLIENTS, 0.3))
            cuts = (np.cumsum(props) * len(pg)).astype(int)[:-1]
            for ci, chunk in enumerate(np.split(pg, cuts)):
                clients[ci].extend(chunk.tolist())
    idx = []
    for cl in clients:
        members = np.isin(groups, cl)
        idx.append(np.where(members)[0])
    return [i for i in idx if len(i) >= K]   # drop empty/too-small clients


def federated_kmeans(X, client_idx, seed):
    """FedAvg-style k-means: clients run local Lloyd iters from shared centroids,
    server count-weighted-averages centroids each round."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()      # global init
    for _ in range(N_ROUNDS):
        agg = np.zeros_like(C); wsum = np.zeros(K)
        for idx in client_idx:
            Xi = X[idx]; Cl = C.copy()
            for _ in range(LOCAL_ITERS):
                d = ((Xi[:, None, :] - Cl[None]) ** 2).sum(-1)
                a = d.argmin(1)
                for k in range(K):
                    if (a == k).any(): Cl[k] = Xi[a == k].mean(0)
            for k in range(K):
                nk = (a == k).sum()
                if nk: agg[k] += nk * Cl[k]; wsum[k] += nk
        for k in range(K):
            if wsum[k]: C[k] = agg[k] / wsum[k]
    d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
    return d.argmin(1)


def central_kmeans(X, seed):
    from sklearn.cluster import KMeans
    return KMeans(K, n_init=10, random_state=seed).fit_predict(X)


def main():
    reps, y, groups = load_reps()
    rows = []
    for name, X in reps.items():
        Xs = PowerTransformer().fit_transform(X)
        cen = [adjusted_rand_score(y, central_kmeans(Xs, s)) for s in SEEDS]
        for mode in ("iid", "noniid"):
            aris = []
            for s in SEEDS:
                parts = partition(y, groups, mode, s)
                aris.append(adjusted_rand_score(y, federated_kmeans(Xs, parts, s)))
            rows.append({"representation": name, "setting": f"federated_{mode}",
                         "ARI_mean": np.mean(aris), "ARI_std": np.std(aris)})
        rows.append({"representation": name, "setting": "centralized",
                     "ARI_mean": np.mean(cen), "ARI_std": np.std(cen)})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "results_federated.csv", index=False)
    piv = df.pivot(index="representation", columns="setting", values="ARI_mean")[
        ["centralized", "federated_iid", "federated_noniid"]]
    print(f"Federated k-means, {N_CLIENTS} clients, {len(SEEDS)} seeds, mean ARI:\n")
    print(piv.round(3).to_string())
    print("\nretention (federated_noniid / centralized):")
    print((piv["federated_noniid"] / piv["centralized"].replace(0, np.nan)).round(2).to_string())


if __name__ == "__main__":
    main()
