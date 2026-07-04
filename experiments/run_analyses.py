"""Supporting analyses for the clustering benchmark.

Uses cached representations only (no streaming):
  significance : paired Wilcoxon across seeds for the headline contrasts (k-means).
  K-sensitivity: AMI for K in 2..8 (AMI is comparable across K).
  nonlinear    : random-forest supervised ceiling (vs the linear LR ceiling).
  imbalance    : natural-imbalance clustering (subsample AF within the cohort).
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from core import config

RES = config.RESULTS_DIR


def load_reps():
    f = np.load(config.PTBXL_DIR / "_feats_2542_lr.npz")
    recs = pickle.load(open(config.PTBXL_DIR / "_cohort_0_lr.pkl", "rb"))
    y = np.array([r.label for r in recs]); g = np.array([r.patient_id for r in recs])
    reps = {"raw_flatten": f["raw"], "rhythm_morph": f["rhythm"],
            "hubert_ecg": np.load(config.PTBXL_DIR / "_hubert_emb.npy"),
            "ecgfounder": np.load(config.PTBXL_DIR / "_ecgfounder_emb_up100.npy")}
    return reps, y, g


def m1_significance():
    """Per-seed k-means ARI from the result CSVs -> paired Wilcoxon."""
    raw = pd.read_csv(RES / "results_raw.csv")
    learned = pd.read_csv(RES / "results_learned.csv")
    strong = pd.read_csv(RES / "results_compare_strong.csv")
    df = pd.concat([raw, learned, strong], ignore_index=True)
    df = df[(df.scaler == "power") & (df.clusterer == "kmeans")]
    seedwise = {r: df[df.representation == r].sort_values("seed").ARI.values
                for r in ["ecgfounder", "rhythm_morph", "hubert_ecg", "raw_flatten"]}
    print("k-means ARI per seed (power):")
    for r, v in seedwise.items():
        print(f"  {r:13s} n={len(v)} mean={v.mean():.3f} sd={v.std(ddof=1):.4f}")
    pairs = [("ecgfounder", "rhythm_morph"), ("rhythm_morph", "hubert_ecg"),
             ("rhythm_morph", "raw_flatten"), ("hubert_ecg", "raw_flatten")]
    print("\npaired Wilcoxon (k-means, 10 seeds):")
    for a, b in pairs:
        va, vb = seedwise[a], seedwise[b]
        n = min(len(va), len(vb))
        if n < 2 or np.allclose(va[:n], vb[:n]):
            print(f"  {a} vs {b}: identical/degenerate (deterministic gap, dmean={va.mean()-vb.mean():+.3f})")
            continue
        try:
            s, p = wilcoxon(va[:n], vb[:n])
            print(f"  {a} vs {b}: dmean={va.mean()-vb.mean():+.3f}  W={s:.1f}  p={p:.4g}")
        except Exception as e:
            print(f"  {a} vs {b}: {e}")


def m2_ksensitivity(reps, y, g):
    print("\nK-sensitivity (k-means, AMI vs 2-class labels, power scaling):")
    hdr = "  K   " + "  ".join(f"{r:12s}" for r in reps)
    print(hdr)
    for K in range(2, 9):
        row = []
        for r, X in reps.items():
            Xs = PowerTransformer().fit_transform(X)
            yp = KMeans(K, n_init=10, random_state=0).fit_predict(Xs)
            row.append(adjusted_mutual_info_score(y, yp))
        print(f"  {K}   " + "  ".join(f"{v:12.3f}" for v in row))


def m3_nonlinear(reps, y, g):
    print("\nNonlinear (random-forest) vs linear ceiling (patient-grouped 5-fold acc):")
    cv = StratifiedGroupKFold(5)
    for r, X in reps.items():
        Xs = PowerTransformer().fit_transform(X)
        rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, class_weight="balanced")
        acc = cross_val_score(rf, Xs, y, groups=g, cv=cv, scoring="accuracy").mean()
        print(f"  {r:13s} RF acc={acc:.3f}")


def d3_imbalance(reps, y, g):
    """Subsample AF within the cohort to ~9:1 SR:AF and recompute ARI."""
    rng = np.random.default_rng(0)
    sr = np.where(y == 0)[0]; af = np.where(y == 1)[0]
    af_keep = rng.permutation(af)[:len(sr) // 9]
    idx = np.sort(np.concatenate([sr, af_keep]))
    yi = y[idx]; maj = np.bincount(yi).max() / len(yi)
    print(f"\nNatural-imbalance ({len(sr)} SR : {len(af_keep)} AF, majority={maj:.3f}) k-means ARI/AMI:")
    for r, X in reps.items():
        Xs = PowerTransformer().fit_transform(X[idx])
        yp = KMeans(2, n_init=10, random_state=0).fit_predict(Xs)
        print(f"  {r:13s} ARI={adjusted_rand_score(yi, yp):+.3f}  AMI={adjusted_mutual_info_score(yi, yp):+.3f}")


def main():
    reps, y, g = load_reps()
    m1_significance()
    m2_ksensitivity(reps, y, g)
    m3_nonlinear(reps, y, g)
    d3_imbalance(reps, y, g)


if __name__ == "__main__":
    main()
