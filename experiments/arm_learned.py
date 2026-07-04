"""Learned-representation arms on the cached PTB-XL SR-vs-AF cohort.

Adds, on top of raw_flatten / rhythm_morph:
  - deep_ae   : the properly-specified 1-D conv autoencoder (deep_repr.py).
                Tests the literature claim that RECONSTRUCTION objectives preserve
                energy, not class structure.
  - <foundation>: a FROZEN public ECG foundation-model embedding (added separately).

Runs the same chance-corrected clustering sweep and APPENDS rows to
results/results_learned.csv so they slot into the main benchmark table.
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd

from core import config, cluster_eval as ce
from core.deep_repr import deep_ae_embedding

COHORT = config.PTBXL_DIR / "_cohort_0_lr.pkl"


def load_cohort():
    recs = pickle.load(open(COHORT, "rb"))
    y = np.array([r.label for r in recs])
    groups = np.array([r.patient_id for r in recs])
    return recs, y, groups


def sweep_representation(name, X, y, groups, K):
    rows = []
    sc = ce.supervised_ceiling(ce.make_scaler().fit_transform(X), y, groups)
    print(f"  [{name}] supervised ceiling acc={sc['sup_acc_mean']:.3f} f1={sc['sup_f1_macro_mean']:.3f} "
          f"(dim={X.shape[1]})")
    orig = config.SCALER
    for scaler in config.SCALERS_SWEEP:
        config.SCALER = scaler
        Xs = ce.make_scaler().fit_transform(X)
        for clu in config.CLUSTERERS:
            for seed in config.SEEDS:
                try:
                    yp = ce.fit_clusterer(clu, Xs, K, seed)
                    m = ce.evaluate(Xs, y, yp)
                except Exception as e:
                    print(f"    !! {name}/{scaler}/{clu}/s{seed}: {e}"); continue
                rows.append({"representation": name, "scaler": scaler,
                             "clusterer": clu, "seed": seed, **m})
                if clu in ("agglomerative", "hdbscan"): break
        config.SCALER = orig
    df = pd.DataFrame(rows)
    pw = df[df.scaler == "power"]
    print(f"  [{name}] ARI by clusterer (power): " +
          "  ".join(f"{c}={pw[pw.clusterer==c].ARI.mean():+.3f}"
                    for c in config.CLUSTERERS))
    return df, sc


def main():
    recs, y, groups = load_cohort()
    K = len(np.unique(y))
    print(f"cohort {len(recs)} | classes {np.bincount(y).tolist()} | K={K}")

    print("computing deep_ae embedding (conv autoencoder) ...")
    Xae = deep_ae_embedding(recs, latent=32, epochs=50, seed=0)
    df, sc = sweep_representation("deep_ae", Xae, y, groups, K)

    out = config.RESULTS_DIR / "results_learned.csv"
    df.to_csv(out, index=False)
    pd.DataFrame([{"metric": "supervised_acc[deep_ae]", "value": sc["sup_acc_mean"]},
                  {"metric": "supervised_f1[deep_ae]", "value": sc["sup_f1_macro_mean"]}]
                 ).to_csv(config.RESULTS_DIR / "baselines_learned.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
