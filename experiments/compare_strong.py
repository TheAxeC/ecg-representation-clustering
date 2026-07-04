"""Strong-foundation head-to-head on the FULL cached SR-vs-AF cohort.

Compares chance-corrected SR-vs-AF clustering of three representations on the SAME 2542-record
cohort: rhythm_morph (16 hand-crafted), hubert_ecg (768 SSL), ecgfounder (1024, 10M-ECG CNN).
rhythm_morph and HuBERT come from the cached feature/embedding matrices; ECGFounder is embedded
from the cached 100 Hz signals upsampled to its native 500 Hz (bulk native-500 streaming was
rate-limited; native500_check.py shows upsampling is near-identical). ECGFounder is the fair
adversarial test (it clusters WELL diagnostically per arXiv 2601.21830).

Produces results_compare_strong.csv (the Table 1 foundation rows: HuBERT 0.195, ECGFounder 0.885).
"""
from __future__ import annotations
import pickle
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd

from core import config, cluster_eval as ce
from core.data import Recording  # noqa: F401  (import registers the legacy-pickle module alias)
from experiments.arm_learned import sweep_representation
from experiments.arm_ecgfounder import load_ecgfounder, preprocess_upsampled, embed, EMB_CACHE


def main():
    recs = pickle.load(open(config.PTBXL_DIR / "_cohort_0_lr.pkl", "rb"))
    y = np.array([r.label for r in recs]); groups = np.array([r.patient_id for r in recs])
    K = len(np.unique(y))
    print(f"cohort {len(recs)} | classes {np.bincount(y).tolist()} | majority {ce.majority_purity(y):.3f}\n")

    feats = np.load(config.PTBXL_DIR / "_feats_2542_lr.npz")
    arms = {"rhythm_morph": feats["rhythm"],
            "hubert_ecg": np.load(config.PTBXL_DIR / "_hubert_emb.npy")}

    # ECGFounder: the cached upsampled-100 -> 500 Hz embedding (built by arm_ecgfounder if absent).
    if EMB_CACHE.exists():
        print(f"[cache] {EMB_CACHE.name}")
        arms["ecgfounder"] = np.load(EMB_CACHE)
    else:
        model = load_ecgfounder()
        print("embedding full cohort (upsampled 100 -> 500 Hz) through frozen ECGFounder ...")
        Xe = embed(model, preprocess_upsampled(recs)); np.save(EMB_CACHE, Xe)
        arms["ecgfounder"] = Xe

    summary, rows_all = {}, []
    for name, X in arms.items():
        print(f"\n--- {name} (dim {X.shape[1]}) ---")
        df, sc = sweep_representation(name, X, y, groups, K)
        rows_all.append(df)
        pw = df[df.scaler == "power"]
        summary[name] = {"sup_acc": sc["sup_acc_mean"],
                         **{c: pw[pw.clusterer == c].ARI.mean() for c in config.CLUSTERERS}}

    pd.concat(rows_all).to_csv(config.RESULTS_DIR / "results_compare_strong.csv", index=False)
    print("\n================ HEAD-TO-HEAD (power scaling) ================")
    tab = pd.DataFrame(summary).T[["sup_acc"] + list(config.CLUSTERERS)]
    print(tab.round(3).to_string())
    print("\nbest-clusterer ARI:  " +
          "  ".join(f"{k}={max(v[c] for c in config.CLUSTERERS):+.3f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
