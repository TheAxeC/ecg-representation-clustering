"""Chance-corrected representation benchmark on PTB-XL SR-vs-AF.

Streams a balanced, patient-disjoint SR/AF cohort from PhysioNet (threaded I/O,
cached), extracts both representations (rhythm_morph in parallel), then runs the
full sweep: representation x clusterer x seed x scaler, plus a supervised ceiling
and majority/random baselines. Writes publication-ready CSVs to results/.

Usage:
  python benchmark.py                  # full published cohort: all AF + matched SR (~2542 records), the default
  python benchmark.py --per-class 800  # quick 800 SR + 800 AF smoke run
  python benchmark.py --res hr         # 500 Hz (cleaner morphology)
"""
from __future__ import annotations
import argparse, ast, pickle, posixpath
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

from core import config, cluster_eval as ce
from core import features as featmod
from core.data import Recording

PN_DIR = "ptb-xl/1.0.3"


# --------------------------------------------------------------------------- #
# Cohort construction + streaming                                             #
# --------------------------------------------------------------------------- #
def build_picks(per_class: int, res: str):
    db = pd.read_csv(config.PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    def lab(codes):
        c = {k for k, v in codes.items() if v >= config.MIN_SCP_LIKELIHOOD}
        if c & config.AF_CODES: return "AF"
        if (c & config.SR_CODES) and not (c & config.AF_CODES): return "SR"
        return None
    db["lab"] = db.scp_codes.apply(lab)

    af = db[db.lab == "AF"]
    n = af.patient_id.nunique() if per_class in (0, None) else per_class
    picks, used = [], set()
    for name in ("SR", "AF"):
        sub = db[db.lab == name].sample(frac=1.0, random_state=config.RANDOM_STATE)
        k = 0
        for ecg_id, row in sub.iterrows():
            if row.patient_id in used:                    # patient-disjoint across the cohort
                continue
            used.add(row.patient_id)
            picks.append((ecg_id, name, str(row.patient_id), row[f"filename_{res}"]))
            k += 1
            if k >= n: break
    return picks


def _stream_one(args):
    import wfdb
    ecg_id, name, pid, fn = args
    sub, base = posixpath.split(fn)
    sig, meta = wfdb.rdsamp(base, pn_dir=f"{PN_DIR}/{sub}")
    return Recording(signal=sig.astype(np.float32), fs=int(meta["fs"]),
                     label=0 if name == "SR" else 1, label_name=name,
                     patient_id=pid, record_id=str(ecg_id))


def stream_cohort(per_class: int, res: str):
    cache = config.PTBXL_DIR / f"_cohort_{per_class}_{res}.pkl"
    if cache.exists():
        print(f"[cache] {cache.name}")
        return pickle.load(open(cache, "rb"))
    picks = build_picks(per_class, res)
    print(f"streaming {len(picks)} records (threaded)...")
    recs = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, r in enumerate(ex.map(_stream_one, picks)):
            recs.append(r)
            if (i + 1) % 200 == 0: print(f"  {i+1}/{len(picks)}")
    pickle.dump(recs, open(cache, "wb"))
    return recs


# --------------------------------------------------------------------------- #
# Parallel feature extraction (rhythm_morph is the slow part)                  #
# --------------------------------------------------------------------------- #
def _rhythm_one(rec: Recording):
    return featmod.rhythm_morph([rec])[0]


def extract_features(recs, res: str):
    cache = config.PTBXL_DIR / f"_feats_{len(recs)}_{res}.npz"
    if cache.exists():
        print(f"[cache] {cache.name}")
        d = np.load(cache)
        return {"raw_flatten": d["raw"], "rhythm_morph": d["rhythm"]}
    print("extracting raw_flatten ...")
    raw = featmod.raw_flatten(recs)
    print(f"extracting rhythm_morph ({len(recs)} records, parallel) ...")
    with Pool() as pool:
        rows = pool.map(_rhythm_one, recs, chunksize=8)
    rhythm = np.vstack(rows).astype(np.float32)
    col_med = np.nanmedian(rhythm, axis=0)
    idx = np.where(np.isnan(rhythm)); rhythm[idx] = np.take(col_med, idx[1])
    np.savez(cache, raw=raw, rhythm=rhythm)
    return {"raw_flatten": raw, "rhythm_morph": rhythm}


# --------------------------------------------------------------------------- #
# Sweep                                                                        #
# --------------------------------------------------------------------------- #
def _ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def run(per_class, res):
    recs = stream_cohort(per_class, res)
    y = np.array([r.label for r in recs]); groups = np.array([r.patient_id for r in recs])
    K = len(np.unique(y))
    print(f"cohort: {len(recs)} | classes={np.bincount(y).tolist()} | "
          f"patients={len(np.unique(groups))} | majority={ce.majority_purity(y):.3f}")
    feats = extract_features(recs, res)

    rows, base_rows = [], []
    base_rows.append({"metric": "majority_purity", "value": ce.majority_purity(y)})
    base_rows.append({"metric": "random_ARI",
                      "value": float(np.mean([ce.random_baseline(y, K, s)["ARI"] for s in config.SEEDS]))})

    orig_scaler = config.SCALER
    for rep, X in feats.items():
        for scaler in config.SCALERS_SWEEP:
            config.SCALER = scaler
            Xs = ce.make_scaler().fit_transform(X)
            if scaler == config.SCALERS_SWEEP[0]:   # ceiling once per rep (scaler-invariant enough)
                sc = ce.supervised_ceiling(Xs, y, groups)
                base_rows.append({"metric": f"supervised_acc[{rep}]", "value": sc["sup_acc_mean"]})
                base_rows.append({"metric": f"supervised_f1[{rep}]", "value": sc["sup_f1_macro_mean"]})
            for clu in config.CLUSTERERS:
                for seed in config.SEEDS:
                    try:
                        yp = ce.fit_clusterer(clu, Xs, K, seed)
                        m = ce.evaluate(Xs, y, yp)
                    except Exception as e:
                        print(f"  !! {rep}/{scaler}/{clu}/s{seed}: {e}"); continue
                    rows.append({"representation": rep, "scaler": scaler,
                                 "clusterer": clu, "seed": seed, **m})
                    if clu in ("agglomerative", "hdbscan"): break
    config.SCALER = orig_scaler

    raw = pd.DataFrame(rows)
    raw.to_csv(config.RESULTS_DIR / "results_raw.csv", index=False)
    pd.DataFrame(base_rows).to_csv(config.RESULTS_DIR / "baselines.csv", index=False)
    mcols = ["ARI", "AMI", "NMI", "Vmeasure", "FMI", "purity", "silhouette",
             "largest_cluster_frac", "n_nonempty_clusters"]
    summ = raw.groupby(["representation", "scaler", "clusterer"])[mcols].agg(["mean", _ci95])
    summ.to_csv(config.RESULTS_DIR / "summary.csv")

    print("\n=== mean ARI: representation x clusterer (scaler=power) ===")
    piv = (raw[raw.scaler == "power"]
           .pivot_table(index="clusterer", columns="representation", values="ARI", aggfunc="mean"))
    print(piv.round(3))
    print("\n=== scaler sensitivity (rhythm_morph, mean ARI over clusterers) ===")
    print(raw[raw.representation == "rhythm_morph"].groupby("scaler")["ARI"].mean().round(3))
    print(f"\nwrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=0)   # 0 = full published cohort (all AF + matched SR, ~2542)
    ap.add_argument("--res", choices=["lr", "hr"], default="lr")
    a = ap.parse_args()
    run(a.per_class, a.res)
