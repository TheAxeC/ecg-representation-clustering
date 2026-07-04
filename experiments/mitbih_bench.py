"""MIT-BIH generalization arm: representation-determines-clusterability on a
BEAT-MORPHOLOGY task (Normal vs PVC), the dataset the competing ECG-FM clustering
benchmark omits.

Task: N (normal) vs V (premature ventricular contraction) beats, balanced,
record-grouped. PVCs are the canonical morphology+timing abnormality (wide bizarre
QRS, premature with compensatory pause), present across many records (less
patient-confounded than LBBB/RBBB/paced).

Arms:
  raw_beat   : R-aligned 2-lead beat window, flattened (Euclidean).
  beat_morph : hand-crafted RR-context + per-lead QRS morphology (de Chazal-style).
  deep_ae    : conv autoencoder on the beat window (optional learned arm).

The foundation arm (HuBERT-ECG) is intentionally absent: it ingests 12-lead 5 s
strips, not 2-lead beats -> a documented scoping limit (foundation ECG models are
12-lead-strip-locked), reported as such.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from core import config, cluster_eval as ce
from core.data import Recording

WIN = (0.30, 0.25)            # seconds before/after R
PER_RECORD_CAP = 400          # cap beats/record/class to limit patient dominance


def load_beats(target_syms=("N",), v_syms=("V",), cap=PER_RECORD_CAP):
    """Return beats with RR context. label 0 = normal(target), 1 = PVC(V)."""
    import wfdb
    recs = [p.stem for p in sorted(config.MITBIH_DIR.glob("*.hea"))]
    out = []
    for rid in recs:
        rec = wfdb.rdrecord(str(config.MITBIH_DIR / rid))
        ann = wfdb.rdann(str(config.MITBIH_DIR / rid), "atr")
        fs = rec.fs
        sig = rec.p_signal.astype(np.float32)              # (n, 2)
        pre, post = int(WIN[0] * fs), int(WIN[1] * fs)
        samp = np.asarray(ann.sample); syms = np.asarray(ann.symbol)
        nb = {"+", "~", "|", '"', "x", "[", "]", "!", "?"}
        keep = np.array([s not in nb for s in syms])
        bsamp, bsym = samp[keep], syms[keep]               # beats only, in order
        rr = np.diff(bsamp) / fs                           # seconds, len-1
        n_norm = n_v = 0
        for i in range(1, len(bsamp) - 1):
            sym = bsym[i]
            if sym in target_syms: lab, cls = 0, "N"
            elif sym in v_syms:    lab, cls = 1, "V"
            else: continue
            if cls == "N" and n_norm >= cap: continue
            if cls == "V" and n_v >= cap: continue
            a, b = bsamp[i] - pre, bsamp[i] + post
            if a < 0 or b > len(sig): continue
            beat = sig[a:b]                                 # (pre+post, 2)
            rr_prev, rr_next = rr[i - 1], rr[i]
            lo = max(0, i - 6); rr_local = rr[lo:i].mean() if i > lo else rr_prev
            out.append((beat, lab, rid, rr_prev, rr_next, rr_local))
            if cls == "N": n_norm += 1
            else: n_v += 1
    return out


def _beat_morph_row(beat, rr_prev, rr_next, rr_local, template):
    f = []
    f += [rr_prev, rr_next, rr_prev / (rr_next + 1e-6),
          rr_prev / (rr_local + 1e-6), rr_local]           # RR-context (timing)
    for j in range(beat.shape[1]):                         # per lead morphology
        x = beat[:, j]; r = x[len(x) // 2 - 5:len(x) // 2 + 5]
        r_amp = x[np.argmax(np.abs(x))]
        energy = float(np.sum(x ** 2))
        above = np.abs(x) > 0.5 * np.abs(r_amp + 1e-6)
        qrs_w = float(above.sum())                         # samples above 50% R
        f += [r_amp, energy, qrs_w, float(x.max()), float(x.min()),
              float(skew(x)), float(kurtosis(x)),
              float(np.corrcoef(x, template[:, j])[0, 1])] # vs global template
    return f


def features(beats, kind):
    arr = np.stack([b[0] for b in beats])                  # (N, T, 2)
    if kind == "raw_beat":
        L = min(b[0].shape[0] for b in beats)
        return np.stack([b[0][:L].reshape(-1) for b in beats]).astype(np.float32)
    if kind == "beat_morph":
        template = np.median(arr, axis=0)                  # (T,2) global median beat
        rows = [_beat_morph_row(b[0], b[3], b[4], b[5], template) for b in beats]
        X = np.asarray(rows, dtype=np.float32)
        cm = np.nanmedian(X, axis=0); idx = np.where(np.isnan(X)); X[idx] = np.take(cm, idx[1])
        return X
    if kind == "deep_ae":
        from core.deep_repr import deep_ae_embedding
        L = min(b[0].shape[0] for b in beats)
        rl = [Recording(signal=b[0][:L], fs=360, label=b[1], label_name="", patient_id=b[2], record_id="") for b in beats]
        return deep_ae_embedding(rl, latent=32, epochs=40, seed=0)
    raise ValueError(kind)


def _ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


N_SUB_PER_CLASS = 1500   # balanced subsample so spectral clustering stays tractable


def subsample(beats):
    rng = np.random.default_rng(config.RANDOM_STATE)
    lab = np.array([b[1] for b in beats])
    keep = []
    for c in (0, 1):
        idx = np.where(lab == c)[0]
        keep += list(rng.permutation(idx)[:N_SUB_PER_CLASS])
    return [beats[i] for i in sorted(keep)]


def main():
    beats = subsample(load_beats())
    y = np.array([b[1] for b in beats]); groups = np.array([b[2] for b in beats])
    K = len(np.unique(y))
    print(f"MIT-BIH N-vs-V: {len(beats)} beats | classes(N,V)={np.bincount(y).tolist()} | "
          f"records={len(np.unique(groups))} | majority={ce.majority_purity(y):.3f}")

    rows, base = [], [{"metric": "majority_purity", "value": ce.majority_purity(y)}]
    for rep in ["raw_beat", "beat_morph", "deep_ae"]:
        X = features(beats, rep)
        sc = ce.supervised_ceiling(ce.make_scaler().fit_transform(X), y, groups)
        base += [{"metric": f"supervised_acc[{rep}]", "value": sc["sup_acc_mean"]}]
        print(f"[{rep}] dim={X.shape[1]} supervised acc={sc['sup_acc_mean']:.3f}")
        orig = config.SCALER
        for scaler in config.SCALERS_SWEEP:
            config.SCALER = scaler
            Xs = ce.make_scaler().fit_transform(X)
            for clu in config.CLUSTERERS:
                for seed in config.SEEDS:
                    try:
                        yp = ce.fit_clusterer(clu, Xs, K, seed); m = ce.evaluate(Xs, y, yp)
                    except Exception as e:
                        print(f"  !! {rep}/{scaler}/{clu}/s{seed}: {e}"); continue
                    rows.append({"representation": rep, "scaler": scaler, "clusterer": clu, "seed": seed, **m})
                    if clu in ("agglomerative", "hdbscan"): break
            config.SCALER = orig
        pw = pd.DataFrame(rows); pw = pw[(pw.representation == rep) & (pw.scaler == "power")]
        print(f"   ARI(power): " + "  ".join(f"{c}={pw[pw.clusterer==c].ARI.mean():+.3f}" for c in config.CLUSTERERS))

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "results_mitbih.csv", index=False)
    pd.DataFrame(base).to_csv(config.RESULTS_DIR / "baselines_mitbih.csv", index=False)
    print("\n=== MIT-BIH mean ARI: representation x clusterer (power) ===")
    print(df[df.scaler == "power"].pivot_table(index="clusterer", columns="representation", values="ARI", aggfunc="mean").round(3))
    print(f"wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
