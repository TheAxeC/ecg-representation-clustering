"""Native-500 Hz consistency check for ECGFounder (peer-review M4).

Streams a small balanced subset at NATIVE 500 Hz and embeds it through ECGFounder,
then embeds the SAME records upsampled from the cached 100 Hz, and compares clustering
ARI. Tests whether the upsampled-100 Hz input materially changed the ECGFounder result.
Small N so streaming stays tractable despite PhysioNet rate-limiting.
"""
from __future__ import annotations
import pickle, posixpath
from concurrent.futures import ThreadPoolExecutor
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from core import config
from experiments.arm_ecgfounder import load_ecgfounder, preprocess, PN_DIR

N_PER_CLASS = 150
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def pick_and_stream():
    base = pickle.load(open(config.PTBXL_DIR / "_cohort_0_lr.pkl", "rb"))
    y = np.array([r.label for r in base])
    idx = list(np.where(y == 0)[0][:N_PER_CLASS]) + list(np.where(y == 1)[0][:N_PER_CLASS])
    db = pd.read_csv(config.PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    sub100 = [base[i] for i in idx]
    cache = config.PTBXL_DIR / f"_native500_{len(idx)}.pkl"
    if cache.exists():
        return sub100, pickle.load(open(cache, "rb"))

    def _one(rec):
        import wfdb
        fn = db.loc[int(rec.record_id), "filename_hr"]; sub, b = posixpath.split(fn)
        sig, meta = wfdb.rdsamp(b, pn_dir=f"{PN_DIR}/{sub}")
        return sig.astype(np.float32)
    print(f"streaming {len(sub100)} records at native 500 Hz ...")
    with ThreadPoolExecutor(max_workers=12) as ex:
        sigs500 = list(ex.map(_one, sub100))
    pickle.dump(sigs500, open(cache, "wb"))
    return sub100, sigs500


@torch.no_grad()
def embed_native(model, sigs500):
    X = np.zeros((len(sigs500), 12, 5000), dtype=np.float32)
    for i, s in enumerate(sigs500):
        s = s[:5000, :12].T
        if s.shape[1] < 5000: s = np.pad(s, ((0, 0), (0, 5000 - s.shape[1])))
        X[i] = (s - s.mean()) / (s.std() + 1e-8)
    out = []
    for i in range(0, len(X), 64):
        _, f = model(torch.tensor(X[i:i+64]).to(DEV)); out.append(f.cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def embed_upsampled(model, sub100):
    X = np.zeros((len(sub100), 12, 5000), dtype=np.float32)
    for i, r in enumerate(sub100):
        s = resample(r.signal[:, :12].T, 5000, axis=1)
        X[i] = (s - s.mean()) / (s.std() + 1e-8)
    out = []
    for i in range(0, len(X), 64):
        _, f = model(torch.tensor(X[i:i+64]).to(DEV)); out.append(f.cpu().numpy())
    return np.vstack(out)


def ari(X, y):
    Xs = PowerTransformer().fit_transform(X)
    yp = KMeans(2, n_init=10, random_state=0).fit_predict(Xs)
    return adjusted_rand_score(y, yp), adjusted_mutual_info_score(y, yp)


def main():
    sub100, sigs500 = pick_and_stream()
    y = np.array([r.label for r in sub100])
    model = load_ecgfounder().to(DEV)
    Xn = embed_native(model, sigs500)
    Xu = embed_upsampled(model, sub100)
    an = ari(Xn, y); au = ari(Xu, y)
    print(f"\nECGFounder on n={len(y)} (k-means, power):")
    print(f"  native 500 Hz   : ARI={an[0]:+.3f}  AMI={an[1]:+.3f}")
    print(f"  upsampled 100 Hz: ARI={au[0]:+.3f}  AMI={au[1]:+.3f}")
    print(f"  difference (native - upsampled): ARI {an[0]-au[0]:+.3f}")


if __name__ == "__main__":
    main()
