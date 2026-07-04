"""ECGFounder (PKUDigitalHealth, 10M-ECG CNN foundation model): loader + embedding.

The strong-foundation arm demanded after the lit check: ECGFounder is one of the models
arXiv 2601.21830 reports as clustering WELL by clinical label (unlike HuBERT-ECG). We load it as
a frozen feature extractor (1024-d global-avg-pooled deep_features).

ECGFounder expects 12-lead x 5000 @ 500 Hz with a GLOBAL z-score (its dataset.py). Bulk native
500 Hz streaming from PhysioNet was rate-limited, so the published benchmark embeds the cached
100 Hz cohort UPSAMPLED to 500 Hz; native500_check.py confirms this is near-identical to native
500 Hz (delta ARI 0.013), so there is no handicap.

This module is the ECGFounder loader (load_ecgfounder / preprocess / preprocess_upsampled / embed),
imported by compare_strong.py and native500_check.py. Run it standalone to (re)build the cached
full-cohort embedding `_ecgfounder_emb_up100.npy` plus a single-arm sweep (results_ecgfounder.csv,
which the paper does not use; the published table comes from compare_strong.py).

Provenance: Net1D reconstructed from the official repo net1d.py (inspected: torch/numpy only, no
os/subprocess/eval/network); user-authorized direct load of the public MIT-licensed checkpoint.
Correct load confirmed: 0 missing / 0 unexpected.
"""
from __future__ import annotations
import pickle
import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from core import config, cluster_eval as ce
from core.data import Recording  # noqa: F401  (import registers the legacy-pickle module alias)
from experiments.arm_learned import sweep_representation

PN_DIR = "ptb-xl/1.0.3"
COHORT = config.PTBXL_DIR / "_cohort_0_lr.pkl"               # full 2542-record SR/AF cohort, 100 Hz
EMB_CACHE = config.PTBXL_DIR / "_ecgfounder_emb_up100.npy"  # full cohort, upsampled 100 -> 500 Hz


def load_ecgfounder():
    from huggingface_hub import hf_hub_download
    from core.ecgfounder_net1d import Net1D
    ckpt = hf_hub_download("PKUDigitalHealth/ECGFounder", "12_lead_ECGFounder.pth")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items() if not k.startswith("dense.")}
    model = Net1D(in_channels=12, base_filters=64, ratio=1,
                  filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                  m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2,
                  groups_width=16, n_classes=150, use_bn=False, use_do=False,
                  return_features=True, verbose=False)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"ECGFounder loaded: missing(non-dense)={len([m for m in miss if not m.startswith('dense')])} "
          f"unexpected={len(unexp)}")
    model.eval()
    return model


def preprocess(recs):
    """(12, 5000) from a NATIVE 500 Hz signal, global z-score (matches ECGFounder dataset.py).
    Used by native500_check.py for the native-rate comparison."""
    X = np.zeros((len(recs), 12, 5000), dtype=np.float32)
    for i, r in enumerate(recs):
        s = r.signal[:5000, :12].T                              # (12, 5000)
        if s.shape[1] < 5000:
            s = np.pad(s, ((0, 0), (0, 5000 - s.shape[1])))
        X[i] = (s - s.mean()) / (s.std() + 1e-8)
    return X


def preprocess_upsampled(recs):
    """(12, 5000) from the cached 100 Hz signal upsampled to ECGFounder's 500 Hz, global z-score.
    This is the PUBLISHED input path (bulk native-500 streaming was rate-limited; see native500_check.py)."""
    from scipy.signal import resample
    X = np.zeros((len(recs), 12, 5000), dtype=np.float32)
    for i, r in enumerate(recs):
        s = resample(r.signal[:, :12].T, 5000, axis=1)          # (12, 5000), upsampled 100 -> 500 Hz
        X[i] = (s - s.mean()) / (s.std() + 1e-8)
    return X


@torch.no_grad()
def embed(model, X, batch=32):
    out = []
    for i in range(0, len(X), batch):
        _, feat = model(torch.tensor(X[i:i + batch]))
        out.append(feat.cpu().numpy())
        if (i // batch) % 10 == 0: print(f"  embedded {min(i+batch, len(X))}/{len(X)}")
    return np.vstack(out).astype(np.float32)


def main():
    recs = pickle.load(open(COHORT, "rb"))
    y = np.array([r.label for r in recs]); groups = np.array([r.patient_id for r in recs])
    K = len(np.unique(y))
    print(f"cohort {len(recs)} | classes {np.bincount(y).tolist()}")
    if EMB_CACHE.exists():
        print("[cache] ECGFounder embeddings"); Xe = np.load(EMB_CACHE)
    else:
        model = load_ecgfounder()
        print("embedding full cohort (upsampled 100 -> 500 Hz) through frozen ECGFounder ...")
        Xe = embed(model, preprocess_upsampled(recs)); np.save(EMB_CACHE, Xe)
    print(f"embedding shape {Xe.shape}")

    df, sc = sweep_representation("ecgfounder", Xe, y, groups, K)
    df.to_csv(config.RESULTS_DIR / "results_ecgfounder.csv", index=False)
    pd.DataFrame([{"metric": "supervised_acc[ecgfounder]", "value": sc["sup_acc_mean"]},
                  {"metric": "supervised_f1[ecgfounder]", "value": sc["sup_f1_macro_mean"]}]
                 ).to_csv(config.RESULTS_DIR / "baselines_ecgfounder.csv", index=False)
    print("wrote results/results_ecgfounder.csv")


if __name__ == "__main__":
    main()
