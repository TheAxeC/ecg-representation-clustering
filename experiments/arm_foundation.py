"""Frozen foundation-model arm: HuBERT-ECG (Edoardo-BS/hubert-ecg-base).

This is the novelty-critical arm. We load the PUBLIC, self-supervised HuBERT-ECG
foundation model (pretrained on 9.1M 12-lead ECGs) as a FROZEN feature extractor
and ask: does a 90M-param SSL embedding cluster SR-vs-AF better than 16 hand-crafted
features under chance-corrected metrics?

Safety / provenance:
  HuBERTECG is a thin subclass of transformers.HubertModel adding only pretraining
  codebook heads (final_proj, label_embedding), see the inspected hubert_ecg.py
  (46 lines, no os/subprocess/eval/network). We therefore load the weights into a
  STOCK transformers.HubertModel (no trust_remote_code / no remote code execution),
  dropping the pretraining heads.

Preprocessing (faithful to the official repo, hubert_ecg/dataset.py `encode` path):
  first 5 s, 12 leads flattened lead-major, decimated to 100 Hz -> length-6000 1-D
  input; no explicit z-scoring (conv front-end uses GroupNorm). Our cached cohort is
  already 100 Hz, so we take the first 500 samples/lead x 12 = 6000.
"""
from __future__ import annotations
import json, pickle
import numpy as np
import pandas as pd
import torch

from core import config, cluster_eval as ce
from experiments.arm_learned import sweep_representation

REPO = "Edoardo-BS/hubert-ecg-base"
COHORT = config.PTBXL_DIR / "_cohort_0_lr.pkl"
SAMPLES_PER_LEAD_100HZ = 500          # 5 s at 100 Hz
INPUT_LEN = 12 * SAMPLES_PER_LEAD_100HZ  # 6000


def load_frozen_hubert():
    from huggingface_hub import hf_hub_download
    from transformers import HubertConfig, HubertModel
    from safetensors.torch import load_file

    cfg_path = hf_hub_download(REPO, "config.json")
    w_path = hf_hub_download(REPO, "model.safetensors")
    cfg_d = json.load(open(cfg_path))
    # keep only stock-Hubert fields; drop hubert_ecg extras + custom model_type
    for k in ("ensemble_length", "vocab_sizes", "model_type", "auto_map", "architectures"):
        cfg_d.pop(k, None)
    cfg = HubertConfig(**cfg_d)
    model = HubertModel(cfg)

    sd = load_file(w_path)
    # drop pretraining heads that don't exist on stock HubertModel
    sd = {k: v for k, v in sd.items()
          if not (k.startswith("final_proj") or k.startswith("label_embedding"))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if "masked_spec_embed" not in m]
    print(f"loaded HuBERT-ECG: {len(sd)} tensors | missing(non-mask)={len(real_missing)} "
          f"| unexpected={len(unexpected)}")
    if real_missing[:5]:
        print("  missing sample:", real_missing[:5])
    model.eval()
    return model


def preprocess(recs):
    X = np.zeros((len(recs), INPUT_LEN), dtype=np.float32)
    for i, r in enumerate(recs):
        sig = r.signal[:SAMPLES_PER_LEAD_100HZ, :12]          # (500, 12)
        if sig.shape[0] < SAMPLES_PER_LEAD_100HZ:             # pad short records
            sig = np.pad(sig, ((0, SAMPLES_PER_LEAD_100HZ - sig.shape[0]), (0, 0)))
        X[i] = sig.T.reshape(-1)                               # lead-major flatten -> 6000
    return X


@torch.no_grad()
def embed(model, X, batch=64):
    embs = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch])
        out = model(xb).last_hidden_state                     # (B, T', 768)
        embs.append(out.mean(dim=1).cpu().numpy())            # mean-pool over time
        if (i // batch) % 5 == 0:
            print(f"  embedded {min(i+batch, len(X))}/{len(X)}")
    return np.vstack(embs).astype(np.float32)


def main():
    recs = pickle.load(open(COHORT, "rb"))
    y = np.array([r.label for r in recs]); groups = np.array([r.patient_id for r in recs])
    K = len(np.unique(y))
    print(f"cohort {len(recs)} | classes {np.bincount(y).tolist()}")

    cache = config.PTBXL_DIR / "_hubert_emb.npy"
    if cache.exists():
        print("[cache] hubert embeddings"); Xh = np.load(cache)
    else:
        model = load_frozen_hubert()
        print("embedding cohort through frozen HuBERT-ECG ...")
        Xh = embed(model, preprocess(recs))
        np.save(cache, Xh)
    print(f"embedding shape {Xh.shape}")

    df, sc = sweep_representation("hubert_ecg", Xh, y, groups, K)
    out = config.RESULTS_DIR / "results_foundation.csv"
    df.to_csv(out, index=False)
    pd.DataFrame([{"metric": "supervised_acc[hubert_ecg]", "value": sc["sup_acc_mean"]},
                  {"metric": "supervised_f1[hubert_ecg]", "value": sc["sup_f1_macro_mean"]}]
                 ).to_csv(config.RESULTS_DIR / "baselines_foundation.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
