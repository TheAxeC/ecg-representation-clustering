"""Consolidate result CSVs -> publication figures (code/figures/*.pdf) + printed tables.

The figures are written to code/figures/ (regenerable, deposit-excluded); copy the four PDFs into
the manuscript's figures dir before rebuilding the PDF (the paper includes figures/figN_*.pdf).

Figures (colorblind-safe, vector PDF, Times-compatible):
  fig1_ari_heatmap     : PTB-XL ARI, representation x clusterer (power scaling)
  fig2_silhouette_trap : ARI vs silhouette, sized by largest-cluster fraction
  fig3_sep_vs_clust    : supervised ceiling vs best clustering ARI (separability!=clusterability)
  fig4_federated       : centralized / federated-IID / federated-nonIID per representation
Also prints Table 1 (main benchmark) and the MIT-BIH / federated numbers for the manuscript.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"   # code/results
FIG = Path(__file__).resolve().parents[1] / "figures"   # code/figures (copy the 4 PDFs into the manuscript figures dir before rebuilding)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}

REPS = ["raw_flatten", "deep_ae", "rhythm_morph", "hubert_ecg", "ecgfounder"]
REP_LABEL = {"raw_flatten": "raw-flatten", "deep_ae": "conv-AE",
             "rhythm_morph": "hand-crafted", "hubert_ecg": "HuBERT-ECG",
             "ecgfounder": "ECGFounder"}
CLU = ["kmeans", "agglomerative", "spectral", "gmm", "hdbscan"]
CEILING = {"raw_flatten": 0.519, "deep_ae": 0.678, "rhythm_morph": 0.942,
           "hubert_ecg": 0.899, "ecgfounder": 0.983}


def load_ptbxl_power():
    """One long df of PTB-XL rows at scaler=power, all 5 reps."""
    raw = pd.read_csv(RES / "results_raw.csv")
    learned = pd.read_csv(RES / "results_learned.csv")
    strong = pd.read_csv(RES / "results_compare_strong.csv")
    df = pd.concat([raw[raw.representation == "raw_flatten"],
                    learned[learned.representation == "deep_ae"],
                    strong[strong.representation.isin(["rhythm_morph", "hubert_ecg", "ecgfounder"])]],
                   ignore_index=True)
    return df[df.scaler == "power"]


def fig1_heatmap(df):
    M = np.full((len(REPS), len(CLU)), np.nan)
    for i, r in enumerate(REPS):
        for j, c in enumerate(CLU):
            v = df[(df.representation == r) & (df.clusterer == c)].ARI
            if len(v): M[i, j] = v.mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLU))); ax.set_xticklabels([c[:5] for c in CLU])
    ax.set_yticks(range(len(REPS))); ax.set_yticklabels([REP_LABEL[r] for r in REPS])
    for i in range(len(REPS)):
        for j in range(len(CLU)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="ARI", fraction=0.046, pad=0.04)
    ax.set_title("PTB-XL SR-vs-AF: clustering ARI")
    fig.savefig(FIG / "fig1_ari_heatmap.pdf"); plt.close(fig)
    return M


def fig2_silhouette_trap(df):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for r in REPS:
        d = df[df.representation == r]
        ax.scatter(d.silhouette, d.ARI, s=20 + 120 * d.largest_cluster_frac,
                   alpha=0.55, label=REP_LABEL[r],
                   color=[CB["red"], CB["orange"], CB["green"], CB["purple"], CB["blue"]][REPS.index(r)])
    ax.set_xlabel("silhouette (internal index)"); ax.set_ylabel("ARI (external, chance-corrected)")
    ax.axhline(0, color=CB["grey"], lw=0.6)
    ax.set_title("The silhouette trap\n(marker size = largest-cluster fraction)")
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.9)
    fig.savefig(FIG / "fig2_silhouette_trap.pdf"); plt.close(fig)


def fig3_sep_vs_clust(df):
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for r in REPS:
        best = df[df.representation == r].groupby("clusterer").ARI.mean().max()
        ax.scatter(CEILING[r], best, s=70,
                   color=[CB["red"], CB["orange"], CB["green"], CB["purple"], CB["blue"]][REPS.index(r)])
        ax.annotate(REP_LABEL[r], (CEILING[r], best), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)
    ax.plot([0.5, 1], [0.5, 1], ls="--", color=CB["grey"], lw=0.7, label="parity")
    ax.set_xlabel("supervised ceiling (patient-grouped acc)")
    ax.set_ylabel("best clustering ARI")
    ax.set_xlim(0.45, 1.02); ax.set_ylim(-0.05, 1.0)
    ax.set_title("Separability ≠ clusterability")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig3_sep_vs_clust.pdf"); plt.close(fig)


def fig4_federated():
    fed = pd.read_csv(RES / "results_federated.csv")
    piv = fed.pivot(index="representation", columns="setting", values="ARI_mean")
    order = ["ecgfounder", "rhythm_morph", "hubert_ecg", "raw_flatten"]
    cols = ["centralized", "federated_iid", "federated_noniid"]
    labels = ["centralized", "federated IID", "federated non-IID"]
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for k, (c, lab) in enumerate(zip(cols, labels)):
        ax.bar(x + (k - 1) * w, [piv.loc[r, c] for r in order], w, label=lab,
               color=[CB["blue"], CB["green"], CB["orange"]][k])
    ax.set_xticks(x); ax.set_xticklabels([REP_LABEL[r] for r in order], rotation=12)
    ax.set_ylabel("ARI"); ax.set_title("Federated robustness (10 clients)")
    ax.legend(fontsize=7); ax.set_ylim(0, 1)
    fig.savefig(FIG / "fig4_federated.pdf"); plt.close(fig)
    return piv


def main():
    df = load_ptbxl_power()
    M = fig1_heatmap(df); fig2_silhouette_trap(df); fig3_sep_vs_clust(df)
    fed = fig4_federated()

    print("\n=== TABLE 1  PTB-XL mean ARI (power) + supervised ceiling ===")
    t1 = pd.DataFrame(M, index=[REP_LABEL[r] for r in REPS], columns=CLU).round(3)
    t1["sup_ceiling"] = [CEILING[r] for r in REPS]
    print(t1.to_string())

    print("\n=== silhouette trap exemplars (power) ===")
    for r in ["raw_flatten", "rhythm_morph"]:
        for c in ["agglomerative", "kmeans"]:
            d = df[(df.representation == r) & (df.clusterer == c)]
            print(f"  {REP_LABEL[r]:11s}/{c:13s} ARI={d.ARI.mean():.3f} "
                  f"sil={d.silhouette.mean():.3f} largest={d.largest_cluster_frac.mean():.2f}")

    print("\n=== scaler sensitivity (rhythm_morph & ecgfounder, mean ARI over clusterers) ===")
    raw = pd.read_csv(RES / "results_raw.csv"); strong = pd.read_csv(RES / "results_compare_strong.csv")
    print("  rhythm_morph:", raw[raw.representation == "rhythm_morph"].groupby("scaler").ARI.mean().round(3).to_dict())
    print("  ecgfounder  :", strong[strong.representation == "ecgfounder"].groupby("scaler").ARI.mean().round(3).to_dict())

    print("\n=== TABLE 2 federated ARI ===")
    print(fed.round(3).to_string())

    print("\n=== MIT-BIH (power, mean ARI) ===")
    mit = pd.read_csv(RES / "results_mitbih.csv")
    mp = mit[mit.scaler == "power"].pivot_table(index="clusterer", columns="representation", values="ARI", aggfunc="mean")
    print(mp.round(3).to_string())
    print(f"\nwrote figures to {FIG}")


if __name__ == "__main__":
    main()
