# Which representation clusters? Benchmark code

Reproducible analysis pipeline for the paper *"Which representation clusters? A chance-corrected benchmark of ECG representations for unsupervised arrhythmia structure"* (Dongho Chun, Axel Faes).

The study benchmarks how the **representation** of an ECG governs whether unsupervised clustering recovers cardiac structure, using chance-corrected external metrics (ARI/AMI), patient-grouped folds, ten seeds, and a supervised ceiling. It began as a re-examination of an MSc project that clustered whole 12-lead recordings and reported a misleading silhouette near 0.89 on a degenerate (one-cluster) solution.

## Key results (reproduced by this code)
| Representation | What it carries | PTB-XL SR-vs-AF, best-clusterer ARI |
|---|---|---|
| `raw_flatten` (~2000-d, the original pipeline) | amplitude/energy, no temporal alignment | 0.00, indistinguishable from majority vote |
| `deep_ae` (32-d conv autoencoder) | learned reconstruction code | 0.01, reconstruction is not clusterability |
| `rhythm_morph` (16 hand-crafted) | RR irregularity, P-wave, QRS morphology | **0.711** |
| `hubert_ecg` (768-d, frozen) | self-supervised foundation embedding | 0.195, separable but not clusterable |
| `ecgfounder` (1024-d, frozen) | foundation embedding (10M-ECG CNN) | **0.885** |

Two methodological cautions: internal indices (silhouette) reward the degenerate collapse (silhouette 0.81 at ARI 0.00), and the feature scaler changes the outcome by a wide margin. Two deployment stressors (9:1 class imbalance, non-IID federation across ten sites) favour the cheap 16-d hand-crafted features. The finding replicates on a MIT-BIH beat-morphology task (hand-crafted ARI 0.783).

## Data
- **PTB-XL 1.0.3** (PhysioNet, DOI 10.13026/x4td-x982), used at 100 Hz: the SR-vs-AF rhythm task.
- **MIT-BIH Arrhythmia Database** (PhysioNet, DOI 10.13026/C2F305): the N-vs-V beat-morphology task.
- Frozen foundation weights (public): HuBERT-ECG (DOI 10.1101/2024.11.14.24317328) and ECGFounder (PKUDigitalHealth). Both public; no proprietary or patient-identifiable data.

## Layout (this repository is a Python package; run drivers from this directory)
Paths resolve from `core/config.py`: inputs from `PTBXL_DIR` / `MITBIH_DIR`, result CSVs to `results/`, figures to `figures/` (copy the four PDFs into the manuscript figures dir before rebuilding the PDF). **Edit `PTBXL_DIR` / `MITBIH_DIR` in `core/config.py` first.**

`core/` (shared library):
- `config.py`: paths, label map, seeds, the experiment matrix.
- `data.py`: PTB-XL label reconstruction from `scp_codes`; MIT-BIH beat segmentation.
- `features.py`: `raw_flatten` baseline + the 16-d `rhythm_morph` features (NeuroKit2).
- `cluster_eval.py`: clusterers (k-means, Ward, spectral, GMM, HDBSCAN), the metric suite (ARI/AMI/NMI/V/FMI/purity/silhouette/largest-cluster fraction), majority and random baselines, and the supervised ceiling.
- `deep_repr.py`: the `deep_ae` 1-D conv autoencoder.
- `ecgfounder_net1d.py`: the reconstructed Net1D backbone used to load ECGFounder weights.

`experiments/` (drivers; each writes to `results/`):
- `benchmark.py`: primary PTB-XL SR-vs-AF sweep (raw + hand-crafted).
- `arm_learned.py`: conv-AE (`deep_ae`) arm; also exports `sweep_representation`, used by the foundation arms.
- `arm_foundation.py`: HuBERT-ECG frozen-embedding arm.
- `compare_strong.py`: full-cohort head-to-head including ECGFounder (upsampled-100 Hz embedding). Produces the Table 1 foundation rows.
- `arm_ecgfounder.py`: the ECGFounder loader (`load_ecgfounder`/`preprocess`/`preprocess_upsampled`/`embed`), imported by `compare_strong` and `native500_check`.
- `fed_cluster.py`: federated k-means (FedAvg centroid aggregation), IID and non-IID, 10 sites.
- `mitbih_bench.py`: MIT-BIH N-vs-V beat-morphology arm.
- `run_analyses.py`: significance (paired Wilcoxon), K-sensitivity AMI (K=2..8), random-forest nonlinear ceiling, 9:1 imbalance. Prints to stdout.
- `native500_check.py`: ECGFounder native-500 Hz vs upsampled-100 Hz consistency check. Prints to stdout.
- `figures.py`: builds Figs 1-4 and prints Tables 1/2/3 from the CSVs.

## Setup and reproduce
```bash
pip install -r requirements.txt    # scikit-learn, scipy, wfdb, neurokit2, torch, transformers, matplotlib
# download PTB-XL 1.0.3 + MIT-BIH (see Data above), then set PTBXL_DIR / MITBIH_DIR in core/config.py
python3 paper.py                                # rebuild every result + figure, in order
# or run a single driver, e.g.:
python -m experiments.compare_strong            # ECGFounder head-to-head (Table 1 foundation rows)
python -m experiments.run_analyses      # significance / K-sweep / RF ceiling / imbalance (stdout)
```
Re-runs are near-instant once the cohort, features, and frozen embeddings are cached under `PTBXL_DIR` (`_cohort_0_lr.pkl`, `_feats_2542_lr.npz`, `_hubert_emb.npy`, `_ecgfounder_emb_up100.npy`). The first run streams the PTB-XL cohort and embeds the foundation models.

## Code -> results -> paper mapping
| Result file | Produced by | Used in |
|---|---|---|
| `results_raw.csv`, `summary.csv` | `benchmark.py` | Table 1 raw + hand-crafted rows; Figs 1-3; silhouette-trap and scaler text |
| `baselines.csv` | `benchmark.py` | majority 0.504; ceilings raw 0.519, hand-crafted 0.942 |
| `results_learned.csv`, `baselines_learned.csv` | `arm_learned.py` | Table 1 conv-AE row (ceiling 0.678) |
| `results_foundation.csv`, `baselines_foundation.csv` | `arm_foundation.py` | HuBERT-ECG standalone (ceiling 0.899) |
| `results_compare_strong.csv` | `compare_strong.py` | Table 1 (HuBERT 0.195, ECGFounder 0.885); Figs 1-3 |
| `results_federated.csv` | `fed_cluster.py` | Table 2; Fig 4 (centralized / IID / non-IID, retention) |
| `results_mitbih.csv`, `baselines_mitbih.csv` | `mitbih_bench.py` | Table 3 (beat-morph 0.783, conv-AE, raw-beat) |

The ECGFounder supervised ceiling (0.983) is taken from `compare_strong` and hard-coded as a constant in `figures.py`. Stdout-only numbers (significance deltas/p-values, K-sensitivity AMI, RF ceilings, imbalance, native-500 consistency) come from `run_analyses.py` and `native500_check.py`.

## Notes
- Cohort: 2542 records (1282 SR / 1260 AF), patient-disjoint, majority baseline 0.504.
- PTB-XL encodes rhythm statements at likelihood 0.0 meaning "present", so the label rule is presence-based.
- The default scaler is Yeo-Johnson power; RobustScaler collapses clustering and is reported as a caution.
- ECGFounder expects 500 Hz; the benchmark embeds the upsampled 100 Hz cohort, validated by `native500_check.py` (ARI 0.896 vs 0.909).
- The 12-lead foundation models are intentionally absent from the 2-lead MIT-BIH arm.
- Fixed seeds throughout; MIT license (`LICENSE`).
