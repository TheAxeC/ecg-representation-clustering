"""Central configuration for the corrected ECG-clustering pipeline.

Edit the paths and the label map to match your local PTB-XL / MIT-BIH copies and
the exact SR/AF/VA subset used in the thesis. Everything else (feature sets,
clustering algorithms, metrics, seeds) is driven from here so experiments are
reproducible from a single file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths: CHANGE THESE                                                          #
# --------------------------------------------------------------------------- #
PTBXL_DIR = Path("~/data/ptb-xl").expanduser()          # contains ptbxl_database.csv, scp_statements.csv, records100/ (100 Hz, used by the benchmark)
MITBIH_DIR = Path("~/data/mit-bih").expanduser()        # contains 100.dat/.hea/.atr, ... (full DB, not one record)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"   # code/results (regenerable; deposit-excluded)

# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #
SEEDS = list(range(10))          # >= 10 seeds; report mean +/- CI over these
RANDOM_STATE = 0                 # for one-off deterministic steps

# --------------------------------------------------------------------------- #
# PTB-XL signal / label settings                                              #
# --------------------------------------------------------------------------- #
PTBXL_FS = 500                   # Hz native; the benchmark uses the 100 Hz lr records
PTBXL_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"]
RHYTHM_LEAD = "II"               # lead used for R-peak / HRV (clinical convention)

# SR / AF / VA derivation from PTB-XL scp_codes (likelihood-weighted statements).
# This reconstructs the 3-class "PTB-XL AFib" task. CONFIRM against the thesis subset.
SR_CODES = {"SR"}                                  # sinus rhythm
AF_CODES = {"AFIB", "AFLT"}                        # atrial fibrillation / flutter
# Everything else that is a *rhythm/arrhythmia* statement -> VA (heterogeneous).
# Recommendation: lead with the well-posed SR-vs-AF task and treat
# VA as background. Set INCLUDE_VA=False to drop it.
INCLUDE_VA = False
# PTB-XL RHYTHM statements (SR/AFIB/AFLT) are encoded at likelihood 0.0 == "present"
# (the 0/15/35/50/80/100 grading applies to *diagnostic/form* statements, not rhythm).
# Threshold must therefore be 0.0 (presence-based) or SR collapses to zero records.
# Verified 2026-06-10: presence-based gives SR=16743, AF=1570 (AFIB 1466 + AFLT 56+48),
# 134 patients shared across classes -> patient-grouped CV is mandatory.
MIN_SCP_LIKELIHOOD = 0.0         # presence-based for rhythm labels (CONFIRMED against PTB-XL 1.0.3)

# --------------------------------------------------------------------------- #
# Experiment matrix                                                            #
# --------------------------------------------------------------------------- #
REPRESENTATIONS = [
    "raw_flatten",   # Route-B baseline: reproduces the thesis (downsample + flatten)
    "rhythm_morph",  # Route-A fix: clinically grounded rhythm + morphology features
    # "deep_ae",     # optional: 1-D conv autoencoder embedding (see deep_repr.py)
]

CLUSTERERS = ["kmeans", "agglomerative", "spectral", "gmm", "hdbscan"]

# raw_flatten baseline knobs (mirror the thesis: 60000 -> 10000 via moving-average downsample)
RAW_TARGET_LEN_PER_LEAD = 833    # ~ 5000 / 6, so 12 * 833 ~= 10000 features
RAW_MA_WINDOW = 6

SCALER = "power"                 # 'power' (Yeo-Johnson; default, normalises heavy HRV tails)
SCALERS_SWEEP = ["power", "standard", "quantile", "robust"]  # conditioning-sensitivity axis


@dataclass
class ClusteringResult:
    representation: str
    clusterer: str
    n_clusters: int
    seed: int
    metrics: dict = field(default_factory=dict)
    cluster_sizes: list = field(default_factory=list)


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
