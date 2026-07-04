"""Data loading for PTB-XL and the full MIT-BIH Arrhythmia DB.

Returns, per dataset, a list of `Recording` objects carrying the raw multi-lead
signal, sampling rate, an integer label, a string label name, and a patient id
(for leakage-safe, patient-grouped evaluation).

Design notes
------------
* PTB-XL labels are reconstructed from `ptbxl_database.csv` scp_codes. Confirm the
  mapping in config.py matches the exact subset used in the thesis.
* MIT-BIH is loaded across ALL records and segmented into beats (the thesis used a
  single record, record 100, explicitly not generalisable). Each
  beat becomes a sample; the record id is used as the patient/group id.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import config


@dataclass
class Recording:
    signal: np.ndarray      # shape (n_samples, n_leads)
    fs: int
    label: int
    label_name: str
    patient_id: str
    record_id: str


# --------------------------------------------------------------------------- #
# PTB-XL                                                                       #
# --------------------------------------------------------------------------- #
def _ptbxl_label(scp_codes: dict) -> str | None:
    """Map a PTB-XL scp_codes dict -> {'SR','AF','VA'} (or None to skip)."""
    codes = {k for k, v in scp_codes.items() if v >= config.MIN_SCP_LIKELIHOOD}
    if codes & config.AF_CODES:
        return "AF"
    if codes & config.SR_CODES and not (codes & config.AF_CODES):
        return "SR"
    if config.INCLUDE_VA and codes:
        return "VA"
    return None


def load_ptbxl(limit: int | None = None) -> list[Recording]:
    import wfdb

    db = pd.read_csv(config.PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    label_names = {"SR": 0, "AF": 1, "VA": 2}
    recs: list[Recording] = []
    for ecg_id, row in db.iterrows():
        name = _ptbxl_label(row.scp_codes)
        if name is None:
            continue
        # records500 path is stored in the 'filename_hr' column (high-res, 500 Hz)
        rec_path = config.PTBXL_DIR / row.filename_hr
        sig, meta = wfdb.rdsamp(str(rec_path))
        recs.append(Recording(
            signal=sig.astype(np.float32),         # (5000, 12)
            fs=int(meta["fs"]),
            label=label_names[name],
            label_name=name,
            patient_id=str(row.patient_id),
            record_id=str(ecg_id),
        ))
        if limit and len(recs) >= limit:
            break
    return recs


# --------------------------------------------------------------------------- #
# MIT-BIH (multi-record, beat-segmented)                                       #
# --------------------------------------------------------------------------- #
# AAMI-style grouping kept minimal: N (normal) vs not-N. Extend as needed.
_MITBIH_NORMAL = {"N", "L", "R", "e", "j"}


def load_mitbih_beats(window_ms=(300, 250), records: list[str] | None = None) -> list[Recording]:
    """Segment every annotated beat from every MIT-BIH record into a fixed window.

    window_ms = (before_R, after_R) in milliseconds. Default [-300, +250] ms matches
    the 'full heartbeat' window from Garcia-Isla et al. (2021) used in the thesis.
    """
    import wfdb

    if records is None:
        # standard 48 records
        records = [p.stem for p in sorted(config.MITBIH_DIR.glob("*.hea"))]

    recs: list[Recording] = []
    for rid in records:
        rec = wfdb.rdrecord(str(config.MITBIH_DIR / rid))
        ann = wfdb.rdann(str(config.MITBIH_DIR / rid), "atr")
        fs = rec.fs
        sig = rec.p_signal.astype(np.float32)             # (n, n_leads), MLII usually lead 0
        pre = int(window_ms[0] / 1000 * fs)
        post = int(window_ms[1] / 1000 * fs)
        for samp, sym in zip(ann.sample, ann.symbol):
            if sym in ("+", "~", "|", '"', "x", "[", "]"):   # non-beat annotations
                continue
            a, b = samp - pre, samp + post
            if a < 0 or b > len(sig):
                continue
            label_name = "N" if sym in _MITBIH_NORMAL else "A"
            recs.append(Recording(
                signal=sig[a:b],                              # (pre+post, n_leads)
                fs=fs,
                label=0 if label_name == "N" else 1,
                label_name=label_name,
                patient_id=rid,
                record_id=f"{rid}_{samp}",
            ))
    return recs
