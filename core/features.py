"""Representations / feature extraction.

Two representations matter for the paper's argument:

  raw_flatten   -- Route-B baseline. Reproduces the thesis: moving-average
                   downsample each lead, flatten all leads to one long vector.
                   Carries amplitude/energy but destroys cross-recording temporal
                   correspondence -> clustering ~ chance (the result to beat).

  rhythm_morph  -- Route-A fix. Per-recording, clinically grounded features that
                   actually encode the arrhythmia: RR-interval statistics and
                   irregularity (the AFib signature), P-wave presence/amplitude,
                   QRS width/amplitude, and median-beat morphology summaries.

Both return a (n_recordings, n_features) float matrix aligned to the input list.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import entropy as shannon_entropy

from core import config

warnings.filterwarnings("ignore")  # neurokit emits many per-record warnings


# --------------------------------------------------------------------------- #
# Baseline representation (the thesis pipeline)                                #
# --------------------------------------------------------------------------- #
def _moving_average_downsample(x: np.ndarray, window: int) -> np.ndarray:
    """Non-overlapping moving average: averages every `window` samples (the thesis
    'downsampling' that doubles as a smoother). x: (n_samples,)."""
    n = (len(x) // window) * window
    return x[:n].reshape(-1, window).mean(axis=1)


def raw_flatten(recordings) -> np.ndarray:
    feats = []
    for r in recordings:
        per_lead = [_moving_average_downsample(r.signal[:, j], config.RAW_MA_WINDOW)
                    for j in range(r.signal.shape[1])]
        L = min(len(p) for p in per_lead)
        feats.append(np.concatenate([p[:L] for p in per_lead]))
    L = min(len(f) for f in feats)
    return np.vstack([f[:L] for f in feats]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Rhythm + morphology representation (the fix)                                 #
# --------------------------------------------------------------------------- #
RHYTHM_MORPH_NAMES = [
    "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_cv", "rr_entropy",
    "hr_mean", "n_beats",
    "pwave_present_frac", "pwave_amp_mean", "pwave_amp_std",
    "qrs_width_mean", "qrs_amp_mean", "qrs_amp_std",
    "beat_corr_mean", "beat_corr_std",   # template-consistency: low in AFib/ectopy
]


def _rr_features(rpeaks: np.ndarray, fs: int) -> dict:
    if rpeaks is None or len(rpeaks) < 3:
        return dict.fromkeys(
            ["rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_cv", "rr_entropy",
             "hr_mean", "n_beats"], np.nan)
    rr = np.diff(rpeaks) / fs * 1000.0          # ms
    drr = np.diff(rr)
    # RR-distribution entropy: irregularity measure that spikes for AFib
    hist, _ = np.histogram(rr, bins=16, range=(200, 2000))
    p = hist / max(hist.sum(), 1)
    return {
        "rr_mean": np.mean(rr),
        "rr_sdnn": np.std(rr),
        "rr_rmssd": np.sqrt(np.mean(drr ** 2)) if len(drr) else np.nan,
        "rr_pnn50": np.mean(np.abs(drr) > 50) if len(drr) else np.nan,
        "rr_cv": np.std(rr) / np.mean(rr),
        "rr_entropy": shannon_entropy(p + 1e-12),
        "hr_mean": 60000.0 / np.mean(rr),
        "n_beats": float(len(rpeaks)),
    }


def _morphology_features(sig_lead: np.ndarray, rpeaks: np.ndarray, fs: int, waves: dict) -> dict:
    out = dict.fromkeys(
        ["pwave_present_frac", "pwave_amp_mean", "pwave_amp_std",
         "qrs_width_mean", "qrs_amp_mean", "qrs_amp_std",
         "beat_corr_mean", "beat_corr_std"], np.nan)
    if rpeaks is None or len(rpeaks) < 3:
        return out

    # P-wave presence/amplitude -> AFib lacks organised P-waves
    p_peaks = np.asarray(waves.get("ECG_P_Peaks", []), dtype=float)
    valid_p = p_peaks[~np.isnan(p_peaks)].astype(int)
    out["pwave_present_frac"] = len(valid_p) / len(rpeaks)
    if len(valid_p):
        amps = sig_lead[np.clip(valid_p, 0, len(sig_lead) - 1)]
        out["pwave_amp_mean"], out["pwave_amp_std"] = np.mean(amps), np.std(amps)

    # QRS width from onset/offset; amplitude at R
    onsets = np.asarray(waves.get("ECG_R_Onsets", []), dtype=float)
    offsets = np.asarray(waves.get("ECG_R_Offsets", []), dtype=float)
    m = ~(np.isnan(onsets) | np.isnan(offsets))
    if m.any():
        out["qrs_width_mean"] = np.mean((offsets[m] - onsets[m]) / fs * 1000.0)
    r_amp = sig_lead[np.clip(rpeaks, 0, len(sig_lead) - 1)]
    out["qrs_amp_mean"], out["qrs_amp_std"] = np.mean(r_amp), np.std(r_amp)

    # Beat-template consistency: correlate each beat to the median beat.
    half = int(0.25 * fs)
    beats = [sig_lead[r - half:r + half] for r in rpeaks
             if r - half >= 0 and r + half <= len(sig_lead)]
    if len(beats) >= 3:
        B = np.vstack(beats)
        template = np.median(B, axis=0)
        corrs = [np.corrcoef(b, template)[0, 1] for b in B]
        out["beat_corr_mean"], out["beat_corr_std"] = np.nanmean(corrs), np.nanstd(corrs)
    return out


def rhythm_morph(recordings) -> np.ndarray:
    import neurokit2 as nk

    lead_idx = config.PTBXL_LEAD_NAMES.index(config.RHYTHM_LEAD) \
        if recordings[0].signal.shape[1] == 12 else 0

    rows = []
    for r in recordings:
        fs = r.fs
        sig_lead = r.signal[:, lead_idx]
        feat = dict.fromkeys(RHYTHM_MORPH_NAMES, np.nan)
        try:
            cleaned = nk.ecg_clean(sig_lead, sampling_rate=fs)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
            feat.update(_rr_features(rpeaks, fs))
            waves = {}
            if len(rpeaks) >= 3:
                _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")
            feat.update(_morphology_features(cleaned, rpeaks, fs, waves))
        except Exception:
            pass  # leave NaNs; imputed downstream
        rows.append([feat[k] for k in RHYTHM_MORPH_NAMES])
    X = np.asarray(rows, dtype=np.float32)
    # column-median imputation for the occasional failed delineation
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])
    return X


def build_representation(name: str, recordings) -> np.ndarray:
    if name == "raw_flatten":
        return raw_flatten(recordings)
    if name == "rhythm_morph":
        return rhythm_morph(recordings)
    if name == "deep_ae":
        from core.deep_repr import deep_ae_embedding
        return deep_ae_embedding(recordings)
    raise ValueError(f"unknown representation: {name}")
