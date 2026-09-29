"""Tonic-normalised pitch features.

A raga is (largely) defined relative to the performer's Sa (tonic), so every pitch value is
converted to cents above the tonic and folded into one octave. Frequencies <= 0 mean unvoiced.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def load_pitch_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a CompMusic/Saraga pitch file (tab-separated ``time_s  freq_hz``). Fast C parser."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], dtype=np.float64)
    return df[0].to_numpy(), df[1].to_numpy()


def frame_step(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else 0.0


def pitch_stats(times: np.ndarray, freqs: np.ndarray) -> dict:
    dt = frame_step(times)
    voiced = np.isfinite(freqs) & (freqs > 0)
    return {
        "duration_s": float(times[-1]) if len(times) else 0.0,
        "voiced_s": float(voiced.sum() * dt),
        "frame_step_s": dt,
    }


def cents_above_tonic(freqs: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Cents above Sa (NaN where unvoiced). Not octave-folded."""
    freqs = np.asarray(freqs, dtype=np.float64)
    out = np.full(freqs.shape, np.nan)
    voiced = np.isfinite(freqs) & (freqs > 0)
    out[voiced] = 1200.0 * np.log2(freqs[voiced] / tonic_hz)
    return out


def _bin_index(cents: np.ndarray, n_bins: int) -> np.ndarray:
    """Octave-folded bin index, with bin 0 centred on Sa."""
    width = 1200.0 / n_bins
    return np.floor((cents + width / 2.0) / width).astype(np.int64) % n_bins


def pitch_class_histogram(
    freqs: np.ndarray, tonic_hz: float, n_bins: int = 120, smooth_bins: float = 1.0
) -> np.ndarray:
    """Normalised (sums to 1) octave-folded histogram of pitch relative to the tonic."""
    cents = cents_above_tonic(freqs, tonic_hz)
    cents = cents[np.isfinite(cents)]
    hist = np.bincount(_bin_index(cents, n_bins), minlength=n_bins).astype(np.float64)
    return _smooth_and_normalise(hist[None, :], smooth_bins)[0]


def _smooth_and_normalise(hists: np.ndarray, smooth_bins: float) -> np.ndarray:
    if smooth_bins > 0:
        hists = gaussian_filter1d(hists, smooth_bins, axis=1, mode="wrap")
    total = hists.sum(axis=1, keepdims=True)
    return np.divide(hists, total, out=np.zeros_like(hists), where=total > 0)


def chunk_histograms(
    times: np.ndarray,
    freqs: np.ndarray,
    tonic_hz: float,
    chunk_s: float = 30.0,
    hop_s: float = 15.0,
    n_bins: int = 120,
    smooth_bins: float = 1.0,
    min_voiced_s: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window over the recording and histogram each chunk.

    Returns ``(hists, starts_s)`` with ``hists`` of shape (n_chunks, n_bins). Chunks with less
    than ``min_voiced_s`` seconds of voiced pitch (silence, tanpura only) are dropped.
    """
    if len(times) < 2:
        return np.zeros((0, n_bins)), np.zeros(0)
    dt = frame_step(times)
    cents = cents_above_tonic(freqs, tonic_hz)
    voiced = np.isfinite(cents)
    idx = np.zeros(len(times), dtype=np.int64)
    idx[voiced] = _bin_index(cents[voiced], n_bins)

    starts = np.arange(0.0, max(times[-1] - chunk_s, 0.0) + 1e-9, hop_s)
    lo = np.searchsorted(times, starts, side="left")
    hi = np.searchsorted(times, starts + chunk_s, side="left")
    hists, kept = [], []
    for start, a, b in zip(starts, lo, hi):
        v = voiced[a:b]
        if v.sum() * dt < min_voiced_s:
            continue
        hists.append(np.bincount(idx[a:b][v], minlength=n_bins))
        kept.append(start)
    if not hists:
        return np.zeros((0, n_bins)), np.zeros(0)
    hists = _smooth_and_normalise(np.asarray(hists, dtype=np.float64), smooth_bins)
    return hists, np.asarray(kept)
