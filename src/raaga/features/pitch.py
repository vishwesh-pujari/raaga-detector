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


def chunk_pitch_sequences(
    times: np.ndarray,
    freqs: np.ndarray,
    tonic_hz: float,
    chunk_s: float = 30.0,
    hop_s: float = 15.0,
    step_s: float = 0.1,
    min_voiced_s: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Chunk a recording into fixed-length pitch *sequences* for M1 (unlike ``chunk_histograms``,
    which collapses each chunk to a single time-averaged vector, this keeps melodic order/movement).

    Each chunk becomes a ``(T, 3)`` sequence, ``T = round(chunk_s / step_s)`` steps of ``step_s``
    seconds each. A step's 3 features are ``(mean_sin, mean_cos, voiced_frac)``, where
    ``mean_sin``/``mean_cos`` are the mean of ``sin``/``cos`` of ``2*pi*cents_above_tonic/1200``
    (the octave-folded angle) over that step's voiced native frames -- this is the standard
    circular-mean trick: it's automatically octave-invariant (sin/cos are 2*pi-periodic) and its
    magnitude naturally shrinks towards 0 when the pitch is unstable/noisy within the step, instead
    of picking an arbitrary single value. A fully-unvoiced step is exactly ``(0, 0, 0)``.

    Returns ``(seqs, starts_s)`` with ``seqs`` of shape ``(n_chunks, T, 3)``. Chunks with less than
    ``min_voiced_s`` seconds of voiced pitch are dropped, same as ``chunk_histograms``.
    """
    T = round(chunk_s / step_s)
    if len(times) < 2:
        return np.zeros((0, T, 3)), np.zeros(0)
    dt = frame_step(times)
    cents = cents_above_tonic(freqs, tonic_hz)
    voiced = np.isfinite(cents)
    theta = np.zeros(len(times))
    theta[voiced] = 2 * np.pi * cents[voiced] / 1200.0
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    starts = np.arange(0.0, max(times[-1] - chunk_s, 0.0) + 1e-9, hop_s)
    lo = np.searchsorted(times, starts, side="left")
    hi = np.searchsorted(times, starts + chunk_s, side="left")
    seqs, kept = [], []
    for start, a, b in zip(starts, lo, hi):
        v = voiced[a:b]
        if v.sum() * dt < min_voiced_s:
            continue
        step_idx = np.clip(((times[a:b] - start) / step_s).astype(np.int64), 0, T - 1)
        counts = np.bincount(step_idx, minlength=T).astype(np.float64)
        vcounts = np.bincount(step_idx[v], minlength=T).astype(np.float64)
        sin_sum = np.bincount(step_idx[v], weights=sin_t[a:b][v], minlength=T)
        cos_sum = np.bincount(step_idx[v], weights=cos_t[a:b][v], minlength=T)
        safe = np.where(vcounts > 0, vcounts, 1.0)
        seq = np.stack([sin_sum / safe, cos_sum / safe, vcounts / np.where(counts > 0, counts, 1.0)], axis=1)
        seqs.append(seq)
        kept.append(start)
    if not seqs:
        return np.zeros((0, T, 3)), np.zeros(0)
    return np.asarray(seqs, dtype=np.float32), np.asarray(kept)


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
