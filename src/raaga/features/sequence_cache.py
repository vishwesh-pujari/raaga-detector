"""Compute pitch-sequence chunks (for M1) once per recording and cache them as small ``.npz``
files -- the sequence analogue of ``cache.py`` (which caches histograms, for B1).

Kept as a separate module rather than folded into ``cache.py``: the two representations (a single
time-averaged vector per chunk vs. an ordered sequence per chunk) have different build logic and
different cache-folder naming, but share the same ``ChunkSet`` container and per-uid file naming.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from raaga.features import pitch as P
from raaga.features.cache import ChunkSet, _file  # same file-naming convention as the histogram cache

__all__ = ["cache_dir", "build", "load", "ChunkSet"]


def cache_dir(root: Path, chunk_s: float, hop_s: float, step_s: float, variant: str) -> Path:
    name = f"seq_c{chunk_s:g}_h{hop_s:g}_s{step_s:g}_{variant}"
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def build(
    catalog: pd.DataFrame,
    cache: Path,
    chunk_s: float = 30.0,
    hop_s: float = 15.0,
    step_s: float = 0.1,
    min_voiced_s: float = 5.0,
    verbose: bool = True,
) -> None:
    """Requires columns: uid, pitch_path, tonic_hz. Skips recordings already cached."""
    for i, row in enumerate(catalog.itertuples(index=False)):
        out = _file(cache, row.uid)
        if out.exists():
            continue
        times, freqs = P.load_pitch_file(row.pitch_path)
        seqs, starts = P.chunk_pitch_sequences(times, freqs, row.tonic_hz, chunk_s, hop_s, step_s, min_voiced_s)
        np.savez_compressed(out, seqs=seqs.astype(np.float32), starts=starts)
        if verbose:
            print(f"[{i + 1}/{len(catalog)}] {row.uid}: {len(seqs)} chunks")


def load(catalog: pd.DataFrame, cache: Path, label_col: str = "raga") -> ChunkSet:
    """Concatenate the cached chunks of every recording in ``catalog`` (needs uid + label_col).
    ``ChunkSet.X`` is ``(n_chunks, T, 3)`` here, instead of histogram's ``(n_chunks, n_bins)``."""
    X, y, uid, start = [], [], [], []
    missing = []
    for row in catalog.itertuples(index=False):
        f = _file(cache, row.uid)
        if not f.exists():
            missing.append(row.uid)
            continue
        with np.load(f) as z:
            s_arr, starts = z["seqs"], z["starts"]
        X.append(s_arr)
        y += [getattr(row, label_col)] * len(s_arr)
        uid += [row.uid] * len(s_arr)
        start.append(starts)
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings not cached, e.g. {missing[:3]}. Run build first.")
    return ChunkSet(np.concatenate(X), np.asarray(y), np.asarray(uid), np.concatenate(start))
