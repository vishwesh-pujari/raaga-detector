"""Compute pitch-histogram chunks once per recording and cache them as small ``.npz`` files.

Training never touches the big raw pitch files again, so the cache (a few MB) can live on Drive.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from raaga.features import pitch as P


def cache_dir(root: Path, n_bins: int, chunk_s: float, hop_s: float, smooth_bins: float, variant: str) -> Path:
    name = f"hist_b{n_bins}_c{chunk_s:g}_h{hop_s:g}_s{smooth_bins:g}_{variant}"
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file(cache: Path, uid: str) -> Path:
    return cache / (uid.replace(":", "__").replace("/", "_") + ".npz")


def build(
    catalog: pd.DataFrame,
    cache: Path,
    n_bins: int = 120,
    chunk_s: float = 30.0,
    hop_s: float = 15.0,
    smooth_bins: float = 1.0,
    min_voiced_s: float = 5.0,
    verbose: bool = True,
) -> None:
    """Requires columns: uid, pitch_path, tonic_hz. Skips recordings already cached."""
    for i, row in enumerate(catalog.itertuples(index=False)):
        out = _file(cache, row.uid)
        if out.exists():
            continue
        times, freqs = P.load_pitch_file(row.pitch_path)
        hists, starts = P.chunk_histograms(
            times, freqs, row.tonic_hz, chunk_s, hop_s, n_bins, smooth_bins, min_voiced_s
        )
        np.savez_compressed(out, hists=hists.astype(np.float32), starts=starts)
        if verbose:
            print(f"[{i + 1}/{len(catalog)}] {row.uid}: {len(hists)} chunks")


@dataclass
class ChunkSet:
    X: np.ndarray  # (n_chunks, n_bins)
    y: np.ndarray  # (n_chunks,) raga label (str)
    uid: np.ndarray  # (n_chunks,) recording id
    start_s: np.ndarray  # (n_chunks,)

    def subset(self, uids: Iterable[str]) -> "ChunkSet":
        mask = np.isin(self.uid, list(uids))
        return ChunkSet(self.X[mask], self.y[mask], self.uid[mask], self.start_s[mask])


def load(catalog: pd.DataFrame, cache: Path, label_col: str = "raga") -> ChunkSet:
    """Concatenate the cached chunks of every recording in ``catalog`` (needs uid + label_col)."""
    X, y, uid, start = [], [], [], []
    missing = []
    for row in catalog.itertuples(index=False):
        f = _file(cache, row.uid)
        if not f.exists():
            missing.append(row.uid)
            continue
        with np.load(f) as z:
            h, s = z["hists"], z["starts"]
        X.append(h)
        y += [getattr(row, label_col)] * len(h)
        uid += [row.uid] * len(h)
        start.append(s)
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings not cached, e.g. {missing[:3]}. Run cache.build first.")
    return ChunkSet(np.concatenate(X), np.asarray(y), np.asarray(uid), np.concatenate(start))
