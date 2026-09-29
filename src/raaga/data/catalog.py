"""One table describing every usable Hindustani recording, regardless of source dataset.

Columns: uid, dataset, track_id, raga_raw, raga (normalised), artist, concert, mbid, tonic_hz,
pitch_path, has_pitch, vocal_class, plus duration_s / voiced_s after ``add_pitch_stats``.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from raaga.data import hmd, saraga
from raaga.data.names import load_aliases, normalize_raga
from raaga.features import pitch as P


def build_catalog(
    raw_home: Path,
    include_saraga: bool = True,
    tonic_kind: str = "tonic",
    raga_aliases: Optional[Path] = None,
    artist_aliases: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the recording table from datasets already downloaded under ``raw_home``.

    tonic_kind: HMD tonic file, 'tonic' or 'tonicFine' (manually fine-tuned). Saraga has one tonic.
    """
    rows = hmd.rows(raw_home, tonic_kind)
    if include_saraga:
        rows += saraga.rows(raw_home)
    df = pd.DataFrame(rows)
    ra, aa = load_aliases(raga_aliases), load_aliases(artist_aliases)
    df["raga"] = df["raga_raw"].map(lambda n: normalize_raga(n, ra))
    df["artist"] = df["artist"].map(lambda n: normalize_raga(n, aa) if isinstance(n, str) else n)
    df["has_pitch"] = df["pitch_path"].map(lambda p: bool(p) and os.path.exists(p))
    # The same recording can be in both datasets; keep the first (HMD, listed first).
    has_mbid = df["mbid"].notna() & (df["mbid"] != "")
    dup = df.duplicated("mbid", keep="first") & has_mbid
    return df[~dup].reset_index(drop=True)


def add_pitch_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration_s / voiced_s by reading each pitch file (~1-2 min for the whole of HMD)."""
    stats = []
    for row in df.itertuples(index=False):
        if row.has_pitch:
            t, f = P.load_pitch_file(row.pitch_path)
            stats.append(P.pitch_stats(t, f))
        else:
            stats.append({"duration_s": None, "voiced_s": None, "frame_step_s": None})
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(stats)], axis=1)


def usable(df: pd.DataFrame, vocal_ok=("vocal", "unknown")) -> pd.DataFrame:
    """Recordings with a label, a tonic and a pitch track, and a vocal class we accept."""
    ok = df["raga"].notna() & df["tonic_hz"].notna() & df["has_pitch"] & df["vocal_class"].isin(vocal_ok)
    return df[ok].reset_index(drop=True)
