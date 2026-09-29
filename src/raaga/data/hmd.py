"""CompMusic Hindustani Music Dataset (HMD): 300 recordings, 30 ragas x 10 recordings.

Pitch, tonic and raga labels are open (Zenodo 7278506, CC BY 4.0). The audio is restricted (request
on Zenodo 7278511) and not needed for the pitch-based models.

mirdata's ``compmusic_raga`` index only lists the Carnatic half, so we read the files directly.
"""

import json
from pathlib import Path
from typing import List, Optional

URL = (
    "https://zenodo.org/record/7278506/files/"
    "Indian%20Art%20Music%20Raga%20Recognition%20Dataset%20%28features%29.zip?download=1"
)


def home(raw_home: Path) -> Path:
    return Path(raw_home) / "compmusic_raga" / "RagaDataset" / "Hindustani"


def _wanted(name: str) -> bool:
    if "__MACOSX" in name or "/Hindustani/" not in name:
        return False
    if "/_info_/" in name:
        return name.endswith(".json")
    return name.endswith((".pitch", ".tonic", ".tonicFine"))


def download(raw_home: Path) -> None:
    """Downloads the 3.6 GB zip once, keeps the Hindustani pitch/tonic files (~2.3 GB) + label JSONs, deletes the zip."""
    from raaga.data.remote import fetch_members

    fetch_members(URL, Path(raw_home) / "compmusic_raga", _wanted, md5="5dfc26dd1c2652ab75a62faec7f45f08")


def _read_float(path: str) -> Optional[float]:
    try:
        with open(path) as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def rows(raw_home: Path, tonic_kind: str = "tonic") -> List[dict]:
    """One dict per recording. ``tonic_kind``: 'tonic' or 'tonicFine' (manually fine-tuned)."""
    base = home(raw_home)
    with open(base / "_info_" / "path_mbid_ragaid.json") as f:
        meta = json.load(f)
    with open(base / "_info_" / "ragaId_to_ragaName_mapping.json") as f:
        names = json.load(f)
    out = []
    for mbid, v in meta.items():
        # RagaDataset/Hindustani/audio/<ragaid>/<artist>/<concert>/<recording>
        parts = v["path"].split("/")
        feature_stem = str(base / "features" / "/".join(parts[3:]))
        out.append(
            dict(
                uid=f"hmd:{mbid}",
                dataset="hmd",
                track_id=mbid,
                raga_raw=names[v["ragaid"]],
                artist=parts[4],
                concert=parts[5],
                mbid=mbid,
                tonic_hz=_read_float(f"{feature_stem}.{tonic_kind}"),
                pitch_path=f"{feature_stem}.pitch",
                vocal_class="unknown",  # see data/musicbrainz.py
            )
        )
    return out
