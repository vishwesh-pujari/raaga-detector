"""CompMusic Hindustani Music Dataset (HMD): 300 recordings, 30 ragas x 10 recordings.

Pitch, tonic and raga labels are open (Zenodo 7278506, CC BY 4.0). The audio is restricted (request
on Zenodo 7278511) and not needed for the pitch-based models.

mirdata's ``compmusic_raga`` index only lists the Carnatic half, so we read the files directly.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# Every feature filename ends in "_<mbid>": e.g. "Raga_Bhageshri_6cb0fc24-...-8552b70127ea.tonic".
_MBID_SUFFIX = re.compile(r"_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")

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


def _index_by_mbid(base: Path) -> Dict[str, str]:
    """Map mbid -> extension-less feature path, by scanning the extracted ``.tonic`` files.

    ``path_mbid_ragaid.json``'s ``path`` field is not reliable for this: the zip's folder names
    have ``:`` and ``&`` sanitised to ``_`` (Windows-illegal filename characters), but the JSON
    keeps the original punctuation. E.g. json path ".../Raag_Marwa_&_Hameer/..." but the actual
    folder on disk is ".../Raag_Marwa___Hameer/...". Matching by mbid suffix instead of
    reconstructing the path sidesteps this (verified: 300/300 vs 236/300 recordings found).
    """
    out = {}
    for f in (base / "features").rglob("*.tonic"):
        m = _MBID_SUFFIX.search(f.stem)
        if m:
            out[m.group(1)] = str(f)[: -len(".tonic")]
    return out


def rows(raw_home: Path, tonic_kind: str = "tonic") -> List[dict]:
    """One dict per recording. ``tonic_kind``: 'tonic' or 'tonicFine' (manually fine-tuned)."""
    base = home(raw_home)
    with open(base / "_info_" / "path_mbid_ragaid.json") as f:
        meta = json.load(f)
    with open(base / "_info_" / "ragaId_to_ragaName_mapping.json") as f:
        names = json.load(f)
    by_mbid = _index_by_mbid(base)
    out = []
    for mbid, v in meta.items():
        # RagaDataset/Hindustani/audio/<ragaid>/<artist>/<concert>/<recording> -- only used for the
        # artist/concert grouping labels below, not for locating files (see _index_by_mbid).
        parts = v["path"].split("/")
        stem = by_mbid.get(mbid)
        out.append(
            dict(
                uid=f"hmd:{mbid}",
                dataset="hmd",
                track_id=mbid,
                raga_raw=names[v["ragaid"]],
                artist=parts[4],
                concert=parts[5],
                mbid=mbid,
                tonic_hz=_read_float(f"{stem}.{tonic_kind}") if stem else None,
                pitch_path=f"{stem}.pitch" if stem else None,
                vocal_class="unknown",  # see data/musicbrainz.py
            )
        )
    return out
