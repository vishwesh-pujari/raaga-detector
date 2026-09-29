"""Saraga Hindustani 1.5: 108 tracks, 61 ragas (Zenodo 4301737, CC BY-NC-SA 4.0: non-commercial only).

Each track folder has ``<name>.json`` (metadata incl. raga and lead instrument), ``<name>.pitch.txt``
and ``<name>.ctonic.txt``. We skip the mp3s (~4 GB of the zip).
"""

import json
from pathlib import Path
from typing import List, Optional

URL = "https://zenodo.org/record/4301737/files/saraga1.5_hindustani.zip?download=1"


def home(raw_home: Path) -> Path:
    return Path(raw_home) / "saraga_hindustani" / "saraga1.5_hindustani"


def _wanted(name: str) -> bool:
    return "__MACOSX" not in name and name.endswith((".json", ".pitch.txt", ".ctonic.txt", "file_paths.csv"))


def download(raw_home: Path) -> None:
    """Downloads the ~4.1 GB zip once, keeps metadata, pitch and tonic (no audio), deletes the zip."""
    from raaga.data.remote import fetch_members

    fetch_members(URL, Path(raw_home) / "saraga_hindustani", _wanted, md5="ea9ed2885ea37a1b10e42f60cf299702")


def vocal_class(metadata: dict) -> str:
    """'vocal' if a lead artist plays Voice, 'instrumental' if the lead is another instrument."""
    leads = [a for a in metadata.get("artists", []) if a.get("lead")]
    if not leads:
        return "unknown"
    names = {(a.get("instrument") or {}).get("name", "").lower() for a in leads}
    return "vocal" if names & {"voice", "vocals"} else "instrumental"


def _read_float(path: str) -> Optional[float]:
    try:
        with open(path) as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def rows(raw_home: Path) -> List[dict]:
    out = []
    for jf in sorted(home(raw_home).glob("*/*/*.json")):
        stem = str(jf)[: -len(".json")]
        with open(jf) as f:
            md = json.load(f)
        raags = md.get("raags") or []
        # a track with several ragas has no single label: leave it empty so it is dropped later
        raga = (raags[0].get("name") or raags[0].get("common_name")) if len(raags) == 1 else None
        album_artists = md.get("album_artists") or []
        lead = [a for a in md.get("artists", []) if a.get("lead")]
        artist = (album_artists[0] if album_artists else (lead[0]["artist"] if lead else {})).get("name")
        out.append(
            dict(
                uid=f"saraga:{md.get('mbid') or jf.stem}",
                dataset="saraga",
                track_id=jf.stem,
                raga_raw=raga,
                artist=artist,
                concert=jf.parent.parent.name,  # album folder
                mbid=md.get("mbid"),
                tonic_hz=_read_float(f"{stem}.ctonic.txt"),
                pitch_path=f"{stem}.pitch.txt",
                vocal_class=vocal_class(md),
            )
        )
    return out
