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


def _wanted_with_audio(name: str) -> bool:
    return _wanted(name) or ("__MACOSX" not in name and name.endswith(".mp3.mp3"))


def download(raw_home: Path, include_audio: bool = False) -> None:
    """Downloads the ~4.1 GB zip once and keeps metadata, pitch and tonic, deleting the zip.

    ``include_audio=False`` (the default, used by every existing notebook): skips the ~3.9 GB of
    mp3s, since B1/M1 never need Saraga's audio. Set ``True`` for Phase 4a's extractor validation,
    which specifically does need it -- this downloads the full zip either way (Zenodo doesn't
    support partial zip download, see ``remote.py``), just keeps more of it.
    """
    from raaga.data.remote import fetch_members

    wanted = _wanted_with_audio if include_audio else _wanted
    marker = ".complete_with_audio" if include_audio else ".complete"
    fetch_members(
        URL, Path(raw_home) / "saraga_hindustani", wanted, md5="ea9ed2885ea37a1b10e42f60cf299702", marker_name=marker
    )


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
                audio_path=f"{stem}.mp3.mp3",  # only present if downloaded with include_audio=True
                vocal_class=vocal_class(md),
            )
        )
    return out
