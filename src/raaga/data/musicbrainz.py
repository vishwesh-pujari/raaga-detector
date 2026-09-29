"""Find out whether a recording is vocal, from its MusicBrainz artist relationships.

The CompMusic raga dataset has no instrument metadata, but every recording has an MBID and
MusicBrainz often lists ``vocal`` / ``instrument`` credits. Coverage is patchy, so the result
is one of ``vocal`` / ``instrumental`` / ``unknown`` and the ``unknown`` ones need a manual look.
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests

API = "https://musicbrainz.org/ws/2/recording/{mbid}"
# MusicBrainz asks for a User-Agent that identifies the app and a way to reach the author.
USER_AGENT = "raaga-detector/0.1 (research; https://github.com/vishwesh-pujari/raaga-detector)"


def classify_recording(recording_json: dict) -> dict:
    """Pure function: MusicBrainz recording JSON (``inc=artist-rels``) -> vocal/instrumental/unknown."""
    vocal, instruments = False, set()
    for rel in recording_json.get("relations", []):
        kind = rel.get("type")
        if kind == "vocal":
            vocal = True
        elif kind == "instrument":
            instruments.update(a.lower() for a in rel.get("attributes", []))
    label = "vocal" if vocal else ("instrumental" if instruments else "unknown")
    return {"vocal_class": label, "instruments": sorted(instruments)}


def fetch(mbid: str, cache_dir: Path, pause_s: float = 1.1) -> Optional[dict]:
    """Fetch (with on-disk JSON cache, 1 request/second as MusicBrainz requires) and classify."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{mbid}.json"
    if f.exists():
        return classify_recording(json.loads(f.read_text()))
    resp = requests.get(
        API.format(mbid=mbid),
        params={"inc": "artist-rels", "fmt": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    time.sleep(pause_s)
    if resp.status_code == 404:
        f.write_text(json.dumps({"relations": []}))
        return classify_recording({})
    resp.raise_for_status()
    f.write_text(resp.text)
    return classify_recording(resp.json())
