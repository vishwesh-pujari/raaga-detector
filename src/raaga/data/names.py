"""Raga-name normalisation so the same raga spelled differently across datasets gets one label."""

import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

import yaml

_FILLER = re.compile(r"\b(raag|raaga|raga|rag)\b")


def slug(name: str) -> str:
    """Lowercase, ASCII, no filler words/punctuation, repeated letters collapsed, w->v.

    'Rāg Asawari' and 'Asavari' both map to 'asavari'. Differences that this cannot bridge
    (e.g. 'Bhoopali' vs 'Bhupali') go in configs/raga_aliases.yaml.
    """
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = _FILLER.sub(" ", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    return text.replace("w", "v")


def load_aliases(path: Optional[Path]) -> Dict[str, str]:
    """YAML mapping ``canonical name: [spelling, spelling, ...]`` -> ``{slug(spelling): slug(canonical)}``."""
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    out = {}
    for canonical, spellings in raw.items():
        for spelling in spellings or []:
            out[slug(spelling)] = slug(canonical)
    return out


def normalize_raga(name: Optional[str], aliases: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not name or not str(name).strip():
        return None
    key = slug(str(name))
    if not key:
        return None
    return (aliases or {}).get(key, key)
