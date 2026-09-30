"""Download a big Zenodo zip once and keep only the members we need.

Zenodo rate-limits anonymous clients (~60 requests/minute; parallel per-file range requests get
HTTP 429), so the robust route is one streaming, resumable download of the whole zip, extraction of
just the wanted files, then deleting the zip.
"""

import hashlib
import time
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

import requests


def md5sum(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, retries: int = 8, verbose: bool = True) -> Path:
    """Streaming download with resume (HTTP Range) and back-off on 429/5xx."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            pos = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={pos}-"} if pos else {}
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code == 416:  # nothing left to fetch
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(int(r.headers.get("Retry-After", 2**attempt * 5)), 120))
                    continue
                r.raise_for_status()
                if pos and r.status_code != 206:  # server ignored Range: start over
                    pos = 0
                total = int(r.headers.get("Content-Length", 0)) + pos
                next_report = pos + (250 << 20)
                with open(part, "ab" if pos else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        pos += len(chunk)
                        if verbose and pos >= next_report:
                            print(f"  {pos / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
                            next_report += 250 << 20
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(min(2**attempt * 2, 60))
    else:
        raise RuntimeError(f"gave up downloading {url}")
    part.rename(dest)
    return dest


def fetch_members(
    url: str,
    dest: Path,
    include: Callable[[str], bool],
    md5: Optional[str] = None,
    verbose: bool = True,
    marker_name: str = ".complete",
) -> List[Path]:
    """Download the zip at ``url``, extract members whose name satisfies ``include`` under
    ``dest``, delete the zip. Skips everything if all wanted files are already extracted.

    ``marker_name``: the "already done" marker is scoped to what was actually extracted, not just
    to ``dest``. Two calls with different ``include`` predicates but the same ``dest`` (e.g.
    Saraga's text-only vs. text+audio downloads share a folder) must use different ``marker_name``s
    -- otherwise the second call would see the first's marker and skip, silently not fetching the
    extra files it was asked for. Reusing the same ``marker_name`` for the same ``include`` logic
    is what makes the skip-if-already-done behaviour work at all.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / marker_name  # written when a previous run finished extracting this variant
    if marker.exists():
        if verbose:
            print(f"{dest} already downloaded and extracted (found {marker}), reusing -- not re-downloading.", flush=True)
        return sorted(p for p in dest.rglob("*") if p.is_file() and include(p.relative_to(dest).as_posix()))
    zip_path = dest / "_download.zip"
    if verbose:
        print(f"downloading {url.split('?')[0].split('/')[-1]} ...", flush=True)
    download_file(url, zip_path, verbose=verbose)
    if md5 and md5sum(zip_path) != md5:
        zip_path.unlink()
        raise IOError("checksum mismatch, download deleted; run again")
    out = []
    with zipfile.ZipFile(zip_path) as z:
        wanted = [i for i in z.infolist() if not i.is_dir() and include(i.filename)]
        for i, info in enumerate(wanted, 1):
            out.append(Path(z.extract(info, dest)))  # zipfile strips '..' and absolute paths
            if verbose and (i % 200 == 0 or i == len(wanted)):
                print(f"  extracted {i}/{len(wanted)}", flush=True)
    zip_path.unlink()
    marker.write_text("ok")
    return out
