"""Where things live on Colab / Kaggle / a plain machine.

Persistent, small stuff (catalog, feature caches, splits, checkpoints) goes under ``root()`` which
is Google Drive on Colab and ``/kaggle/working`` on Kaggle. Big raw datasets (mirdata downloads)
go to fast local disk and are simply re-downloaded in a new session.
"""

import os
from pathlib import Path


def root() -> Path:
    """Persistent working directory. Override with the RAAGA_ROOT environment variable."""
    override = os.environ.get("RAAGA_ROOT")
    if override:
        path = Path(override)
    elif Path("/content/drive/MyDrive").exists():
        path = Path("/content/drive/MyDrive/raaga-work")
    elif Path("/kaggle/working").exists():
        path = Path("/kaggle/working/raaga-work")
    else:
        path = Path.cwd() / "workdir"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_home() -> Path:
    """Fast local disk for large raw downloads (not persisted across sessions)."""
    override = os.environ.get("RAAGA_RAW")
    if override:
        path = Path(override)
    elif Path("/content").exists():
        path = Path("/content/mir_datasets")
    elif Path("/kaggle/working").exists():
        path = Path("/kaggle/temp/mir_datasets")
    else:
        path = Path.cwd() / "workdir" / "mir_datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def subdir(name: str) -> Path:
    path = root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
