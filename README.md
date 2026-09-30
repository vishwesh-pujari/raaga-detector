# raaga-detector

Hindustani raga recognition from audio (research project, non-commercial). See [PLAN.md](PLAN.md) for the plan and current status, and [CONCEPTS.md](CONCEPTS.md) for what the terms mean and why things are done this way.

## Run on Colab / Kaggle

1. Push this repo to GitHub, open a notebook from `notebooks/` in Colab (File > Open notebook > GitHub).
2. Run `01_data_audit.ipynb`, then `02_baseline_pitch_histogram.ipynb`. CPU runtime is enough for these two.

Storage: the big Zenodo zips (~7.7 GB combined) are downloaded to Colab/Kaggle's own local disk and trimmed down to their pitch/tonic files (~2.5 GB) there -- never to your Google Drive. Drive is only used, optionally, to persist small results (catalog table, split file, caches; well under 100 MB) between sessions. It's off by default (`MOUNT_DRIVE = False` in the setup cell); notebook 01 instead ends by downloading `catalog.csv` and `splits/v1.csv` straight to your computer. Set `MOUNT_DRIVE = True` if you have Drive space and want caches to survive a session restart.

## Local (tests only)

```bash
pip install -e ".[dev]"
pytest
```

Training is meant to run on free cloud GPUs, not locally.

## Data

| Dataset | Licence | Used for |
|---|---|---|
| CompMusic Hindustani Music Dataset (Zenodo 7278506) | CC BY 4.0 | main training data (pitch, tonic, labels) |
| Saraga Hindustani 1.5 (Zenodo 4301737) | CC BY-NC-SA 4.0 | extra recordings, audio for front-end tests |

Raw data and caches are never committed. Only `data/splits/*.csv` (ids and labels) are.
