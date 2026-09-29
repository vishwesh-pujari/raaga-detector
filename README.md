# raaga-detector

Hindustani raga recognition from audio (research project, non-commercial). See [PLAN.md](PLAN.md) for the plan and current status.

## Run on Colab / Kaggle

1. Push this repo to GitHub, open a notebook from `notebooks/` in Colab (File > Open notebook > GitHub).
2. Run `01_data_audit.ipynb`, then `02_baseline_pitch_histogram.ipynb`. CPU runtime is enough for these two.

Each notebook clones the repo, installs it and stores caches/splits/results on Google Drive (`raaga-work/`) or `/kaggle/working`.

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
