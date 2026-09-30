# Raaga Detector: Project Plan

Goal: a web app that takes an audio clip of **Hindustani** classical music and returns a probability distribution over ragas.
Current focus: **the deep-learning model only** (Phases 0-5). The web app (Phase 6) comes after.
All training runs on free cloud GPUs/CPUs (Colab / Kaggle), never locally.

_Last updated: 2026-09-28. Sections marked **(verified)** were checked against the actual archives; the rest comes from papers/search summaries and should be treated as "to verify"._

---

## 0. Decisions so far

| Date | Decision | Consequence |
|---|---|---|
| 2026-09-28 | **Vocal only** at first | Drop instrumental recordings. Saraga labels the lead instrument, so it is filtered exactly. HMD has no instrument field: its 55 artists look like vocalists (possible exception: Gopal Mishra, 4 recordings), confirmed via MusicBrainz in notebook 01. |
| 2026-09-28 | **No fixed raga list**: use whatever the data supports | Label set = ragas with enough recordings after merging datasets (HMD alone gives 30). |
| 2026-09-28 | **Non-commercial use only** | Saraga (CC BY-NC-SA 4.0) can be used. Re-check licences before any public deployment of trained weights. |
| 2026-09-29 | **Bug found and fixed**: HMD loader was silently dropping recordings | `path_mbid_ragaid.json`'s `path` field keeps punctuation (`:`, `&`) that the archive's real folder names sanitise to `_`. Reconstructing the feature-file path from that field missed any recording whose concert name has one of those characters -- 64 of 300 (21%) in the first audit run. Fixed by matching files via the mbid every filename ends in, instead of rebuilding the path from text (`src/raaga/data/hmd.py`). Verified twice: against the real archive listing (300/300 matched) and, after a genuine re-run (needed a Colab runtime restart -- see below), against the actual regenerated catalog (0 missing tonic/pitch, up from 64). |
| 2026-09-29 | Colab silently reused a stale kernel on the first "re-run" | Restarting cells / opening a "new" notebook does not guarantee a fresh Python process. An already-`import`-ed module keeps running from memory even after `git pull` updates the file on disk and `pip install -e .` reruns. The giveaway: the output was byte-identical to the pre-fix run, including a `pitch_path` string only the *old* code could construct. Fix: **Runtime -> Restart runtime** (or "Disconnect and delete runtime") before re-running, not just re-running cells. |

---

## 1. What exists already

### Datasets

| Dataset | Content | Access | Notes |
|---|---|---|---|
| **HMD**: CompMusic Hindustani Music Dataset (Zenodo [7278506](https://zenodo.org/records/7278506), [details](https://compmusic.upf.edu/node/328)) **(verified)** | 300 recordings, **30 ragas x 10 recordings**, ~116 h, 55 artists, 146 concerts. Per recording: raw predominant pitch (4.4 ms hop), tonic (auto and manually fine-tuned), raga label, MBID. | **Pitch/tonic/labels open** (CC BY 4.0). **Audio restricted**, request via Zenodo [7278511](https://zenodo.org/record/7278511). | Primary dataset. Balanced (10 per raga). The pitch files make a pitch-based model possible without the audio. |
| **Saraga Hindustani 1.5** ([Zenodo 4301737](https://mtg.github.io/saraga/access.html)) **(verified)** | 108 tracks, ~44 h, 61 ragas, metadata with **lead instrument**, pitch, tonic. | Open. CC BY-NC-SA 4.0. Zip is 4.1 GB, of which 3.9 GB is mp3. | Too few tracks per raga to train alone. Adds recordings to HMD ragas and gives extra ragas. Audio is included, so it is the only source of raw audio for testing an audio-based front-end. |
| PIM-v1 (Prasar Bharati) from [Explainable DL for Raga ID](https://arxiv.org/html/2406.02443v1) | Reported 191 h, 501 recordings, 144 ragas | **Unknown**, need to find out if public | Largest labeled Hindustani set reported. |
| [Kaggle "Indian Music Raga"](https://www.kaggle.com/datasets/kcwaghmarewaghmare/indian-music-raga) | 8 ragas, short clips | Kaggle | Not used: clips likely come from few recordings, so it invites leakage. |

Practical facts learned while building the loaders **(verified)**:
- **mirdata's `compmusic_raga` index lists only the 477 Carnatic tracks, no Hindustani ones.** We read the files directly (`src/raaga/data/hmd.py`).
- Zenodo **rate-limits anonymous clients** (HTTP 429 after a handful of parallel range requests, roughly 60 requests/min). So each zip is downloaded once as a single resumable stream and only the wanted files are kept.
- HMD's Hindustani part after filtering is ~2.3 GB of pitch text. Colab downloads are one-off; the small chunk-histogram caches go to Drive.
- Khamaj comes from only 4 concerts (4+4+1+1 recordings), so under concert-level grouping it cannot be in every fold. We put it in the test fold and flag such "thin" ragas.
- **Test/train artist overlap is high (~80-90%)**: 55 artists over 300 recordings makes an artist-disjoint split impossible while keeping all 30 ragas. Results therefore measure "new recording/concert, known artists", and will be optimistic for unseen singers. We track the overlap number and check unseen-artist generalisation on Saraga later.

### Models / approaches

- **No ready-made pretrained Hindustani raga classifier turned up.** We train our own.
- Published approaches that work: tonic-normalised chroma/CQT or pitch features into a CNN(+LSTM). The [PIM-v1 paper](https://arxiv.org/html/2406.02443v1) reports F1 ~ 0.89 on 12 ragas at 30 s chunk level. Pitch-histogram methods are the classic baseline.
- General music foundation models (MERT, CultureMERT, MusicFM, CLAP): [reported](https://arxiv.org/pdf/2411.18611) to be **weaker than task-specific pitch/CNN-LSTM features** for raga ID. Worth testing as a baseline/ensemble member, not as the main bet.
- Raga ID is intrinsically hard: ragas differ by note *movement*, ornaments (meend, gamak) and phrase, not only the note set. Related ragas will confuse any model.

---

## 2. Key design decisions

1. **Pitch-first.** HMD's audio needs an access request, but its pitch tracks are open. So the first models work on tonic-normalised pitch. This is also the classic, strong representation for raga. An audio front-end is added when audio access arrives (or via Saraga's audio).
2. **Split by recording, grouped by concert, never by clip.** Random clip splits leak and make every model look great. `data/splits/v1.csv` is committed, and every experiment uses it. Folds 0-3 = cross-validation, **fold 4 = held-out test, evaluated once at the end**. The code validates: no concert or MBID spans folds, and fold 4 contains every raga.
3. **Tonic normalisation.** Performers pick their own Sa; transposing pitch to Sa lets the model learn the raga instead of the key. Caveat: baselines use the tonic shipped with the dataset (an upper bound). The app must **estimate the tonic** from audio, so tonic-estimation error is an explicit robustness experiment (Phase 4). The HMD auto tonic already looks ~20 cents off on the one file inspected, so we compare `tonic` vs `tonicFine`.
4. **Chunk-level training, clip-level prediction.** Train on 30 s chunks (15 s hop). At inference, average chunk log-probabilities. Same pipeline the app will use.
5. **Calibrated probabilities.** Softmax is over-confident: temperature scaling on validation folds, report ECE, add an "uncertain" path for ragas outside the label set.
6. **Domain gap at inference.** HMD pitch came from a predominant-melody extractor (Melodia-style). The app needs a compatible extractor for arbitrary audio. To be tested: Melodia/pYIN/CREPE-style extractors on Saraga audio vs Saraga's own pitch tracks.

---

## 3. Compute plan (free)

| Platform | Free tier (approx., check current limits) | Use for |
|---|---|---|
| **Kaggle Notebooks** | ~30 GPU-hours/week (T4 x2 / P100), 12 h sessions | Main deep-model training. |
| **Google Colab (free)** | T4, ~12 h sessions, can disconnect, Drive mount | Data prep (CPU is enough), quick iteration. |
| Lightning AI / others | Small free GPU allotments | Backup |

Working rules:
- Code lives in this repo; notebooks are thin runners that `git clone` + call `src/raaga`. (**The repo must be pushed to GitHub for Colab to clone it.**)
- **Cache features once** (small `.npz` on Drive) and train from the cache. Never re-decode raw data per epoch.
- Persistent files go to Drive (`raaga-work/`); big downloads go to fast local disk and are re-fetched in a new session.
- Checkpoints and logs written every epoch. Sessions die.

---

## 4. Phases and status

### Phase 0: Setup: **code done**
- Package (`src/raaga`), tests (26 passing), Colab/Kaggle notebooks, loaders for HMD and Saraga, split code.
- **Still needs:** push repo to GitHub; **user requests HMD audio access** on Zenodo (only needed for audio-based models).

### Phase 1: Data audit: **done**
- `notebooks/01_data_audit.ipynb`: downloads HMD + Saraga, builds the catalog, checks vocal/instrumental, per-raga table (recordings, artists, concerts, hours), near-duplicate raga names, writes `splits/v1.csv`.
- **Catalog** (`data/catalog.csv`, not committed -- has machine-local paths): 370 recordings (300 HMD + 70 Saraga after dedup). 0 missing tonic/pitch (was 64 before the fix). Vocal/instrumental: 341 vocal, 27 unknown, 2 instrumental (Saraga's own metadata; HMD's MusicBrainz check found no clear instrumentalists). 67 distinct raga names before the min-recordings filter.
- **Split** (`data/splits/v1.csv`, committed): after the min-8-recordings and vocal filters (`vocal`/`unknown` kept) -- **306 recordings, 30 ragas** (all of HMD's ragas, 9-11 recordings each; 298 HMD + 8 Saraga), 55 artists, 147 concerts. 5 folds, balanced (61-62 each), no leakage (`validate_folds` clean). Only Khamaj is "thin" (4 concerts, so it can't be in every CV fold; correctly placed in the test fold regardless). Test-fold/train artist overlap: 0.85 -- as flagged before, results measure "new recording, mostly-known artists", not fully unseen singers.
- **Exit:** `data/splits/v1.csv` committed. **Raga list is fixed: the 30 ragas in `data/splits/v1.csv`.**

### Phase 2: Baselines: **B1 code + notebook ready**
- **B1** tonic-normalised pitch-class histogram (120 bins) -> logistic regression. Notebook `02_baseline_pitch_histogram.ipynb`, 4-fold CV then one final test run.
- **B2** frozen MERT embeddings -> linear probe (needs audio, so Saraga only at first; deferred until audio access).
- **Exit:** recording-level top-1/top-3/macro-F1/ECE and confusion matrix for B1 (and B2) on the fixed split.

### Phase 3: Main model
- **M1** sequence model on tonic-normalised pitch (CNN+BiLSTM/GRU or small Transformer over 30 s of pitch relative to Sa), needs a GPU. This is where the deep-learning gain over B1's histogram should come from (note order, glides, phrases).
- **M2** tonic-normalised chroma/CQT CNN+BiLSTM from audio (PIM-v1 style) once audio is available.
- **M3 (optional)** fine-tune MERT with a small head.
- Augmentation must keep the raga intact: no naive transposition without re-normalising to Sa.
- **Exit:** beats B1 on macro-F1 with an ablation table (tonic normalisation, chunk length, augmentation).

### Phase 4: Calibration and robustness
- Temperature scaling, ECE, reliability plot, clip-level aggregation.
- Tonic **estimated** instead of annotated. Clip lengths 10/30/60 s. Noisy/phone recordings. Unseen artists (Saraga vs HMD). Open-set behaviour with ragas outside the label set.
- **Exit:** calibrated top-3 output with an "uncertain" fallback.

### Phase 5: Package the model
- Single `predict(audio_path) -> {raga: prob}`, ONNX/TorchScript export, model card (data, licences, known confusions).
- **Exit:** CPU inference in a few seconds per 30 s clip.

### Phase 6: Web app (later)
- FastAPI (or Gradio first) around `predict()`, hosted on Hugging Face Spaces (free CPU). Upload or record audio, show top-k bar chart.

---

## 5. Repo layout

```
raaga-detector/
  PLAN.md  README.md  pyproject.toml  requirements.txt
  configs/            # raga_aliases.yaml, artist_aliases.yaml
  data/splits/        # committed split CSVs (ids + labels only, no audio)
  src/raaga/
    env.py            # Drive/Kaggle/local paths
    data/             # remote (resumable zip download), hmd, saraga, catalog, splits, names, musicbrainz
    features/         # pitch (tonic-normalised histograms), cache (per-recording .npz)
    models/           # baseline (histogram + logistic regression)
    eval/             # metrics (top-k, macro-F1, confusion, ECE)
  tests/
  notebooks/          # thin Colab/Kaggle runners only
```

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| HMD audio access slow/denied | Pitch-first plan does not need it. Saraga has audio for front-end tests. |
| Only 10 recordings per raga (HMD) | Small models, heavy chunk-level augmentation of the *pitch* (small time-warps, micro-detuning), combine with Saraga, report per-raga confidence honestly. |
| Optimistic numbers from artist overlap (~80-90%) | Report the overlap; test on Saraga artists not in HMD; do not over-claim. |
| Annotated tonic vs estimated tonic at inference | Phase 4 experiment, and a tonic estimator in the app pipeline. |
| Pitch-extractor mismatch at inference | Evaluate extractors on Saraga (audio vs its pitch file) before choosing the app front-end. |
| Non-commercial licences (Saraga) | Fine for research; re-check before deploying weights publicly. |
| Free GPU quota / disconnects | Cached features, frequent checkpoints, small models. |

---

## 7. Immediate next steps

1. ~~Push this repo to GitHub~~ -- done, merged into `main` (PR #1, #2).
2. ~~Run `notebooks/01_data_audit.ipynb`~~ -- done. `data/splits/v1.csv` committed: 306 recordings, 30 ragas.
3. ~~Request HMD audio access on Zenodo~~ -- done, waiting on approval.
4. Run `notebooks/02_baseline_pitch_histogram.ipynb`: first accuracy number (with `TONIC = "tonic"` and `"tonicFine"`).
5. Build M1 (pitch-sequence deep model) once the B1 number is in.
