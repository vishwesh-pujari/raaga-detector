# Raaga Detector: Project Plan

Goal: a web app that takes an audio clip of **Hindustani** classical music and returns a probability distribution over ragas.
Current focus: **the deep-learning model only** (Phases 0-5). The web app (Phase 6) comes after.
All training runs on free cloud GPUs/CPUs (Colab / Kaggle), never locally.

_Last updated: 2026-09-30. Sections marked **(verified)** were checked against the actual archives; the rest comes from papers/search summaries and should be treated as "to verify". See [CONCEPTS.md](CONCEPTS.md) for what the terms used below mean and why things are done this way._

---

## 0. Decisions so far

| Date | Decision | Consequence |
|---|---|---|
| 2026-09-28 | **Vocal only** at first | Drop instrumental recordings. Saraga labels the lead instrument, so it is filtered exactly. HMD has no instrument field: its 55 artists look like vocalists (possible exception: Gopal Mishra, 4 recordings), confirmed via MusicBrainz in notebook 01. |
| 2026-09-28 | **No fixed raga list**: use whatever the data supports | Label set = ragas with enough recordings after merging datasets (HMD alone gives 30). |
| 2026-09-28 | **Non-commercial use only** | Saraga (CC BY-NC-SA 4.0) can be used. Re-check licences before any public deployment of trained weights. |
| 2026-09-29 | **Bug found and fixed**: HMD loader was silently dropping recordings | `path_mbid_ragaid.json`'s `path` field keeps punctuation (`:`, `&`) that the archive's real folder names sanitise to `_`. Reconstructing the feature-file path from that field missed any recording whose concert name has one of those characters -- 64 of 300 (21%) in the first audit run. Fixed by matching files via the mbid every filename ends in, instead of rebuilding the path from text (`src/raaga/data/hmd.py`). Verified twice: against the real archive listing (300/300 matched) and, after a genuine re-run (needed a Colab runtime restart -- see below), against the actual regenerated catalog (0 missing tonic/pitch, up from 64). |
| 2026-09-29 | Colab silently reused a stale kernel on the first "re-run" | Restarting cells / opening a "new" notebook does not guarantee a fresh Python process. An already-`import`-ed module keeps running from memory even after `git pull` updates the file on disk and `pip install -e .` reruns. The giveaway: the output was byte-identical to the pre-fix run, including a `pitch_path` string only the *old* code could construct. Fix: **Runtime -> Restart runtime** (or "Disconnect and delete runtime") before re-running, not just re-running cells. |
| 2026-09-30 | **HMD audio access request rejected** | Zenodo's restricted-access form for the HMD audio (record [7278511](https://zenodo.org/record/7278511)) requires a genuine research purpose and academic institution affiliation -- the user doesn't have one to supply, so the request was denied. **HMD audio is now assumed permanently unavailable for this project.** Consequence: any audio-based model (M2, B2/MERT beyond Saraga) can only use Saraga's audio (108 tracks, 61 ragas, CC BY-NC-SA) -- a much smaller set than HMD's 300. The pitch-first plan (B1, M1) is unaffected, since it only ever needed HMD's open pitch/tonic files, never its audio. |

---

## 1. What exists already

### Datasets

| Dataset | Content | Access | Notes |
|---|---|---|---|
| **HMD**: CompMusic Hindustani Music Dataset (Zenodo [7278506](https://zenodo.org/records/7278506), [details](https://compmusic.upf.edu/node/328)) **(verified)** | 300 recordings, **30 ragas x 10 recordings**, ~116 h, 55 artists, 146 concerts. Per recording: raw predominant pitch (4.4 ms hop), tonic (auto and manually fine-tuned), raga label, MBID. | **Pitch/tonic/labels open** (CC BY 4.0). **Audio restricted, and permanently unavailable to this project**: the access request (Zenodo [7278511](https://zenodo.org/record/7278511)) was **rejected 2026-09-30** -- it requires a genuine research purpose / academic institution affiliation, which isn't available here. | Primary dataset. Balanced (10 per raga). The pitch files make a pitch-based model possible without the audio -- which is now the only option for HMD, not just the initial plan. |
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

1. **Pitch-first.** HMD's audio access was requested and **rejected** (see section 0 -- Zenodo requires an academic affiliation this project doesn't have), but its pitch tracks are open regardless. So the first models work on tonic-normalised pitch. This is also the classic, strong representation for raga. An audio front-end, if built, uses Saraga's audio only (HMD's is now assumed permanently out of reach).
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

### Phase 0: Setup: **done**
- Package (`src/raaga`), tests (27 passing), Colab/Kaggle notebooks, loaders for HMD and Saraga, split code. Repo pushed to GitHub.
- HMD audio access requested and **rejected** (see section 0) -- resolved as "not available," not a blocker.

### Phase 1: Data audit: **done**
- `notebooks/01_data_audit.ipynb`: downloads HMD + Saraga, builds the catalog, checks vocal/instrumental, per-raga table (recordings, artists, concerts, hours), near-duplicate raga names, writes `splits/v1.csv`.
- **Catalog** (`data/catalog.csv`, not committed -- has machine-local paths): 370 recordings (300 HMD + 70 Saraga after dedup). 0 missing tonic/pitch (was 64 before the fix). Vocal/instrumental: 341 vocal, 27 unknown, 2 instrumental (Saraga's own metadata; HMD's MusicBrainz check found no clear instrumentalists). 67 distinct raga names before the min-recordings filter.
- **Split** (`data/splits/v1.csv`, committed): after the min-8-recordings and vocal filters (`vocal`/`unknown` kept) -- **306 recordings, 30 ragas** (all of HMD's ragas, 9-11 recordings each; 298 HMD + 8 Saraga), 55 artists, 147 concerts. 5 folds, balanced (61-62 each), no leakage (`validate_folds` clean). Only Khamaj is "thin" (4 concerts, so it can't be in every CV fold; correctly placed in the test fold regardless). Test-fold/train artist overlap: 0.85 -- as flagged before, results measure "new recording, mostly-known artists", not fully unseen singers.
- **Exit:** `data/splits/v1.csv` committed. **Raga list is fixed: the 30 ragas in `data/splits/v1.csv`.**

### Phase 2: Baselines: **B1 done (frozen)**
- **B1** tonic-normalised pitch-class histogram (120 bins) -> logistic regression. Notebook `02_baseline_pitch_histogram.ipynb`, 4-fold CV then one final test run.
- **B2** frozen MERT embeddings -> linear probe (needs audio; Saraga is now the only available source, HMD audio access was rejected -- see section 0).
- **Exit:** recording-level top-1/top-3/macro-F1/ECE and confusion matrix for B1 (and B2) on the fixed split.

**B1 cross-validation results (2026-09-30)**, 30 ragas, chance level = 0.033 (1/30). Best `C = 0.1` for both tonic variants by CV top-1.

| Metric (best C) | `tonic` (automatic) | `tonicFine` (manually corrected) |
|---|---|---|
| chunk-level top-1 | 0.690 | 0.711 |
| **recording-level top-1** | **0.935** | **0.935** |
| recording-level top-3 | 0.988 | 0.992 |
| recording-level macro-F1 | 0.923 | 0.921 |
| ECE (calibration error) | 0.139 | 0.121 |
| top-1 std across CV folds | 0.023 | 0.020 |

**Takeaways:**
1. **A simple linear model on tonic-normalised pitch histograms already gets ~93.5% recording-level top-1 accuracy** across all 30 ragas (chance = 3.3%), with 98.8-99.2% top-3. That's a strong floor for M1 (the deep model) to beat, and roughly in line with published tonic-normalised approaches for Hindustani raga ID (e.g. PIM-v1's F1 ~0.89 on 12 ragas).
2. **`tonic` (the dataset's automatic estimate) and `tonicFine` (manually corrected) perform statistically identically at the recording level** (0.935 vs 0.935 top-1) -- `tonicFine` only edges ahead on chunk-level accuracy, calibration and fold-to-fold stability. **This means B1 is fairly robust to small tonic-estimation error**, which is good news: the deployed app will only ever have an *automatically estimated* tonic (no manual correction possible), and this suggests that doesn't cost much accuracy, at least for this representation. Worth re-testing this robustness more aggressively in Phase 4 (larger, deliberately-injected tonic errors, not just auto-vs-manual).
3. **Decision: `tonic` (not `tonicFine`) is the config used for the one-time held-out test**, specifically because it is closer to what inference will actually have available (an automatic estimate, not a manual correction) -- and since the two are tied, there's no accuracy cost to picking the more representative one. `tonicFine` was a cross-validation-only comparison, not separately taken to the held-out test fold (touching the test fold more than once, even to compare configs, would be a mild form of test-set leakage).
4. **Confusions are sparse and mostly musically plausible**, not random noise: worst is `khamaj -> alahaiyabilaval` (3x with `tonic`) and `des -> gaudmalhar` (4x with `tonicFine`; only 1x with `tonic` -- with ~10 recordings/raga, a handful of flipped predictions from a small input change is within normal noise, not a real regression).
5. **Caveats that still apply** (from section 0/1): this uses the dataset's *oracle* tonic (auto or manual), not one estimated from raw audio at inference time -- expect a real drop once the app has to estimate Sa itself. And ~85% train/test artist overlap means part of this accuracy could reflect recognising artists, not purely raga content.

**B1 held-out test result (2026-09-30, frozen -- fold 4, `tonic`, `C = 0.1`, touched once and not revisited):**

| Metric | CV mean (folds 0-3) | Held-out test (fold 4, n=61) |
|---|---|---|
| recording-level top-1 | 0.935 | **0.951** (58/61) |
| recording-level top-3 | 0.988 | 0.967 (59/61) |
| recording-level macro-F1 | 0.923 | 0.936 |
| ECE | 0.139 | 0.112 |

Before trusting this, checked that every raga (including thin Khamaj: 6 recordings spread across folds 0-3, 4 in the test fold) is actually present in the training set -- confirmed, all 30 ragas are. Test top-1/macro-F1/ECE all landed at or slightly better than the CV mean (within 1 CV fold-to-fold std of 0.023), so the CV estimate was honest, not optimistic -- no sign of overfitting to the CV folds. Top-3 looks a little low relative to top-1 only because of the small sample (61 recordings): of the 3 misses, 2 were wrong even within top-3, which at this sample size is not a meaningful signal on its own.

**B1 is now frozen at recording-level top-1 = 95.1% (macro-F1 = 0.936) on the held-out set.** This is the number M1 (the deep model, Phase 3) needs to beat to justify its extra complexity.

### Phase 3: Main model
- **M1** sequence model on tonic-normalised pitch (CNN+BiLSTM/GRU or small Transformer over 30 s of pitch relative to Sa), needs a GPU. This is where the deep-learning gain over B1's histogram should come from (note order, glides, phrases).
- **M2** tonic-normalised chroma/CQT CNN+BiLSTM from audio (PIM-v1 style) -- **Saraga audio only** now; HMD audio access was rejected (section 0). Within our locked 30-raga split, Saraga contributes only 8 of the 306 recordings (the rest are HMD, audio-less), so M2 as originally scoped isn't viable on this split without either narrowing to Saraga's own raga set or scraping more audio some other way. Lower-priority stretch goal, not blocking.
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
| **HMD audio access denied (confirmed 2026-09-30)** | Pitch-first plan (B1, M1) never needed it. Any audio-based model (M2, B2) is now Saraga-only -- thin within our 30-raga split (8/306 recordings), so scoped as a lower-priority stretch goal, not the main path. |
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
3. ~~Request HMD audio access on Zenodo~~ -- done, **rejected** (needs an academic affiliation this project doesn't have). Treated as permanent; not being re-requested unless something about that changes.
4. ~~Run `notebooks/02_baseline_pitch_histogram.ipynb`~~ -- done. B1 frozen: 95.1% recording-level top-1, 0.936 macro-F1 on the held-out fold.
5. Build M1 (pitch-sequence deep model): needs to beat B1's 95.1% top-1 / 0.936 macro-F1 to justify the extra complexity.
