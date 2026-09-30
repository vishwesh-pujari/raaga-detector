# Concepts: what everything means and why we did it this way

This is a reference document, not a status update -- see [PLAN.md](PLAN.md) for what's actually
been done and what the results are. This file explains *why*: the musicology, the signal
processing, and the ML methodology behind the choices in this repo, written so that anyone
(including future us) can understand the reasoning without re-deriving it or re-reading the whole
chat history.

Jump to the [quick-reference glossary](#quick-reference-glossary) if you just need one term.

---

## Part A: What a raga actually is (and why that makes this hard)

A **raga** is not a scale. A Western major scale tells you which seven notes are allowed; a raga
tells you that *and*:

- **Aroha / avaroha** -- the notes used going up may differ from the notes used coming down
  (e.g. a note can be skipped ascending but included descending).
- **Vadi / samvadi** -- a "most important" note and a secondary important note, which get
  emphasised and lingered on far more than the others. Two ragas can share the exact same seven
  notes and still sound completely different because of which notes are emphasised.
- **Pakad** -- one or more characteristic short phrases that identify the raga almost by
  themselves, the way a few notes of a famous melody identify a song.
- **Ornaments** -- *meend* (a smooth glide between two notes, not a discrete jump), *gamak*
  (an oscillating shake around a note), *kan* (a brief grace note). These are part of *how* a note
  is sung, not just *which* note.
- Traditionally, a *time of day* or season it's associated with.

**Why this matters for the project:** a raga is defined by melodic *movement and emphasis*, not
just by which notes occur. A model that only knows "which notes were used, how much" (our B1
baseline, see Part D) is throwing away the aroha/avaroha direction, the ornaments, and the phrase
order -- and yet it still gets ~95% accuracy (see [PLAN.md](PLAN.md)), which tells us the note-usage
distribution alone is already a very strong signal for telling ragas apart, at least among the 30
in our dataset. It also tells us *what's left on the table* for a model that does use melodic
order (the planned deep model, M1) to pick up.

### Sa (the tonic) -- the single most important idea in this whole pipeline

Indian classical music is **relative-pitch music**, not fixed-pitch music. A singer or
instrumentalist picks their own **Sa** (tonic, the "do" of their personal scale) based on their
vocal range or instrument, and every other note in the raga is defined *relative to that Sa* --
as a ratio/interval, not as an absolute frequency. One performer's Sa might be 140 Hz and
another's 240 Hz, and they could be singing the exact same raga.

**Consequence:** if you fed raw pitch-in-Hz into a model, it would have to separately learn every
raga at every possible Sa a performer might choose -- effectively learning the same pattern many
times over, wastefully, and generalising poorly to a Sa it hadn't seen. This is why the very first
thing the pipeline does to any pitch data is **tonic normalisation** (Part B) -- converting
"140 Hz" into "this many cents above *this performer's* Sa" so the model sees the same numbers
for the same raga regardless of who's singing or on what instrument.

---

## Part B: Signal processing -- turning audio into numbers a model can use

### Predominant pitch (F0) extraction

The raw input isn't audio waveforms; it's a **pitch track**: for every ~4.4 ms frame, one
frequency number (or "unvoiced" if there's no clear pitch, e.g. silence, a pause, a drum hit).
This is produced by a *predominant melody extraction* algorithm (Melodia-family, in this
dataset's case) that looks at a recording -- which may have a singer, tabla, tanpura drone, and
harmonium all playing at once -- and tries to isolate just the lead melodic line's pitch over time.
It's not perfect (voice can sound over accompaniment, or vice versa), but it's the standard
approach in this field and it's what the HMD/Saraga datasets ship as their `.pitch` files.

### Tonic (Sa) value

A single number (in Hz) per recording: the performer's Sa, either estimated automatically (the
`.tonic` file) or manually corrected by a human annotator (the `.tonicFine` file). See
[PLAN.md](PLAN.md) for the experiment comparing the two -- short version, they perform almost
identically for our baseline, which is reassuring since the eventual app will only ever have an
automatic estimate.

### Cents -- the unit that makes "relative pitch" numerically well-behaved

A **cent** is 1/1200th of an octave (an octave = 1200 cents, a semitone = 100 cents). Musical
intervals are *ratios* of frequency, not differences -- doubling frequency is always "one octave
up" whether you start at 100 Hz or 400 Hz. Cents convert that multiplicative relationship into an
additive one via a logarithm:

```
cents_above_Sa = 1200 * log2(note_frequency_hz / tonic_hz)
```

This is what [`cents_above_tonic()`](src/raaga/features/pitch.py) computes. Working in cents
(instead of raw Hz, or a raw Hz ratio) means "the same musical interval" always produces "the same
number," which is exactly what we need for the model to recognise the same raga across performers.

### Octave folding -- pitch class

A raga's rules (which notes, aroha/avaroha, vadi/samvadi) are usually the same whether a note is
sung in a low octave or a high one -- "Pa" is "Pa" whether it's the singer's low register or high
register. So after converting to cents-above-Sa, we **fold** everything into a single octave
(0-1200 cents, wrapping around) with [`_bin_index()`](src/raaga/features/pitch.py). This is the
same idea as "pitch class" in Western music theory (C4 and C5 are both just "C"). It also means a
lot of octave-jump noise from the pitch extractor gets absorbed rather than treated as a different
note.

### Pitch-class histogram -- B1's actual input features

Once every voiced frame has a cents-above-Sa value folded into one octave, we bin those values
into **120 bins** (so each bin is 10 cents wide -- finer than a semitone, which is 100 cents) and
count how much time was spent in each bin, normalised so the counts sum to 1. That's the
[`pitch_class_histogram()`](src/raaga/features/pitch.py): a 120-number vector that says
"this is roughly what fraction of the performance was spent on each note, ignoring octave and
ignoring the order things happened in."

This directly encodes vadi/samvadi (the emphasised notes show up as tall peaks) but **throws away
aroha/avaroha, ornaments and phrase order** entirely -- it's a "bag of notes," the audio
equivalent of a bag-of-words text model. That's a deliberate simplification for the baseline (B1):
cheap to compute, cheap to train a model on, and a clean way to measure how much of raga
identification comes from "just" note emphasis before building something that also models melodic
movement (M1).

### Pitch *sequences* -- M1's input, and how it differs from B1's histogram

A histogram answers "which notes, how much" but discards *when*. M1 needs "when," so instead of
one aggregated vector per chunk, [`chunk_pitch_sequences()`](src/raaga/features/pitch.py) produces
an **ordered sequence** per chunk: 300 steps of 100 ms each (for the default 30 s chunk), each step
a 3-number summary of that instant's pitch. Two design choices worth explaining, since they're not
the obvious first thing to reach for:

- **Why `sin`/`cos` of the angle, not the cents value itself.** Cents-above-tonic wraps at the
  octave (1199 cents and 1 cent are adjacent notes, not far apart) -- feeding that raw number to a
  model means it has to somehow learn that wraparound itself, and a small pitch wobble across the
  wrap point looks like a huge jump. Converting the octave-folded cents to an angle
  (`theta = 2*pi*cents/1200`) and taking `(sin(theta), cos(theta))` sidesteps this entirely: it's
  the standard "circular data" trick (the same one used for encoding e.g. time-of-day or compass
  direction), and it's naturally, automatically periodic -- no wraparound discontinuity exists in
  this representation at all.
- **Why the *circular mean*, not a single frame's value.** Native pitch frames are ~4.4 ms apart,
  much finer than the 100 ms step size, so each step actually covers ~23 native frames. Averaging
  their `sin`/`cos` values (rather than picking one frame, or averaging the raw cents) gives a
  useful bonus for free: if those 23 frames mostly agree on one note, the averaged `(sin, cos)`
  stays close to the unit circle; if the pitch is unstable or mid-transition within that step, the
  averaged vector shrinks toward `(0, 0)`. So the *magnitude* of the pair doubles as an implicit
  "how confident/stable was the pitch here" signal, without a separately engineered confidence
  feature. A fully unvoiced step is exactly `(0, 0)` -- sitting at the same "low confidence" region
  the noisy case shrinks toward, rather than being an arbitrary special value the model has to
  learn to treat differently.
- The third number per step, `voiced_frac`, is just the fraction of that step's native frames that
  had a pitch at all -- silence/consonants/breaths vs. sustained singing.

Verified (`tests/test_pitch.py`): this representation is octave-invariant and tonic-invariant, the
same core properties as B1's histogram -- and, the one property that actually matters for M1's
reason to exist, two chunks using the *same notes in a different order* produce genuinely
*different* sequences (time-reversing one exactly reproduces the other), which a histogram cannot
distinguish at all.

### Chunking (30 s windows, 15 s hop)

Recordings are long -- tens of minutes, sometimes almost two hours for a single performance -- but
a raga's defining characteristics repeat throughout, they aren't a one-time event at the start.
Rather than compute one histogram for an entire recording, [`chunk_histograms()`](src/raaga/features/pitch.py)
slides a 30-second window (moving 15 seconds each step, so windows overlap by half) across the
recording and computes a separate histogram per window. Three reasons:

1. **More training examples per recording** -- a 20-minute recording yields ~80 overlapping
   chunks instead of 1 data point, which matters a lot with only ~10 recordings per raga.
2. **Matches what the real app will do.** A user will upload a short clip (seconds to a couple of
   minutes), not a full concert -- so training and evaluating on short windows is training on
   the actual shape of the real problem, not a mismatched easier one.
3. Chunks with very little voiced signal (silence, a long instrumental tuning passage, tanpura
   only) are dropped (`min_voiced_s`), so the histogram isn't built from mostly-nothing.

---

## Part C: Machine learning methodology

### Why the split is grouped by concert, not random

The single most common mistake in this kind of project: if chunks (or even whole recordings) from
the *same performance* end up in both the training set and the test set, the model can partly
"memorise" that specific recording's quirks (background noise, exact tempo, a particular
harmonium's tuning idiosyncrasy) rather than learning the raga's actual structure, and the
resulting accuracy number is inflated and won't hold up on genuinely new audio. So every split in
this project (`data/splits/v1.csv`) groups by **concert/album**, not by individual recording or
chunk -- an entire performance goes into exactly one fold, never split across two. See
[`assign_folds()`](src/raaga/data/splits.py).

### K-fold cross-validation vs. the held-out test set

We use **5 folds**. Four of them (0-3) are used for **cross-validation**: repeatedly train on
three of the four, evaluate on the fourth held-out one, and rotate which one is held out. This
gives four separate accuracy estimates and lets us pick things like the regularisation strength
`C` *without ever looking at the 5th fold*. Fold **4 is the true held-out test set** -- touched
exactly once, after every other decision (model, hyperparameters, which tonic variant) has already
been made using only folds 0-3. If you peek at fold 4 more than once to help choose between
options, you're slowly turning it into another validation fold and its number stops meaning
"how will this do on data we've never seen."

### "Thin" ragas and why the split code has a special case for them

With as few as ~10 recordings for some ragas, spread across only a handful of distinct concerts,
it's sometimes mathematically impossible for a raga to appear in *every* one of 5 groups (e.g.
Khamaj comes from only 4 concerts). [`thin_ragas()`](src/raaga/data/splits.py) identifies these,
and the fold-assignment logic specifically prioritises keeping the *test* fold (fold 4) complete
--every raga must appear there -- even if that means one of the CV folds has to be the one
missing a thin raga instead. Missing a raga from one CV fold barely affects hyperparameter
selection; missing it from the test fold would mean we never actually measured that raga's
real-world performance at all.

### Chunk-level predictions -> one recording-level prediction

The model predicts a probability distribution **per 30-second chunk**, but what actually matters
(both for evaluation and for the eventual app) is one distribution **per recording/clip**. We
combine a recording's chunks by averaging their **log-probabilities** (not raw probabilities) and
re-normalising -- see [`aggregate()`](src/raaga/models/baseline.py). Averaging in log-space is
equivalent to taking the *geometric mean* of the chunk probabilities, which is more robust than a
plain average: one chunk being very confidently wrong (e.g. a noisy segment) doesn't dominate the
combined result the way it would under a plain arithmetic mean.

### Metrics, and why each one is reported

- **Top-1 accuracy**: does the single highest-probability raga match the true one? The headline
  number, but on its own can hide systematic failures on rare classes.
- **Top-3 accuracy**: is the true raga anywhere in the top 3 guesses? Relevant because the
  eventual app can show a shortlist rather than a single guess -- and because closely related
  ragas that are genuinely hard to tell apart (shared notes, similar mood) should still "count" as
  a near-miss, not a total failure.
- **Macro-F1** (not plain accuracy, and not micro-F1): computed *per raga* and then averaged
  across ragas equally, regardless of how many recordings each raga has. A model that's excellent
  on common ragas but useless on rare ones would still score well on plain accuracy but poorly on
  macro-F1 -- which is what we actually care about, since the point is to recognise *any* raga a
  user might sing, not just the most-represented ones in the dataset.
- **Chance level** = 1 / (number of ragas). With 30 ragas that's 3.3% -- the score a model that
  guesses randomly (weighted by class frequency, roughly) would get. Every accuracy number should
  be read relative to this, not in isolation. B1's 95% is meaningful specifically because it's so
  far above 3.3%, not because 95% is a "high-looking" number in the abstract.
- **Confusion matrix**: which specific ragas get mistaken for which. Diagonal = correct.
  Off-diagonal patterns matter more than the raw miss count -- confusing two ragas that share most
  of their notes is a very different (much more forgivable) kind of error than confusing two
  ragas with nothing in common.
- **ECE (Expected Calibration Error)**: measures whether the model's *stated confidence* matches
  its *actual accuracy* -- e.g., among predictions the model was "80% confident" about, were
  roughly 80% of them actually correct? Raw softmax outputs from most models are systematically
  **overconfident** (they say 99% when the real hit rate at that confidence level is more like
  85%). This matters specifically because the end product is a user-facing probability ("this
  clip is 80% likely to be Raga Yaman") -- if that number is miscalibrated, it's actively
  misleading, not just "a bit off." Planned fix (Phase 4): temperature scaling, calibrated on data
  the model wasn't trained on.

---

## Part D: This project's pipeline, end to end, and why

1. **Data**: two datasets merged, CompMusic's Hindustani Music Dataset (HMD -- balanced, 30 ragas
   x 10 recordings, pitch/tonic open, audio restricted) and Saraga Hindustani (fewer recordings
   but includes audio and explicit instrument metadata). Merged so ragas get more recordings
   than either alone provides, and de-duplicated by MusicBrainz ID where a recording appears in
   both. See [PLAN.md](PLAN.md) section 1 for the exact numbers and where they come from.
2. **Vocal-only filtering**: the user's stated preference. Saraga's metadata names the lead
   instrument directly; HMD has no such field, so a MusicBrainz lookup
   ([`musicbrainz.py`](src/raaga/data/musicbrainz.py)) is used to classify each recording, with
   recordings MusicBrainz has no opinion on kept as "unknown" rather than dropped (since
   over-filtering would lose data we can't otherwise verify one way or the other).
3. **Why B1 (a simple linear model) before any deep learning**: establishes (a) that the whole
   pipeline actually works end to end, on real numbers, before investing in something more complex
   and slower to debug, and (b) a concrete floor -- a number the deep model has to actually beat to
   be worth its cost, rather than us just assuming a bigger model will be better.
4. **The HMD path-matching bug** (see [PLAN.md](PLAN.md) for the full story): the dataset's own
   metadata JSON contained folder names with punctuation (`:`, `&`) that didn't match the real,
   sanitised folder names in the archive -- silently losing 21% of recordings' tonic/pitch data.
   The general lesson that shaped how the rest of the data code is written: **verify a dataset
   loader against the actual raw files, not just against the metadata that's supposed to describe
   them** -- metadata can lie (or just be stale) even in a well-known, citable dataset.
5. **Why we compared `tonic` vs `tonicFine`**: not to pick "the better one" for its own sake, but
   as a robustness check -- does small tonic-estimation error hurt accuracy much? They came out
   statistically tied, which is good news, because the deployed app will only ever have an
   *automatically estimated* tonic (no human to manually correct it), and this result suggests
   that doesn't cost much, at least for the histogram representation. `tonic` was chosen for the
   frozen official test specifically because it's the more realistic stand-in for what inference
   will actually have.
6. **The ~85% train/test artist overlap caveat**: with 55 artists across 300 HMD recordings, an
   artist-disjoint split isn't possible while keeping all 30 ragas represented. So B1's 95%
   partly reflects "new recording, familiar artist" rather than "completely unseen singer" --
   flagged, not hidden, and something to specifically test later against Saraga artists who don't
   appear in HMD at all.
7. **M1's architecture (CNN + BiGRU + attention), and why not a Transformer for v1**: a small 1D
   convolution frontend first, over the pitch-sequence's time axis -- this picks up *local*
   melodic movement (a handful of consecutive notes: an ornament, a short turn of phrase) the same
   way a CNN over an image picks up local edges/textures before anything else sees the whole
   picture. Its output feeds a **bidirectional GRU** (a recurrent network that reads the sequence
   both forward and backward, since the whole 30 s chunk is available at once -- no need to only
   look backward the way a live/streaming model would) to capture longer-range structure: how
   phrases relate to each other across the chunk, closer to pakad-level pattern than note-level.
   **Attention pooling** turns the GRU's per-time-step outputs into one fixed-size vector for the
   classifier, by learning *which moments in the chunk matter most* for identifying the raga
   (e.g. weighting a clear, characteristic phrase more than an ambiguous or noisy stretch) instead
   of treating every instant equally the way a plain average would. A Transformer was the other
   option Phase 3 named, but was set aside for this first version specifically because it's
   substantially more data-hungry than a CNN+GRU, and there are only ~245 training recordings per
   fold -- worth reconsidering later if this architecture plateaus.
8. **Class weighting during training**: ragas don't have perfectly equal recording counts (9-11
   each), so the loss function is weighted inversely to how often each class appears in the
   training data -- a rare raga's mistakes count for more than a common raga's, so the model can't
   get a free ride by just being good at whichever ragas happen to have slightly more examples.
   Mirrors B1's `class_weight="balanced"`, for the same reason macro-F1 (not accuracy) is the
   headline metric: treat every raga as equally important to get right, not weighted by how much
   data happened to be available for it.
9. **Early stopping**: training tracks validation loss every epoch and keeps the best-scoring
   weights, stopping if it hasn't improved for a set number of epochs (`patience`). Deep models can
   keep fitting the training set indefinitely long after they've stopped generalising better to
   unseen data -- this is what actually decides when to stop, rather than an arbitrarily fixed
   number of epochs.

---

## Quick-reference glossary

| Term | Meaning here |
|---|---|
| **Raga** | A melodic framework: allowed notes, aroha/avaroha (ascent/descent), vadi/samvadi (emphasised notes), characteristic phrases (pakad), ornamentation -- more than just a scale. |
| **Sa / tonic** | The performer's own chosen base pitch (in Hz); every other note is relative to it, not absolute. |
| **Swara** | A note, named relative to Sa (e.g. Re, Ga, Ma...), not an absolute pitch. |
| **Cents** | Log-frequency unit: 1200 cents = 1 octave, 100 cents = 1 semitone. Makes musical intervals additive instead of multiplicative. |
| **Tonic normalisation** | Converting Hz pitch to cents-above-Sa, so the same raga looks the same regardless of the performer's chosen key. |
| **Octave folding / pitch class** | Treating a note the same regardless of which octave it's sung in (wrapping cents into 0-1200). |
| **Pitch-class histogram** | A 120-bin distribution of how much time was spent on each (octave-folded) note -- B1's input feature. Ignores note order/melody. |
| **Chunk** | A 30-second (15 s hop) window of a recording; the unit the model actually trains and predicts on. |
| **Concert-grouped split** | Every recording from one performance goes entirely into one fold, never split across train/test, to prevent leakage. |
| **Cross-validation (CV)** | Rotating which of folds 0-3 is held out, to pick hyperparameters without touching the true test set. |
| **Held-out test set** | Fold 4. Touched exactly once, after every other decision is made. |
| **Thin raga** | A raga from too few distinct concerts to appear in every fold; guaranteed to be in the test fold regardless. |
| **Top-1 / top-3 accuracy** | Is the true raga the #1 guess / anywhere in the top 3? |
| **Macro-F1** | Per-raga F1 averaged equally across ragas, regardless of how many recordings each has -- doesn't let common ragas hide poor performance on rare ones. |
| **Chance level** | 1 / (number of ragas) -- the baseline "random guess" score everything should be compared against. |
| **ECE (calibration error)** | How far the model's stated confidence is from its actual accuracy at that confidence level. |
| **Confusion matrix** | Table of true raga vs. predicted raga, showing exactly which pairs get mixed up. |
| **B1** | The frozen baseline: tonic-normalised pitch-class histogram -> logistic regression. |
| **M1** | The deep model: a CNN+BiGRU+attention sequence model over tonic-normalised pitch (sin/cos encoded), meant to capture melodic order/movement that B1's histogram ignores. Beats-B1 is the exit criterion. |
| **Circular mean** | Averaging angles (here, `sin`/`cos` of octave-folded pitch) instead of raw values, so wraparound (e.g. 1199 cents and 1 cent being neighbours) is handled correctly; its magnitude also naturally reflects how consistent/confident the underlying values were. |
| **GRU / BiGRU** | A recurrent neural network that processes a sequence step by step, carrying forward a summary of what it's seen; "bidirectional" means it does this both forward and backward and combines the two, useful when the whole sequence is available at once (not streaming). |
| **Attention pooling** | Turns a sequence of per-step vectors into one fixed-size vector by learning *how much each step should count*, rather than averaging every step equally. |
| **Class weighting** | Weighting the training loss inversely to how often each class appears, so rare classes' mistakes matter as much as common classes' -- mirrors why macro-F1, not accuracy, is the headline metric. |
| **Early stopping** | Stopping training when validation loss hasn't improved for a set number of epochs, and keeping the best-so-far weights -- prevents fitting the training set past the point of actually generalising better. |
| **HMD** | CompMusic Hindustani Music Dataset -- 300 recordings, 30 ragas x 10, open pitch/tonic, restricted audio. |
| **MBID** | MusicBrainz ID -- used to deduplicate recordings that appear in more than one dataset, and to look up instrument metadata. |
