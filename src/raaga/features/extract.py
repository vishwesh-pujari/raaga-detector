"""Pitch and tonic extraction from raw audio, via ``essentia`` -- the same library/ecosystem
CompMusic/Dunya (the source of HMD and Saraga) is itself built on, not something invented for this
project. Not needed for training (HMD/Saraga ship pre-extracted pitch+tonic); this is what the
deployed app needs to turn a user's raw audio into the same kind of input B1/M1 were trained on.

See PLAN.md Phase 4a for why this exists and how it was verified, and CONCEPTS.md for the
musicology (Sa/tonic) behind why tonic estimation specifically needs a sustained, drone-like
signal to work -- a bare monophonic melody line is not enough, but every real Hindustani
recording has a tanpura drone, so this is not a practical limitation.

``essentia`` is an optional dependency (``pip install essentia``): imported lazily inside each
function so the rest of ``raaga`` stays importable without it installed.
"""

from typing import Tuple

import numpy as np


def load_audio(path: str, sample_rate: int = 44100) -> np.ndarray:
    """Mono float32 audio at ``sample_rate`` Hz (resampled if needed), via essentia's MonoLoader
    (handles mp3/wav/flac/... via ffmpeg)."""
    import essentia.standard as es

    return es.MonoLoader(filename=str(path), sampleRate=sample_rate)()


def extract_pitch(
    audio: np.ndarray, sample_rate: int = 44100, hop_size: int = 128, frame_size: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """Predominant-melody pitch track (Melodia), returned as ``(times_s, freqs_hz)`` -- the same
    convention ``pitch.load_pitch_file()`` returns for HMD/Saraga's own shipped files, so this is
    a drop-in replacement anywhere those are used (``chunk_histograms``, ``chunk_pitch_sequences``).
    """
    import essentia.standard as es

    algo = es.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size, sampleRate=sample_rate)
    freqs, _confidence = algo(np.asarray(audio, dtype=np.float32))
    times = np.arange(len(freqs)) * (hop_size / sample_rate)
    return times, np.asarray(freqs, dtype=np.float64)


def estimate_tonic(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """Tonic (Sa) in Hz, via essentia's TonicIndianArtMusic.

    Needs a reasonably realistic signal to work -- specifically a sustained, drone-like tonal
    centre (what a tanpura provides in every real Hindustani recording). Verified on synthetic
    audio before this module was written: without a drone it can lock onto a different note
    entirely (off by a perfect fifth in one test); with one, it found the tonic to within ~3
    cents. See PLAN.md Phase 4a.

    Raises ``RuntimeError`` ("No peak locations") if it can't find any salient tonal centre at
    all -- seen on some synthetic test signals; callers batch-processing many real files (e.g. the
    Saraga validation) should catch this per-file rather than let one bad recording abort the run.
    """
    import essentia.standard as es

    return float(es.TonicIndianArtMusic()(np.asarray(audio, dtype=np.float32)))


def extract_pitch_and_tonic(
    path: str, sample_rate: int = 44100, **pitch_kwargs
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convenience: load the audio once, run both extractors. Returns ``(times_s, freqs_hz, tonic_hz)``."""
    audio = load_audio(path, sample_rate)
    times, freqs = extract_pitch(audio, sample_rate, **pitch_kwargs)
    tonic = estimate_tonic(audio, sample_rate)
    return times, freqs, tonic
