"""Tests for raw-audio pitch/tonic extraction (essentia). Uses synthetic signals rather than real
recordings -- validated by hand against essentia's actual behaviour first (see PLAN.md Phase 4a)
before writing any of this, since a naive synthetic signal turned out to be the wrong kind of test
for TonicIndianArtMusic specifically (needs a drone, not just a melody tone)."""

import numpy as np
import pytest
from scipy.io import wavfile

essentia = pytest.importorskip("essentia", reason="optional dependency, pip install essentia")

from raaga.features import extract as E  # noqa: E402
from raaga.features import pitch as P  # noqa: E402

SR = 44100


def _voice_like(tonic_hz, seconds=6, sr=SR, seed=0):
    """A single sustained note with vibrato + harmonics + a little noise -- what extract_pitch is
    meant to track. (Not what TonicIndianArtMusic needs -- see _drone_and_melody below.)"""
    t = np.arange(0, seconds, 1 / sr)
    vibrato = 1 + 0.005 * np.sin(2 * np.pi * 5 * t)
    phase = 2 * np.pi * np.cumsum(tonic_hz * vibrato) / sr
    signal = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    signal += np.random.default_rng(seed).normal(0, 0.01, len(signal))
    return signal.astype(np.float32)


def _drone_and_melody(tonic_hz, seconds=8, sr=SR, seed=0):
    """A sustained drone at the tonic (tanpura stand-in) plus a melody wandering elsewhere -- what
    TonicIndianArtMusic actually needs: it detects the drone, not 'the most common melody note'."""
    t = np.arange(0, seconds, 1 / sr)
    drone_phase = 2 * np.pi * tonic_hz * t
    drone = 0.3 * np.sin(drone_phase) + 0.15 * np.sin(2 * drone_phase) + 0.08 * np.sin(3 * drone_phase)
    mel_f = tonic_hz * 1.5 * (1 + 0.005 * np.sin(2 * np.pi * 5 * t))  # wanders around the fifth
    mel_phase = 2 * np.pi * np.cumsum(mel_f) / sr
    melody = 0.4 * np.sin(mel_phase) + 0.15 * np.sin(2 * mel_phase)
    signal = drone + melody
    signal += np.random.default_rng(seed).normal(0, 0.01, len(signal))
    return signal.astype(np.float32)


def test_extract_pitch_finds_the_right_note():
    tonic = 146.83
    audio = _voice_like(tonic)
    times, freqs = E.extract_pitch(audio)
    voiced = freqs[freqs > 0]
    assert len(voiced) / len(freqs) > 0.9, "expected mostly-voiced on a clean sustained tone"
    assert abs(np.median(voiced) - tonic) < 1.0


def test_extract_pitch_matches_load_pitch_file_convention():
    """Must be a drop-in replacement for the dataset's own pitch files: same (times, freqs) shape
    and dtype convention, directly usable by chunk_pitch_sequences/chunk_histograms."""
    times, freqs = E.extract_pitch(_voice_like(150.0))
    assert times.shape == freqs.shape
    assert times.dtype.kind == "f" and freqs.dtype.kind == "f"
    assert np.all(np.diff(times) > 0)  # strictly increasing


def test_estimate_tonic_needs_a_drone_not_just_a_melody_tone():
    """Documents the exact finding from PLAN.md Phase 4a as a regression check: TonicIndianArtMusic
    is not simply 'find the fundamental of the melody' -- it needs a sustained drone-like signal."""
    tonic = 146.83
    with_drone = E.estimate_tonic(_drone_and_melody(tonic))
    assert abs(with_drone - tonic) < 5.0  # within ~a few cents worth of Hz at this pitch

    bare_melody = 0.6 * np.sin(2 * np.pi * tonic * 1.5 * np.arange(0, 6, 1 / SR))  # just the "fifth" note
    without_drone = E.estimate_tonic(bare_melody.astype(np.float32))
    assert abs(without_drone - tonic) > 5.0  # unreliable without a drone -- expected, not a bug


def test_extract_pitch_integrates_with_the_existing_feature_pipeline(tmp_path):
    """The actual point of this module: extract_pitch's output must be directly usable by the same
    feature code B1/M1 were trained with, unmodified -- no adapter layer needed.

    Deliberately does *not* also require estimate_tonic to succeed on this same signal: a crude
    additive-sine synthetic drone+melody mix turns out to have no amplitude ratio where both
    PredominantPitchMelodia and TonicIndianArtMusic work well simultaneously (checked by hand --
    strengthen the drone enough for tonic detection and pitch voicing drops to 0%, or the tonic
    algorithm just errors out; this is a synthetic-signal artifact, not a real limitation, since
    each extractor already has its own dedicated, appropriately-designed test above). So this test
    uses a known tonic (the true one) to isolate what it's actually testing: format compatibility.
    """
    tonic = 130.0
    audio = _voice_like(tonic * 1.5)  # a clean, extractable melody tone
    wav_path = tmp_path / "a.wav"
    wavfile.write(wav_path, SR, audio)

    times, freqs = E.extract_pitch(E.load_audio(str(wav_path)))
    hist = P.pitch_class_histogram(freqs, tonic, n_bins=120)
    assert hist.shape == (120,) and abs(hist.sum() - 1) < 1e-6

    seqs, starts = P.chunk_pitch_sequences(times, freqs, tonic, chunk_s=3.0, hop_s=3.0, min_voiced_s=0.5)
    assert seqs.ndim == 3 and seqs.shape[1:] == (30, 3)  # 3s / 0.1s default step
    assert len(starts) > 0


def test_load_audio_reads_a_real_file_and_resamples(tmp_path):
    native_sr = 22050
    t = np.arange(0, 1.0, 1 / native_sr)
    tone = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    wav_path = tmp_path / "tone.wav"
    wavfile.write(wav_path, native_sr, tone)

    audio = E.load_audio(str(wav_path), sample_rate=44100)
    assert isinstance(audio, np.ndarray)
    # resampled to the target rate: ~1s of audio at 44100 Hz, not the native 22050
    assert abs(len(audio) - 44100) < 200
