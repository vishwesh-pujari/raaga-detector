import numpy as np

from raaga.features import pitch as P


def test_pitch_class_histogram_peaks_at_the_right_note():
    tonic = 146.83
    fifth = tonic * 2 ** (7 / 12)  # Pa: 700 cents -> bin 70 of 120
    h = P.pitch_class_histogram(np.full(1000, fifth), tonic, n_bins=120)
    assert h.argmax() == 70
    assert abs(h.sum() - 1) < 1e-9


def test_octave_invariance_and_unvoiced_ignored():
    tonic = 130.0
    f1 = np.array([tonic * 1.5] * 100 + [0.0] * 50)
    f2 = np.array([tonic * 3.0] * 100 + [0.0] * 50)  # same note, an octave up
    np.testing.assert_allclose(P.pitch_class_histogram(f1, tonic), P.pitch_class_histogram(f2, tonic))


def test_tonic_normalisation_removes_transposition():
    notes = 2 ** (np.array([0, 2, 4, 7, 9]) / 12)
    a = np.tile(notes * 100.0, 50)
    b = np.tile(notes * 150.0, 50)
    ha = P.pitch_class_histogram(a, 100.0)
    hb = P.pitch_class_histogram(b, 150.0)
    np.testing.assert_allclose(ha, hb, atol=1e-9)


def test_chunk_histograms_drop_silence_and_count_chunks():
    dt = 0.01
    times = np.arange(0, 100, dt)
    freqs = np.full(len(times), 200.0)
    freqs[(times >= 60) & (times < 100)] = 0.0  # silence at the end
    hists, starts = P.chunk_histograms(times, freqs, 200.0, chunk_s=30, hop_s=15, min_voiced_s=5)
    # chunks start at 0,15,30,45,60 (70 is beyond len-chunk); the ones starting >=60 are silent
    assert list(starts) == [0.0, 15.0, 30.0, 45.0]
    assert hists.shape == (4, 120)
    np.testing.assert_allclose(hists.sum(1), 1.0)


def test_pitch_stats():
    times = np.arange(0, 10, 0.01)
    freqs = np.where(times < 5, 100.0, 0.0)
    s = P.pitch_stats(times, freqs)
    assert abs(s["voiced_s"] - 5.0) < 0.05 and abs(s["duration_s"] - 9.99) < 1e-6
