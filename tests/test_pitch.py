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


def test_chunk_pitch_sequences_shape_and_silence_dropping():
    dt = 0.01
    times = np.arange(0, 100, dt)
    freqs = np.full(len(times), 200.0)
    freqs[(times >= 60) & (times < 100)] = 0.0  # silence at the end
    seqs, starts = P.chunk_pitch_sequences(times, freqs, 200.0, chunk_s=30, hop_s=15, step_s=0.1, min_voiced_s=5)
    assert list(starts) == [0.0, 15.0, 30.0, 45.0]
    assert seqs.shape == (4, 300, 3)  # 300 = 30s / 0.1s step
    # the first 3 chunks (0-15, 15-45, 30-60) are entirely within the voiced region
    assert (seqs[:3, :, 2] > 0.99).all()  # voiced_frac
    # the 4th chunk (45-75) runs past the voiced region (ends at 60) -> its tail is unvoiced
    assert (seqs[3, :150, 2] > 0.99).all() and (seqs[3, 150:, 2] == 0).all()
    assert np.abs(seqs[:3, :, 0]).max() < 1e-6  # sin(0) = 0 exactly at the tonic, for voiced steps


def test_chunk_pitch_sequences_octave_and_tonic_invariance():
    dt = 0.01
    times = np.arange(0, 40, dt)
    tonic = 130.0
    fifth = tonic * 2 ** (7 / 12)
    f_low = np.full(len(times), fifth)
    f_high = np.full(len(times), fifth * 2)  # same note, an octave up
    a, _ = P.chunk_pitch_sequences(times, f_low, tonic, chunk_s=30, hop_s=15)
    b, _ = P.chunk_pitch_sequences(times, f_high, tonic, chunk_s=30, hop_s=15)
    np.testing.assert_allclose(a, b, atol=1e-5)

    # same relative note, different (tonic, absolute frequency) pair -> identical sequence
    other_tonic = 200.0
    f_other = other_tonic * 2 ** (7 / 12)
    c, _ = P.chunk_pitch_sequences(times, np.full(len(times), f_other), other_tonic, chunk_s=30, hop_s=15)
    np.testing.assert_allclose(a, c, atol=1e-5)


def test_chunk_pitch_sequences_unvoiced_step_is_exactly_zero():
    times = np.arange(0, 30, 0.1)
    freqs = np.zeros(len(times))  # entirely unvoiced
    seqs, starts = P.chunk_pitch_sequences(times, freqs, 130.0, chunk_s=30, hop_s=30, min_voiced_s=0.0)
    assert len(starts) == 1
    np.testing.assert_array_equal(seqs[0], 0.0)


def test_chunk_pitch_sequences_captures_note_order_unlike_histogram():
    """The whole point of M1 over B1: two chunks with the same notes in a different ORDER must
    produce different sequences (a histogram would treat them as identical)."""
    dt = 0.01
    tonic = 100.0
    sa = tonic
    pa = tonic * 2 ** (7 / 12)
    up = np.concatenate([np.full(1000, sa), np.full(1000, pa)])
    down = np.concatenate([np.full(1000, pa), np.full(1000, sa)])
    times = np.arange(len(up)) * dt
    seq_up, _ = P.chunk_pitch_sequences(times, up, tonic, chunk_s=20, hop_s=20, min_voiced_s=0.0)
    seq_down, _ = P.chunk_pitch_sequences(times, down, tonic, chunk_s=20, hop_s=20, min_voiced_s=0.0)
    assert not np.allclose(seq_up, seq_down)
    # but time-reversing one gives the other (sanity: order is the only thing that differs)
    np.testing.assert_allclose(seq_up[0, ::-1], seq_down[0], atol=1e-5)
