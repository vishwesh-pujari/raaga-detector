"""Shape/gradient sanity checks for M1, plus a synthetic end-to-end check (cache -> model ->
train -> eval) mirroring test_baseline.py's rigor -- on tiny, fast, deterministic toy data, not
real audio (that needs a GPU and lives in the Colab notebook)."""

import numpy as np
import pandas as pd
import pytest
import torch

from raaga.eval import metrics
from raaga.features import pitch as P
from raaga.features import sequence_cache as SC
from raaga.models import sequence as M


def test_forward_pass_shape_and_gradients():
    torch.manual_seed(0)
    cfg = M.ModelConfig(n_classes=5, conv_channels=(8, 16), gru_hidden=12)
    model = M.PitchSequenceModel(cfg)
    x = torch.randn(4, 300, 3)  # (batch, T, channels) -- 300 = 30s/0.1s, the default chunking
    logits = model(x)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()

    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_forward_pass_handles_short_sequences():
    """Conv+maxpool halves T twice (two conv_channels), so T must be >= 4; check it doesn't
    silently break on the smallest sane input instead of just the default 300."""
    cfg = M.ModelConfig(n_classes=3, conv_channels=(4, 8), gru_hidden=6)
    model = M.PitchSequenceModel(cfg)
    x = torch.randn(2, 4, 3)
    assert model(x).shape == (2, 3)


def test_class_weights_favour_rare_classes():
    y = torch.tensor([0, 0, 0, 0, 1])  # class 0 appears 4x more than class 1
    w = M.class_weights(y, n_classes=2)
    assert w[1] > w[0]


RAGAS = {
    "up": [0, 2, 4, 5, 7, 9, 11],  # notes only ever visited ascending, in-order, in these tests
    "down": [11, 9, 7, 5, 4, 2, 0],
}


def _make_ordered_recording(rng, semitone_order, tonic, seconds=40, note_s=1.0):
    """A recording that strictly cycles through its notes IN ORDER, repeatedly -- something only
    a model that sees sequence order (not just a note-usage histogram) can tell apart from its
    reverse. Notes are held (not glided) so this is easy for even a tiny CNN+GRU to pick up fast.
    """
    dt = 0.01
    times = np.arange(0, seconds, dt)
    step = int(note_s / dt)
    idx = (np.arange(len(times)) // step) % len(semitone_order)
    cents = np.array(semitone_order)[idx] * 100.0 + rng.normal(0, 3, len(times))
    freqs = tonic * 2 ** (cents / 1200)
    return times, freqs


def test_end_to_end_learns_note_order_not_just_which_notes(tmp_path):
    """The actual point of M1 over B1: 'up' and 'down' use the exact same 7 notes (same pitch-class
    histogram), differing only in order. A model that ignores order can't beat chance (50%); one
    that uses order should nail this easily since it's a noiseless, deterministic toggle."""
    rng = np.random.default_rng(0)
    rows = []
    for raga, order in RAGAS.items():
        for i in range(8):
            tonic = float(rng.uniform(120, 260))
            t, f = _make_ordered_recording(rng, order, tonic)
            p = tmp_path / f"{raga}{i}.tsv"
            pd.DataFrame({"t": t, "f": f}).to_csv(p, sep="\t", header=False, index=False)
            rows.append(dict(uid=f"x:{raga}{i}", raga=raga, pitch_path=str(p), tonic_hz=tonic, fold=i % 4))
    catalog = pd.DataFrame(rows)

    cache = SC.cache_dir(tmp_path, chunk_s=20.0, hop_s=20.0, step_s=0.1, variant="test")
    SC.build(catalog, cache, chunk_s=20.0, hop_s=20.0, step_s=0.1, min_voiced_s=0.0, verbose=False)
    data = SC.load(catalog, cache)
    assert data.X.shape[1:] == (200, 3)  # 20s / 0.1s

    classes = sorted(RAGAS)
    y_idx = np.array([classes.index(y) for y in data.y])
    train = np.isin(catalog.set_index("uid").loc[data.uid, "fold"].to_numpy(), [0, 1, 2])
    val = ~train

    cfg = M.ModelConfig(n_classes=2, conv_channels=(8, 16), gru_hidden=16)
    model = M.PitchSequenceModel(cfg)
    tcfg = M.TrainConfig(epochs=15, batch_size=8, lr=2e-3, patience=15, device="cpu")
    M.fit(model, data.X[train], y_idx[train], data.X[val], y_idx[val], tcfg, verbose=False)

    proba = M.predict_proba(model, data.X[val], device="cpu")
    acc = metrics.topk_accuracy(proba, y_idx[val], 1)
    assert acc > 0.9, f"expected the model to learn note order easily, got {acc:.2f} val accuracy"


def test_sequence_cache_roundtrip(tmp_path):
    dt = 0.01
    times = np.arange(0, 40, dt)
    freqs = np.full(len(times), 150.0)
    p = tmp_path / "a.tsv"
    pd.DataFrame({"t": times, "f": freqs}).to_csv(p, sep="\t", header=False, index=False)
    catalog = pd.DataFrame([dict(uid="x:a", raga="r1", pitch_path=str(p), tonic_hz=150.0)])

    cache = SC.cache_dir(tmp_path, chunk_s=20.0, hop_s=20.0, step_s=0.1, variant="rt")
    SC.build(catalog, cache, chunk_s=20.0, hop_s=20.0, step_s=0.1, min_voiced_s=0.0, verbose=False)
    data = SC.load(catalog, cache)
    assert data.X.shape == (1, 200, 3)
    assert list(data.y) == ["r1"] and list(data.uid) == ["x:a"]

    with pytest.raises(FileNotFoundError):
        SC.load(pd.DataFrame([dict(uid="missing", raga="r1")]), cache)
