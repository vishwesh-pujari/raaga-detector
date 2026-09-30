"""Unit tests for the offset math, plus an end-to-end check that the sweep methodology actually
detects degradation -- i.e. that evaluating an already-trained B1 model with an increasingly wrong
tonic makes its accuracy get worse, on a synthetic task designed so this is unambiguous."""

import numpy as np
import pandas as pd
import pytest

from raaga.eval import metrics, tonic_robustness as TR
from raaga.features import cache as C
from raaga.models import baseline

RAGAS = {
    "a": [0, 2, 4, 5, 7, 9, 11],
    "b": [0, 1, 4, 5, 7, 8, 11],
    "c": [0, 2, 3, 5, 7, 9, 10],
    "d": [0, 2, 4, 6, 7, 9, 11],
}


def test_offset_tonic_math():
    assert TR.offset_tonic(146.83, 0.0) == pytest.approx(146.83)
    assert TR.offset_tonic(100.0, 1200.0) == pytest.approx(200.0)  # +1 octave -> doubles
    assert TR.offset_tonic(200.0, -1200.0) == pytest.approx(100.0)  # -1 octave -> halves
    assert TR.offset_tonic(100.0, 700.0) == pytest.approx(100.0 * 2 ** (7 / 12))  # +a fifth


def test_perturb_catalog_tonic_shifts_every_row():
    df = pd.DataFrame({"uid": ["a", "b"], "tonic_hz": [100.0, 200.0]})
    out = TR.perturb_catalog_tonic(df, 1200.0)
    assert out.tonic_hz.tolist() == [200.0, 400.0]
    assert df.tonic_hz.tolist() == [100.0, 200.0]  # original untouched


def test_sweep_calls_build_and_evaluate_once_per_offset_including_zero():
    calls = []

    def fake(perturbed):
        calls.append(perturbed.tonic_hz.iloc[0])
        return {"rec_top1": 1.0}

    df = pd.DataFrame({"uid": ["a"], "tonic_hz": [100.0]})
    out = TR.sweep(df, [50.0, 100.0], fake)
    assert calls == [100.0, TR.offset_tonic(100.0, 50.0), TR.offset_tonic(100.0, 100.0)]
    assert out["offset_cents"].tolist() == [0.0, 50.0, 100.0]
    assert (out["rec_top1"] == 1.0).all()


def _fake_recording(rng, semitones, tonic, seconds=200):
    dt = 0.01
    times = np.arange(0, seconds, dt)
    weights = rng.dirichlet(np.ones(len(semitones)) * 3)
    notes = rng.choice(semitones, size=len(times), p=weights)
    cents = notes * 100 + rng.normal(0, 8, len(times))
    return times, tonic * 2 ** (cents / 1200)


def test_sweep_detects_real_degradation_as_tonic_error_grows(tmp_path):
    """The point of the whole module: does evaluating an already-trained model with an
    increasingly wrong tonic actually make accuracy get worse? Uses the same synthetic-raga setup
    as test_baseline.py's end-to-end test, so a large offset should eventually push accuracy
    toward chance (0.25 for 4 ragas), while offset 0 should recover B1's normal high accuracy.
    """
    rng = np.random.default_rng(0)
    rows = []
    for raga, sem in RAGAS.items():
        for i in range(8):
            tonic = float(rng.uniform(100, 250))
            t, f = _fake_recording(rng, sem, tonic)
            p = tmp_path / f"{raga}{i}.tsv"
            pd.DataFrame({"t": t, "f": f}).to_csv(p, sep="\t", header=False, index=False)
            rows.append(dict(uid=f"x:{raga}{i}", raga=raga, pitch_path=str(p), tonic_hz=tonic, fold=i % 4))
    catalog = pd.DataFrame(rows)
    train_catalog = catalog[catalog.fold != 3]
    test_catalog = catalog[catalog.fold == 3]

    train_cache = C.cache_dir(tmp_path, 120, 30, 15, 1.0, "train")
    C.build(train_catalog, train_cache, verbose=False)
    train_data = C.load(train_catalog, train_cache)
    model = baseline.make_model(C=1.0).fit(train_data.X, train_data.y)
    classes = list(model.classes_)

    def build_and_evaluate(perturbed_test_catalog):
        cache = C.cache_dir(tmp_path, 120, 30, 15, 1.0, f"eval_{perturbed_test_catalog.tonic_hz.sum():.3f}")
        C.build(perturbed_test_catalog, cache, verbose=False)
        data = C.load(perturbed_test_catalog, cache)
        y_idx = np.array([classes.index(y) for y in data.y])
        proba = model.predict_proba(data.X)
        return metrics.evaluate_predictions(proba, y_idx, data.uid, classes)

    out = TR.sweep(test_catalog, [50.0, 150.0, 400.0, 900.0], build_and_evaluate)

    assert out.loc[out.offset_cents == 0.0, "rec_top1"].iloc[0] > 0.85  # near-normal at no offset
    # a large, garbled offset should be clearly worse than no offset at all -- not asserting
    # monotonicity at every step (a single random ~900 cent shift can coincidentally land near
    # another note), just that severe error hurts badly on average relative to the clean case
    worst = out.loc[out.offset_cents == 900.0, "rec_top1"].iloc[0]
    assert worst < out.loc[out.offset_cents == 0.0, "rec_top1"].iloc[0]
