"""End-to-end check of cache -> chunk set -> baseline on synthetic 'ragas' (different note sets)."""

import numpy as np
import pandas as pd

from raaga.eval import metrics
from raaga.features import cache as C
from raaga.features import pitch as P
from raaga.models import baseline

RAGAS = {
    "a": [0, 2, 4, 5, 7, 9, 11],
    "b": [0, 1, 4, 5, 7, 8, 11],
    "c": [0, 2, 3, 5, 7, 9, 10],
    "d": [0, 2, 4, 6, 7, 9, 11],
}


def _fake_recording(rng, semitones, tonic, seconds=200):
    dt = 0.01
    times = np.arange(0, seconds, dt)
    weights = rng.dirichlet(np.ones(len(semitones)) * 3)
    notes = rng.choice(semitones, size=len(times), p=weights)
    cents = notes * 100 + rng.normal(0, 8, len(times))
    return times, tonic * 2 ** (cents / 1200)


def test_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for raga, sem in RAGAS.items():
        for i in range(6):
            tonic = float(rng.uniform(100, 250))  # each recording has its own Sa
            t, f = _fake_recording(rng, sem, tonic)
            p = tmp_path / f"{raga}{i}.tsv"
            pd.DataFrame({"t": t, "f": f}).to_csv(p, sep="\t", header=False, index=False)
            rows.append(dict(uid=f"x:{raga}{i}", raga=raga, pitch_path=str(p), tonic_hz=tonic, fold=i % 3))
    catalog = pd.DataFrame(rows)

    cache = C.cache_dir(tmp_path, 120, 30, 15, 1.0, "test")
    C.build(catalog, cache, verbose=False)
    data = C.load(catalog, cache)
    assert data.X.shape[1] == 120 and len(data.X) == len(data.y) == len(data.uid)

    results = baseline.cross_validate(data, dict(zip(catalog.uid, catalog.fold)), folds=(0, 1, 2), C=1.0)
    assert np.mean([r["rec_top1"] for r in results]) > 0.9


def test_aggregate_and_metrics():
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
    uids = np.array(["u1", "u1", "u2"])
    order, rec = baseline.aggregate(proba, uids)
    assert order == ["u1", "u2"]
    np.testing.assert_allclose(rec.sum(1), 1.0)
    assert rec.argmax(1).tolist() == [0, 1]
    assert metrics.topk_accuracy(rec, np.array([0, 1]), 1) == 1.0
    # perfectly confident and correct -> zero calibration error
    assert metrics.expected_calibration_error(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1])) == 0.0
