"""Tonic-error sensitivity: how much does B1/M1 accuracy degrade if the tonic they're given is
wrong by some amount?

Deliberately does **not** retrain per offset -- it evaluates the already-frozen model at inference
time only, since that's the actual deployment scenario: train once (on the best tonic available at
training time), then infer on whatever tonic an estimator produces for new audio later. Training
and testing with the same wrong tonic would conflate two different effects and isn't what happens
in production.

Needs no HMD audio at all: offsets are injected into every recording's already-known *oracle*
tonic (from the cached pitch files), not derived from running an extractor on audio we don't have.
See PLAN.md Phase 4a for the full reasoning and how this combines with the separate extractor
validation (``features/extract.py``, run against Saraga, which measures how large a *real*
estimator's error actually is) to get an end-to-end, audio-access-honest accuracy estimate.
"""

from typing import Callable, Sequence

import pandas as pd


def offset_tonic(tonic_hz: float, offset_cents: float) -> float:
    """Shift a tonic by ``offset_cents`` -- simulates an estimator whose guess is that far off (in
    cents, the same log-frequency unit used everywhere else in this codebase; see CONCEPTS.md)."""
    return float(tonic_hz) * 2 ** (offset_cents / 1200.0)


def perturb_catalog_tonic(catalog: pd.DataFrame, offset_cents: float, tonic_col: str = "tonic_hz") -> pd.DataFrame:
    """A copy of ``catalog`` with every row's tonic shifted by the same ``offset_cents``."""
    out = catalog.copy()
    out[tonic_col] = out[tonic_col].map(lambda hz: offset_tonic(hz, offset_cents))
    return out


def sweep(
    catalog: pd.DataFrame,
    offsets_cents: Sequence[float],
    build_and_evaluate: Callable[[pd.DataFrame], dict],
) -> pd.DataFrame:
    """Run ``build_and_evaluate`` once per offset (plus once, unperturbed, at 0) and collect the
    results into one table.

    ``build_and_evaluate`` closes over whatever's model-specific (the already-trained/loaded model,
    the feature-cache builder to use) and takes just the tonic-perturbed catalog: rebuild that
    model's input features with the perturbed tonic, run inference, return an
    ``eval.metrics.evaluate_predictions``-style dict. Writing it this way means the exact same
    ``sweep()`` call works for B1 and M1 -- only the closure passed in differs.
    """
    rows = []
    for offset in [0.0] + list(offsets_cents):
        result = build_and_evaluate(perturb_catalog_tonic(catalog, offset))
        rows.append({"offset_cents": offset, **result})
    return pd.DataFrame(rows)
