"""Baseline B1: pitch-class histogram (tonic-normalised) -> logistic regression.

Trained on 30 s chunks; a recording's prediction is the mean of its chunks' log-probabilities.
This is exactly the chunk -> clip pipeline the final app will use, so the numbers are directly
comparable with the deep models later.
"""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from raaga.eval import metrics
from raaga.features.cache import ChunkSet


def make_model(C: float = 1.0):
    # sqrt of a normalised histogram (Hellinger-style) evens out dominant vs weak notes
    return make_pipeline(
        FunctionTransformer(np.sqrt),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
    )


def evaluate(model, data: ChunkSet) -> Dict[str, object]:
    classes = list(model.classes_)
    y_idx = np.array([classes.index(y) for y in data.y])
    chunk_p = model.predict_proba(data.X)
    return metrics.evaluate_predictions(chunk_p, y_idx, data.uid, classes)


def cross_validate(
    data: ChunkSet, fold_of_uid: Dict[str, int], folds: Sequence[int] = (0, 1, 2, 3), C: float = 1.0
) -> List[Dict[str, object]]:
    """Leave-one-fold-out over ``folds`` (the test fold is never in ``folds``)."""
    results = []
    fold_arr = np.array([fold_of_uid[u] for u in data.uid])
    for k in folds:
        train = np.isin(fold_arr, [f for f in folds if f != k])
        val = fold_arr == k
        tr = ChunkSet(data.X[train], data.y[train], data.uid[train], data.start_s[train])
        va = ChunkSet(data.X[val], data.y[val], data.uid[val], data.start_s[val])
        model = make_model(C).fit(tr.X, tr.y)
        res = evaluate(model, va)
        res["fold"] = k
        results.append(res)
    return results
