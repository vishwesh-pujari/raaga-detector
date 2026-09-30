from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def topk_accuracy(proba: np.ndarray, y_idx: np.ndarray, k: int = 1) -> float:
    k = min(k, proba.shape[1])
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_idx, top)]))


def macro_f1(proba: np.ndarray, y_idx: np.ndarray) -> float:
    pred = proba.argmax(1)
    # only classes that occur (as truth or prediction), so a fold missing a raga isn't penalised
    labels = np.unique(np.concatenate([y_idx, pred]))
    return float(f1_score(y_idx, pred, average="macro", labels=labels, zero_division=0))


def confusion(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_idx, proba.argmax(1), labels=np.arange(proba.shape[1]))


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> float:
    """Top-1 ECE: |confidence - accuracy| averaged over equal-width confidence bins."""
    conf = proba.max(1)
    correct = (proba.argmax(1) == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def aggregate(proba: np.ndarray, uids: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Chunk probabilities -> one probability vector per recording (mean log-prob, renormalised).

    Shared by every model (B1, M1, ...): this *is* the chunk -> clip pipeline the app will use, so
    it lives here rather than in any one model's module, keeping every model's evaluation directly
    comparable.
    """
    logp = np.log(np.clip(proba, 1e-9, 1.0))
    order = list(dict.fromkeys(uids))
    out = np.zeros((len(order), proba.shape[1]))
    for i, u in enumerate(order):
        m = logp[uids == u].mean(axis=0)
        e = np.exp(m - m.max())
        out[i] = e / e.sum()
    return order, out


def evaluate_predictions(chunk_proba: np.ndarray, y_idx: np.ndarray, uids: np.ndarray, classes: List[str]) -> Dict[str, object]:
    """Chunk-level probabilities + integer labels -> the standard results dict (recording-level
    top-1/top-3/macro-F1/ECE/confusion). Model-agnostic -- every model reports through this so
    results are always structured the same way and safe to compare directly.
    """
    rec_uids, rec_p = aggregate(chunk_proba, uids)
    first = {u: i for i, u in reversed(list(enumerate(uids)))}
    rec_y = np.array([y_idx[first[u]] for u in rec_uids])
    return {
        "n_chunks": len(chunk_proba),
        "n_recordings": len(rec_uids),
        "chunk_top1": topk_accuracy(chunk_proba, y_idx, 1),
        "rec_top1": topk_accuracy(rec_p, rec_y, 1),
        "rec_top3": topk_accuracy(rec_p, rec_y, 3),
        "rec_macro_f1": macro_f1(rec_p, rec_y),
        "rec_ece": expected_calibration_error(rec_p, rec_y),
        "confusion": confusion(rec_p, rec_y),
        "classes": classes,
    }
