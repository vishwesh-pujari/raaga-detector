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
