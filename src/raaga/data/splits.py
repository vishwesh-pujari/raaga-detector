"""Recording-level train/val/test folds.

Rule: a recording (and everything from the same concert/album) is in exactly one fold. Clips or
chunks of one recording must never be spread across folds, or accuracy is meaningless.

Convention: ``N_FOLDS = 5``. Fold ``TEST_FOLD`` (4) is the held-out test set, touched only for
final numbers. Folds 0-3 are used for cross-validation / model selection.
"""

from typing import List

import numpy as np
import pandas as pd

N_FOLDS = 5
TEST_FOLD = 4


def filter_ragas(df: pd.DataFrame, min_recordings: int) -> pd.DataFrame:
    counts = df["raga"].value_counts()
    keep = counts[counts >= min_recordings].index
    return df[df["raga"].isin(keep)].reset_index(drop=True)


def assign_folds(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = 0,
    group_col: str = "concert",
    n_restarts: int = 20,
) -> pd.DataFrame:
    """Add a ``fold`` column: every group (concert; fallback: the recording) is in one fold, and
    each raga's recordings are spread over the folds as evenly as possible.

    sklearn's StratifiedGroupKFold does not guarantee that every raga appears in every fold once
    recordings are grouped (we have only ~10 per raga), so this uses a greedy assignment with
    random restarts, keeping the assignment with the fewest empty (raga, fold) cells and then the
    most even spread. Deterministic for a given seed.
    """
    df = df.reset_index(drop=True).copy()
    groups = df[group_col].where(df[group_col].notna(), df["uid"]).astype(str)
    # concert names are only unique within a dataset; a missing concert falls back to the uid
    groups = df["dataset"].astype(str) + "/" + groups
    g_names, g_of_row = np.unique(groups, return_inverse=True)
    ragas, r_of_row = np.unique(df["raga"], return_inverse=True)
    G = np.zeros((len(g_names), len(ragas)))
    np.add.at(G, (g_of_row, r_of_row), 1)

    target = G.sum(0) / n_folds  # ideal recordings of each raga per fold

    def score(F):
        # fewest empty (raga, fold) cells first, then evenness of every raga and of the fold sizes
        return (int((F == 0).sum()), float(((F - target) ** 2).sum() + ((F.sum(1) - F.sum() / n_folds) ** 2).sum()))

    rng = np.random.default_rng(seed)
    best, best_score = None, None
    for _ in range(n_restarts):
        order = rng.permutation(len(g_names))
        order = order[np.argsort(-G[order].sum(1), kind="stable")]  # big groups first
        F = np.zeros((n_folds, len(ragas)))
        assign = np.empty(len(g_names), dtype=int)
        for g in order:
            in_group = G[g] > 0
            cost = ((F[:, in_group] + G[g, in_group]) ** 2).sum(1) + 1e-3 * F.sum(1)
            k = int(np.argmin(cost + rng.random(n_folds) * 1e-6))
            assign[g] = k
            F[k] += G[g]
        # local search: move one group to another fold while that improves the score
        current, improved = score(F), True
        while improved:
            improved = False
            for g in rng.permutation(len(g_names)):
                k0 = assign[g]
                for k in range(n_folds):
                    if k == k0:
                        continue
                    F[k0] -= G[g]
                    F[k] += G[g]
                    s = score(F)
                    if s < current:
                        current, assign[g], k0, improved = s, k, k, True
                    else:
                        F[k] -= G[g]
                        F[k0] += G[g]
        if best_score is None or current < best_score:
            best, best_score = assign.copy(), current
    fold = best[g_of_row]
    # Make the held-out test fold (TEST_FOLD) the fold that covers the most ragas, so that a raga
    # with too few concerts to be in every fold is at least always in the test set.
    coverage = [df["raga"][fold == k].nunique() for k in range(n_folds)]
    sizes = np.bincount(fold, minlength=n_folds)
    pick = min(range(n_folds), key=lambda k: (-coverage[k], abs(sizes[k] - len(df) / n_folds)))
    if n_folds - 1 == TEST_FOLD and pick != TEST_FOLD:
        swap = {pick: TEST_FOLD, TEST_FOLD: pick}
        fold = np.array([swap.get(k, k) for k in fold])
    df["fold"] = fold
    return df


def thin_ragas(df: pd.DataFrame, n_folds: int = N_FOLDS, group_col: str = "concert") -> pd.DataFrame:
    """Ragas whose recordings come from fewer concerts than there are folds. Under concert-level
    grouping such a raga cannot be present in every fold: its results are less reliable."""
    g = (df["dataset"].astype(str) + "/" + df[group_col].where(df[group_col].notna(), df["uid"]).astype(str))
    n = g.groupby(df["raga"]).nunique()
    return n[n < n_folds].rename("concerts").to_frame()


def validate_folds(df: pd.DataFrame, group_col: str = "concert") -> List[str]:
    """Return a list of problems (empty list = fine)."""
    issues = []
    if (df["fold"] < 0).any():
        issues.append("some recordings have no fold")
    if df["uid"].duplicated().any():
        issues.append("duplicate uid")
    mbid = df[df["mbid"].notna()]
    if (mbid.groupby("mbid")["fold"].nunique() > 1).any():
        issues.append("same MBID appears in several folds")
    if group_col in df:
        g = df.assign(_g=df["dataset"].astype(str) + "/" + df[group_col].astype(str))
        g = g[df[group_col].notna()]
        if (g.groupby("_g")["fold"].nunique() > 1).any():
            issues.append(f"some {group_col} is spread over several folds")
    all_ragas = set(df["raga"])
    n_folds = int(df["fold"].max()) + 1
    thin = set(thin_ragas(df, n_folds, group_col).index) if group_col in df else set()
    for k in range(n_folds):
        missing = all_ragas - set(df.loc[df["fold"] != k, "raga"])
        if missing:
            issues.append(f"fold {k} held out leaves no training data for {sorted(missing)}")
        # a thin raga may legitimately be absent from a CV fold, but never from the test fold
        absent = all_ragas - set(df.loc[df["fold"] == k, "raga"]) - (thin if k != TEST_FOLD else set())
        if absent:
            issues.append(f"fold {k} has no recording of {len(absent)} raga(s), e.g. {sorted(absent)[:3]}")
    return issues


def artist_overlap(df: pd.DataFrame, test_fold: int = TEST_FOLD) -> float:
    """Fraction of test recordings whose artist also appears in the training folds.

    Diagnostic only: raga recognisers can partly learn the artist/recording setup, so if this is
    high, results are optimistic compared with unseen singers.
    """
    test = df[df["fold"] == test_fold]
    train_artists = set(df.loc[df["fold"] != test_fold, "artist"].dropna())
    known = test["artist"].dropna()
    return float(known.isin(train_artists).mean()) if len(known) else float("nan")


SPLIT_COLUMNS = ["uid", "dataset", "raga", "artist", "concert", "mbid", "fold"]


def save_split(df: pd.DataFrame, path) -> None:
    """Write the committed split file (ids and labels only, no paths or audio)."""
    df[SPLIT_COLUMNS].sort_values("uid").to_csv(path, index=False)
