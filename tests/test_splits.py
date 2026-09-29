import pandas as pd

from raaga.data import splits


def _catalog(n_ragas=6, per_raga=10):
    rows = []
    for r in range(n_ragas):
        for i in range(per_raga):
            rows.append(
                dict(
                    uid=f"d:{r}_{i}",
                    dataset="d",
                    raga=f"raga{r}",
                    artist=f"artist{i % 4}",
                    concert=f"concert{r}_{i // 2}",  # two recordings per concert
                    mbid=f"mbid{r}_{i}",
                )
            )
    return pd.DataFrame(rows)


def test_folds_are_grouped_and_valid():
    df = splits.assign_folds(_catalog(), seed=0)
    assert splits.validate_folds(df) == []
    assert set(df["fold"]) == set(range(splits.N_FOLDS))
    # every concert lives in one fold only
    assert (df.groupby("concert")["fold"].nunique() == 1).all()


def test_folds_are_deterministic():
    a = splits.assign_folds(_catalog(), seed=3)
    b = splits.assign_folds(_catalog(), seed=3)
    assert a["fold"].tolist() == b["fold"].tolist()


def test_validate_catches_leakage():
    df = splits.assign_folds(_catalog())
    df.loc[df["concert"] == df["concert"].iloc[0], "fold"] = [0, 1]  # split one concert across folds
    assert any("concert" in issue for issue in splits.validate_folds(df))


def test_filter_ragas():
    df = _catalog()
    df = df[~((df["raga"] == "raga0") & (df["uid"] != "d:0_0"))]
    out = splits.filter_ragas(df, min_recordings=5)
    assert "raga0" not in set(out["raga"]) and out["raga"].nunique() == 5


def test_save_split_has_no_paths(tmp_path):
    df = splits.assign_folds(_catalog())
    df["pitch_path"] = "/secret/path"
    splits.save_split(df, tmp_path / "s.csv")
    assert "pitch_path" not in pd.read_csv(tmp_path / "s.csv").columns


def test_thin_raga_is_still_in_the_test_fold():
    df = _catalog()
    # raga0 comes from only 3 concerts: it cannot be in all 5 folds
    df.loc[df["raga"] == "raga0", "concert"] = [f"c{i % 3}" for i in range(10)]
    out = splits.assign_folds(df, seed=1)
    assert list(splits.thin_ragas(out).index) == ["raga0"]
    assert "raga0" in set(out.loc[out["fold"] == splits.TEST_FOLD, "raga"])
    assert splits.validate_folds(out) == []
