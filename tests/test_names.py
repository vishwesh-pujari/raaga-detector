from raaga.data.names import load_aliases, normalize_raga, slug


def test_slug_variants_match():
    assert slug("Rāg Asawari") == slug("Asavari") == "asavari"
    assert slug("Raga Darbari  Kanada") == slug("darbari kaanada")


def test_aliases(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("bhoopali:\n  - Bhupali\n  - Bhopali\n")
    aliases = load_aliases(f)
    assert normalize_raga("Bhupali", aliases) == normalize_raga("Bhoopali", aliases)


def test_empty_names():
    assert normalize_raga(None) is None
    assert normalize_raga("  ") is None
    assert load_aliases(None) == {}
