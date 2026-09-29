"""hmd/saraga/catalog parsing, using tiny fixtures that mimic the real archive layouts."""

import json

import pandas as pd

from raaga.data import catalog, hmd, saraga

RAGA_ID = "7591faad-e68a-4550-b675-8082842c6056"
MBID = "6cb0fc24-76bf-48a8-8a45-8552b70127ea"


def _make_hmd(raw):
    base = raw / "compmusic_raga" / "RagaDataset" / "Hindustani"
    (base / "_info_").mkdir(parents=True)
    rec = "Raga_Bhageshri_" + MBID
    audio = f"RagaDataset/Hindustani/audio/{RAGA_ID}/Jagdish_Prasad/Shraddhanjali:_Jagdish_Prasad/{rec}"
    (base / "_info_" / "path_mbid_ragaid.json").write_text(json.dumps({MBID: dict(path=audio, mbid=MBID, ragaid=RAGA_ID)}))
    (base / "_info_" / "ragaId_to_ragaName_mapping.json").write_text(json.dumps({RAGA_ID: "Bāgēśrī"}))
    feat = base / "features" / RAGA_ID / "Jagdish_Prasad" / "Shraddhanjali:_Jagdish_Prasad"
    feat.mkdir(parents=True)
    (feat / f"{rec}.tonic").write_text("146.832384\n")
    (feat / f"{rec}.pitch").write_text("0.0\t0.0\n0.0044444\t150.0\n")


def _make_saraga(raw, lead_instrument="Voice"):
    folder = raw / "saraga_hindustani" / "saraga1.5_hindustani" / "Raag Shree by Deb" / "Raag Shree"
    folder.mkdir(parents=True)
    md = {
        "mbid": "b3a43a82-0000-0000-0000-000000000000",
        "artists": [
            {"artist": {"name": "Deb"}, "instrument": {"name": lead_instrument}, "lead": True},
            {"artist": {"name": "Rupa"}, "instrument": {"name": "Harmonium"}, "lead": False},
        ],
        "raags": [{"name": "Śrī", "common_name": "Shree"}],
        "album_artists": [{"name": "Deb"}],
    }
    (folder / "Raag Shree.json").write_text(json.dumps(md))
    (folder / "Raag Shree.ctonic.txt").write_text("138.5\n")
    (folder / "Raag Shree.pitch.txt").write_text("0.0\t0.0\n")


def test_hmd_rows(tmp_path):
    _make_hmd(tmp_path)
    (row,) = hmd.rows(tmp_path)
    assert row["raga_raw"] == "Bāgēśrī" and row["artist"] == "Jagdish_Prasad"
    assert row["concert"] == "Shraddhanjali:_Jagdish_Prasad" and row["tonic_hz"] == 146.832384
    assert row["pitch_path"].endswith(".pitch")


def test_saraga_rows_and_vocal_class(tmp_path):
    _make_saraga(tmp_path)
    (row,) = saraga.rows(tmp_path)
    assert row["vocal_class"] == "vocal" and row["raga_raw"] == "Śrī" and row["tonic_hz"] == 138.5
    assert row["concert"] == "Raag Shree by Deb"

    other = tmp_path / "other"
    _make_saraga(other, lead_instrument="Sitar")
    assert saraga.rows(other)[0]["vocal_class"] == "instrumental"


def test_catalog_merges_and_dedups(tmp_path):
    _make_hmd(tmp_path)
    _make_saraga(tmp_path)
    df = catalog.build_catalog(tmp_path, include_saraga=True)
    assert set(df.dataset) == {"hmd", "saraga"} and df.has_pitch.all()
    # 'Śrī' and 'Bāgēśrī' are different ragas
    assert df.raga.nunique() == 2
    # a recording present in both datasets is kept once (HMD wins)
    saraga_md = next((tmp_path / "saraga_hindustani").rglob("Raag Shree.json"))
    md = json.loads(saraga_md.read_text())
    md["mbid"] = MBID
    saraga_md.write_text(json.dumps(md))
    df = catalog.build_catalog(tmp_path, include_saraga=True)
    assert list(df.dataset) == ["hmd"]


def test_usable_filters_instrumental(tmp_path):
    df = pd.DataFrame(
        dict(raga=["a", "b", None], tonic_hz=[1.0, 1.0, 1.0], has_pitch=[True, True, True],
             vocal_class=["vocal", "instrumental", "vocal"])
    )
    assert list(catalog.usable(df).raga) == ["a"]
