from raaga.data.musicbrainz import classify_recording


def test_vocal():
    rec = {"relations": [{"type": "vocal", "attributes": []}, {"type": "instrument", "attributes": ["tabla"]}]}
    assert classify_recording(rec)["vocal_class"] == "vocal"


def test_instrumental():
    rec = {"relations": [{"type": "instrument", "attributes": ["Sitar"]}]}
    out = classify_recording(rec)
    assert out["vocal_class"] == "instrumental" and out["instruments"] == ["sitar"]


def test_unknown_when_no_credits():
    assert classify_recording({})["vocal_class"] == "unknown"
