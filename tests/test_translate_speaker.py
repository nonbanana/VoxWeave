"""The speaker id a translation prompt carries, and the guard that keeps it off screen.

Without it every cue reaches the model as an unanchored utterance; the diarizer's answer
was already parsed (``<v S3>`` voice tags) but dropped when the payload was built, so the
prompt never saw it.
"""

from __future__ import annotations

from voxweave.translate import _ECHOED_SPEAKER, build_payload, parse_response


def test_payload_carries_the_speaker_id() -> None:
    blocks = [{"text": "こんにちは", "speaker": "S1"}, {"text": "はい", "speaker": "S2"}]
    payload = build_payload(blocks)
    assert payload[0]["sp"] == "S1"
    assert payload[1]["sp"] == "S2"


def test_payload_omits_the_key_when_unnamed() -> None:
    payload = build_payload([{"text": "こんにちは"}])
    assert "sp" not in payload[0]


def test_dash_cue_keeps_its_parts() -> None:
    payload = build_payload([{"text": "-はい\n-いいえ", "speakers": ["A", "B"]}])
    assert payload[0]["parts"] == ["はい", "いいえ"]


def test_an_echoed_speaker_id_is_stripped() -> None:
    assert _ECHOED_SPEAKER.sub("", "[S3] 안녕하세요").strip() == "안녕하세요"
    assert _ECHOED_SPEAKER.sub("", "S12: 안녕하세요").strip() == "안녕하세요"
    # A line that merely starts with a capital S is not an id.
    assert _ECHOED_SPEAKER.sub("", "Sensei!").strip() == "Sensei!"


def test_parse_response_uses_the_stripped_text() -> None:
    raw = {"translations": [{"i": 0, "t": "[S1] 안녕"}, {"i": 1, "t": "네"}]}
    assert parse_response(raw) == {0: "안녕", 1: "네"}
