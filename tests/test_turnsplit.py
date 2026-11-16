"""Splitting a diarizer turn where the voice inside it changes.

Real numbers behind the tests: on the disputed 233.0-238.9 s turn, the pieces either side
of the 234.6 s pause scored cosine -0.011 (different voices) while the next pair scored
0.455 (same voice, threshold 0.35). pyannote called all of it one speaker, so the whole
exchange reached the layout as a single line.
"""

from __future__ import annotations

from voxweave.turnsplit import split_turns_by_voice

UNITS = [
    {"start": 233.05, "end": 234.45},
    {"start": 234.78, "end": 236.20},
    {"start": 236.35, "end": 238.75},
]


def _embed_by_start(spans):
    """Two voices: everything before 234.6 s is one, after it another."""
    return [[1.0, 0.0] if a < 234.6 else [0.0, 1.0] for a, _b in spans]


def test_splits_a_turn_that_holds_two_voices() -> None:
    turns, splits = split_turns_by_voice([(233.0, 238.9, "SPEAKER_09")], UNITS, _embed_by_start)
    assert splits == 1
    assert [t[2] for t in turns] == ["SPEAKER_09", "SPEAKER_09r1"]
    assert turns[0][1] == turns[1][0], "the pieces must abut"


def test_leaves_one_voice_alone() -> None:
    def same(_spans):
        return [[1.0, 0.0] for _ in _spans]

    turns, splits = split_turns_by_voice([(233.0, 238.9, "SPEAKER_09")], UNITS, same)
    assert splits == 0
    assert turns == [(233.0, 238.9, "SPEAKER_09")]


def test_ignores_a_short_turn() -> None:
    turns, splits = split_turns_by_voice([(233.0, 234.0, "A")], UNITS, _embed_by_start)
    assert splits == 0 and turns == [(233.0, 234.0, "A")]


def test_ignores_a_turn_without_internal_pauses() -> None:
    tight = [{"start": 10.0, "end": 11.0}, {"start": 11.05, "end": 12.0}]
    turns, splits = split_turns_by_voice([(10.0, 12.0, "A")], tight, _embed_by_start)
    assert splits == 0


def test_an_embedder_failure_leaves_the_turn_untouched() -> None:
    def boom(_spans):
        raise RuntimeError("model not available")

    turns, splits = split_turns_by_voice([(233.0, 238.9, "SPEAKER_09")], UNITS, boom)
    assert splits == 0
    assert turns == [(233.0, 238.9, "SPEAKER_09")]


def test_the_threshold_is_the_embedder_floor() -> None:
    """With the floor dropped below the pair's cosine, nothing splits."""
    turns, splits = split_turns_by_voice(
        [(233.0, 238.9, "SPEAKER_09")], UNITS, _embed_by_start, floor=-1.0
    )
    assert splits == 0
