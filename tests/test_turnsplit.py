"""Splitting a diarizer turn where the voice inside it changes.

Real numbers: on the disputed 233.0-238.9 s turn, anime-va centroids read 0.383 for the
opening line (speaker A) and 0.578 for the reply (speaker B), while comparing neighbouring
1.4 s pieces read 0.225 for two A pieces and 0.455 for A vs B -- the wrong way round. That
is why the implementation ties itself to centroids, and why these tests inject the
embedder: the rules, not the model, are what is being pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxweave.turnsplit import build_centroids, split_turns_by_voice

# Two voices: anything starting before 234.6 s is A, after it B, except the last line.
A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])


#: The B speaker's own turns (used to build its centroid), plus the reply inside the
#: disputed turn: B runs from the cut at 234.54 to the reply's cut at 235.96.
B_TURNS = {(40.0, 42.0), (50.0, 52.0), (60.0, 62.0)}


def _voice(start: float, end: float) -> np.ndarray:
    if (start, end) in B_TURNS or 234.54 <= start < 235.9:
        return B
    return A


def _embed(spans):
    return np.stack([_voice(start, end) for start, end in spans])


UNITS = [
    {"text": "こ", "start": 233.05, "end": 233.2},
    {"text": "れ", "start": 233.25, "end": 233.4},
    {"text": "。", "start": 234.44, "end": 234.54},
    {"text": "何", "start": 234.78, "end": 234.9},
    {"text": "の", "start": 235.0, "end": 235.1},
    {"text": "?", "start": 235.9, "end": 235.96},
    {"text": "レ", "start": 236.54, "end": 236.7},
    {"text": "。", "start": 238.7, "end": 238.75},
]

TURNS = [
    (10.0, 12.0, "A"),
    (20.0, 22.0, "A"),
    (30.0, 32.0, "A"),
    (40.0, 42.0, "B"),
    (50.0, 52.0, "B"),
    (60.0, 62.0, "B"),
    (233.0, 238.9, "A"),
]


def test_punctuation_pieces_are_used_not_pauses() -> None:
    """``何``/``の`` are 0.1 s apart; cutting on gaps would split the phrase."""
    turns, splits = split_turns_by_voice(TURNS, UNITS, _embed)
    pieces = [t for t in turns if t[0] >= 233.0]
    assert splits == 2
    assert [round(p[0], 2) for p in pieces] == [233.0, 234.54, 235.96]


def test_centroids_come_from_long_turns_only() -> None:
    short = [(100.0 + i, 100.3 + i, "C") for i in range(5)]
    centroids = build_centroids(TURNS + short, _embed)
    assert set(centroids) == {"A", "B"}, "C has no turn long enough to sample"


def test_a_weak_assignment_keeps_the_diarizer_label() -> None:
    """Both centroids nearly equally far: do not split on a coin flip."""

    def ambiguous(spans):
        return np.stack([np.array([0.70, 0.71, 0.0]) for _ in spans])

    turns, splits = split_turns_by_voice(TURNS, UNITS, ambiguous, margin=0.15)
    assert splits == 0
    assert [t for t in turns if t[0] >= 233.0] == [(233.0, 238.9, "A")]


def test_a_single_voice_turn_is_untouched() -> None:
    turns, splits = split_turns_by_voice(TURNS, UNITS, lambda spans: np.stack([A] * len(spans)))
    assert splits == 0


def test_an_embedder_failure_leaves_the_turn_untouched() -> None:
    def boom(_spans):
        raise RuntimeError("model not available")

    turns, splits = split_turns_by_voice(TURNS, UNITS, boom)
    assert splits == 0
    assert len(turns) == len(TURNS)


def test_no_punctuation_means_no_split() -> None:
    plain = [{"text": "あ", "start": 233.1, "end": 233.2}, {"text": "い", "start": 236.0, "end": 236.1}]
    turns, splits = split_turns_by_voice(TURNS, plain, _embed)
    assert splits == 0
