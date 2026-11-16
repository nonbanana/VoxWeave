"""Turn hygiene: a diarizer that flips speakers inside one word is not evidence.

Measured on a 24 min anime episode: 109 of 642 turn boundaries (17%) fell inside a word,
and 29% of adjacent turns were the same speaker. The A-B-A pass targets the first number
(a short flip between two turns of one speaker, only when the three are contiguous) and
the existing merge pass targets the second.
"""

from __future__ import annotations

from voxweave.diarize import _smooth_turns


def test_drops_a_contiguous_a_b_a_thrash() -> None:
    turns = [(0.0, 1.0, "A"), (1.05, 1.30, "B"), (1.35, 2.0, "A")]
    assert _smooth_turns(turns) == [(0.0, 2.0, "A")]


def test_keeps_a_short_reply_after_a_pause() -> None:
    """A pause means someone answered: that B is dialogue, not noise."""
    turns = [(0.0, 1.0, "A"), (2.0, 2.3, "B"), (2.4, 3.0, "A")]
    assert _smooth_turns(turns) == [(0.0, 1.0, "A"), (2.0, 2.3, "B"), (2.4, 3.0, "A")]


def test_keeps_a_three_way_exchange() -> None:
    turns = [(0.0, 1.0, "A"), (1.05, 1.3, "B"), (1.35, 2.0, "C")]
    assert len(_smooth_turns(turns)) == 3


def test_keeps_a_long_middle_turn() -> None:
    turns = [(0.0, 1.0, "A"), (1.05, 2.0, "B"), (2.05, 3.0, "A")]
    assert len(_smooth_turns(turns)) == 3


def test_two_thrashes_in_a_row_both_go() -> None:
    turns = [
        (0.0, 1.0, "A"),
        (1.02, 1.2, "B"),
        (1.22, 1.4, "A"),
        (1.42, 1.6, "C"),
        (1.62, 2.0, "A"),
    ]
    out = _smooth_turns(turns)
    assert [t[2] for t in out] == ["A"]


def test_clean_input_is_unchanged() -> None:
    turns = [(0.0, 2.0, "A"), (2.1, 4.0, "B"), (4.1, 6.0, "A")]
    assert _smooth_turns(turns) == turns
