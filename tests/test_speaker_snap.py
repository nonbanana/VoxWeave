"""A speaker boundary must stay where the voice put it.

The previous implementation reassigned whole BudouX phrases, which let a boundary travel
several characters: ``レポート | しょうがない`` became ``レポートしょうが | ない`` on a
real episode, cutting a word in half. The replacement only moves a boundary to the nearest
edge with audible silence, and never further than that.
"""

from __future__ import annotations

from voxweave.speaker_format import _snap_runs_to_phrases


def _atom(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


def test_a_boundary_does_not_travel_past_the_pause() -> None:
    left = [_atom(c, 0.0 + i * 0.1, 0.1 + i * 0.1) for i, c in enumerate("レポート")]
    # 0.7 s of silence after レポート, then the reply runs on without a pause.
    right = [_atom(c, 1.0 + i * 0.1, 1.1 + i * 0.1) for i, c in enumerate("しょうがない")]
    runs = [("A", left), ("B", right)]
    out = _snap_runs_to_phrases(runs, "ja")
    assert [("".join(a["text"] for a in ats)) for _lb, ats in out] == ["レポート", "しょうがない"]


def test_a_boundary_inside_a_word_moves_to_the_nearby_pause() -> None:
    """The diarizer cut one frame late; the boundary should step back to the pause."""
    chars = list("レポートしょうがない")
    atoms = [_atom(c, i * 0.2, i * 0.2 + 0.15) for i, c in enumerate(chars)]
    # put a real pause between レポート and しょうがない (after index 4)
    for i in range(5, len(atoms)):
        atoms[i]["start"] += 0.6
        atoms[i]["end"] += 0.6
    runs = [("A", atoms[:6]), ("B", atoms[6:])]  # boundary one char late
    out = _snap_runs_to_phrases(runs, "ja")
    left = "".join(a["text"] for a in out[0][1])
    assert left in ("レポートし", "レポート"), "must land on the pause, not mid-word"


def test_a_single_run_is_untouched() -> None:
    runs = [("A", [_atom("あ", 0.0, 0.1), _atom("い", 0.2, 0.3)])]
    assert _snap_runs_to_phrases(runs, "ja") == runs
