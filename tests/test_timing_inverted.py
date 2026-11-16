"""Never write a cue whose end is before its start.

Measured on episode 03 of a broadcast anime: fourteen cues in a 40 ms burst came out with
``end`` 0.07-0.08 s *before* ``start``, because the gap-closing step computed
``nxt_start - TWO_FRAME_S`` for cues that start a few milliseconds apart.
"""

from __future__ import annotations

from voxweave.core.timing import TWO_FRAME_S, _cleanup_cues


def _cue(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end, "word_data": []}


def test_a_burst_of_cues_never_inverts() -> None:
    cues = [_cue(f"あ{i}", 139.300 + i * 0.007, 139.310 + i * 0.007) for i in range(6)]
    out = _cleanup_cues(cues, min_cue_s=0.5, max_cue_s=7.0, cps=7.0, lag_out_s=0.25)
    assert all(c["end"] > c["start"] for c in out), [(c["start"], c["end"]) for c in out]


def test_the_existing_spans_survive() -> None:
    cues = [_cue("あ", 1.0, 2.0,), _cue("い", 3.0, 4.0)]
    out = _cleanup_cues(cues, min_cue_s=0.5, max_cue_s=7.0, cps=7.0, lag_out_s=0.25)
    assert out[0]["end"] <= out[1]["start"], "a normal gap is still respected"


def test_a_degenerate_cue_gets_one_frame() -> None:
    out = _cleanup_cues([_cue("あ", 5.0, 5.0)], min_cue_s=0.0, max_cue_s=7.0, cps=0.0, lag_out_s=0.0)
    assert abs((out[0]["end"] - out[0]["start"]) - TWO_FRAME_S) < 1e-9
