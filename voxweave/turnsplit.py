"""Split a diarizer turn where the voice inside it changes.

pyannote decides *where* turns begin and end; the clustering step only labels those turns.
When the model puts two speakers inside one turn -- measured on a broadcast anime episode,
a 5.9 s turn held two exchanges -- nothing downstream can separate them: raising
``min_speakers`` only relabels, and both the community-1 and 3.1 pipelines produce the very
same turn. The layout then renders one line with two people in it, and the translator has
no way to tell who says what.

The evidence for a split is inside the turn: a speaker's turns are separated by pauses, and
a voice change is audible even when pyannote did not cut there. So:

1. take a long turn,
2. cut it at the word gaps inside it (a real takeover starts after a breath),
3. embed each piece with the same voice model ``--voiceprints`` uses (``anime-va`` for
   Japanese, an ECAPA-TDNN trained on anime voice actors),
4. split where adjacent pieces fall below the embedder's own same-speaker threshold.

Measured on the disputed turn (233.0-238.9 s): the two pieces either side of the 234.6 s
pause score cosine -0.011 (different voices) while the next pair scores 0.455 (same
voice, ``anime-va`` suggests 0.35 as the boundary). pyannote had called all of it one
speaker.

Everything here is pure: the embedder is injected, so the rules are testable without a
model and without audio.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

log = logging.getLogger("voxweave")

Turn = tuple[float, float, str]

#: Only turns at least this long are candidates: a short turn cannot hold two speakers, and
#: embedding it costs a forward pass.
MIN_TURN_S = 2.5
#: A pause inside the turn this long is a plausible takeover point.
MIN_GAP_S = 0.25
#: The embedder needs this much audio to say anything (anime-va's own floor is 1.0 s).
MIN_PIECE_S = 1.0
#: Cosine below this means "different voice". The value comes from the embedder spec
#: (``suggest``); this is only the fallback when the caller does not pass one.
DEFAULT_FLOOR = 0.35
#: New pieces shorter than this are folded into their neighbour: a 0.4 s fragment is not a
#: cue, and the layout would extend it into the pause anyway.
MIN_NEW_PIECE_S = 0.6


def _piece_spans(
    turn: Turn, units: Sequence[dict[str, Any]], min_gap_s: float, min_piece_s: float
) -> list[tuple[float, float]]:
    """Cut ``turn`` at its internal word gaps, keeping pieces long enough to embed."""
    start, end, _label = turn
    inside = sorted(
        (float(u["start"]), float(u["end"]))
        for u in units
        if u.get("start") is not None
        and u.get("end") is not None
        and float(u["start"]) >= start - 1e-6
        and float(u["end"]) <= end + 1e-6
    )
    if len(inside) < 2:
        return [(start, end)]
    cuts: list[float] = []
    for (_ps, pe), (ns, _ne) in zip(inside, inside[1:]):
        if ns - pe >= min_gap_s:
            cuts.append((pe + ns) / 2.0)
    if not cuts:
        return [(start, end)]
    bounds = [start, *cuts, end]
    pieces: list[tuple[float, float]] = []
    for a, b in zip(bounds, bounds[1:]):
        if pieces and b - a < min_piece_s:
            pieces[-1] = (pieces[-1][0], b)  # too short to embed: join the left piece
        elif not pieces and b - a < min_piece_s:
            pieces.append((a, b))  # the first piece sets the anchor even if short
        else:
            pieces.append((a, b))
    return pieces or [(start, end)]


def split_turns_by_voice(
    turns: Sequence[Turn],
    units: Sequence[dict[str, Any]],
    embed: Callable[[Sequence[tuple[float, float]]], Any],
    *,
    floor: float = DEFAULT_FLOOR,
    min_turn_s: float = MIN_TURN_S,
    min_gap_s: float = MIN_GAP_S,
    min_piece_s: float = MIN_PIECE_S,
) -> tuple[list[Turn], int]:
    """Return ``(turns, splits)`` with voices separated inside long turns.

    ``embed(spans)`` returns an ``[N, D]`` matrix of L2-normalised vectors, one per span
    (``voxweave.voiceembed.embed_segments``). A turn is only touched when every candidate
    piece embeds successfully; any failure leaves that turn exactly as the diarizer left
    it, because a guessed split is worse than a merged line.
    """
    if not turns:
        return [], 0
    ordered = sorted(turns)
    out: list[Turn] = []
    splits = 0
    for turn in ordered:
        start, end, label = turn
        if end - start < min_turn_s:
            out.append(turn)
            continue
        pieces = _piece_spans(turn, units, min_gap_s, min_piece_s)
        if len(pieces) < 2:
            out.append(turn)
            continue
        try:
            matrix = embed(pieces)
        except Exception as exc:  # noqa: BLE001 - an optional refinement must not fail a run
            log.debug("voice split skipped at %.2fs: %s", start, exc)
            out.append(turn)
            continue
        keep = [0]
        for i in range(1, len(pieces)):
            sim = float(sum(a * b for a, b in zip(matrix[i - 1], matrix[i])))
            if not math.isfinite(sim) or sim < floor:
                keep.append(i)
        if len(keep) < 2:
            out.append(turn)
            continue
        bounds = keep + [len(pieces)]
        for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
            p_start, p_end = pieces[a][0], pieces[b - 1][1]
            if (
                p_end - p_start < MIN_NEW_PIECE_S
                and out
                and abs(out[-1][1] - p_start) < 1e-6
            ):
                # Fold a scrap into the piece before it instead of emitting a flash cue.
                out[-1] = (out[-1][0], p_end, out[-1][2])
                continue
            if k == 0:
                out.append((p_start, p_end, label))
            else:
                # A new label, derived so it stays traceable to the turn it came from
                # (SPEAKER_09 -> SPEAKER_09r1). Names are mapped per id downstream, so an
                # unmapped id simply has no name yet.
                splits += 1
                out.append((p_start, p_end, f"{label}r{k}"))
    return out, splits
