"""Split a diarizer turn where the voice inside it changes.

pyannote decides *where* turns begin and end; the clustering step only labels those turns.
When two speakers land inside one turn nothing downstream can separate them -- raising
``min_speakers`` only relabels, and community-1 and 3.1 emit the very same turn. The layout
then renders one line with two people in it, and the translator cannot tell who says what.

Two things had to be measured before this worked:

* **Cut points are punctuation, not pauses.** Inside a turn the aligner leaves 0.16-0.30 s
  gaps between characters, and the real line boundary (0.24 s) is indistinguishable from
  that noise: cutting on gaps split ``何の/接点`` mid-phrase. A speaker change in dialogue
  lands after 「、」「。」「?」, so the cut set is those units.
* **Compare against centroids, not the neighbour.** A 1.4 s piece compared to the piece
  beside it is unreliable: with anime-va two pieces of the *same* speaker scored 0.225
  while two *different* speakers scored 0.455. Compared to per-speaker centroids built
  from many turns, the same pieces read 0.383 / 0.578 and the boundary is visible.

A piece only counts when its best centroid beats the runner-up by ``margin``; otherwise the
turn keeps the diarizer's label. That gate is what keeps an A-B-A exchange as A-B-A
instead of letting a weak third assignment split it again.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

log = logging.getLogger("voxweave")

Turn = tuple[float, float, str]

#: Only turns at least this long are candidates: a short turn cannot hold two speakers.
MIN_TURN_S = 2.5
#: The pieces either side of a cut must be this long before the embedder says anything
#: (anime-va's own floor is 1.0 s).
MIN_PIECE_S = 1.0
#: A turn shorter than this is not worth a centroid sample.
CENTROID_MIN_TURN_S = 1.5
#: How many samples a speaker needs before its centroid is trusted.
CENTROID_MIN_SAMPLES = 3
#: How far the best centroid must beat the runner-up before an assignment is used.
DEFAULT_MARGIN = 0.15

#: Clause and sentence punctuation. A comma is a real speaker boundary in dialogue: on the
#: disputed turn the exchange is A-B-A-B, with ``レポート、`` and ``しょうがないっすね。``
#: alternating -- cutting only at sentence enders merges them into one piece.
CUT_PUNCT = frozenset("、。，．？！…‥,.!?")


def _cut_after(unit: dict[str, Any]) -> bool:
    text = str(unit.get("text") or unit.get("word") or "").strip()
    return bool(text) and text[-1] in CUT_PUNCT


def _pieces(turn: Turn, units: Sequence[dict[str, Any]], min_piece_s: float) -> list[tuple[float, float]]:
    """Cut ``turn`` after its clause punctuation, merging pieces too short to embed."""
    start, end, _label = turn
    inside = sorted(
        (
            (float(u["start"]), float(u["end"]), u)
            for u in units
            if u.get("start") is not None
            and u.get("end") is not None
            and float(u["start"]) >= start - 1e-6
            and float(u["end"]) <= end + 1e-6
        ),
        key=lambda row: (row[0], row[1]),
    )
    if len(inside) < 2:
        return [(start, end)]
    cuts = [pe for (_ps, pe, u) in inside[:-1] if _cut_after(u)]
    if not cuts:
        return [(start, end)]
    bounds = [start, *cuts, end]
    # Accumulate until a piece is long enough to embed. A 0.1 s scrap (the punctuation
    # unit itself often sits at zero width) must never become its own piece: it carries no
    # voice, and the embedder rejects it.
    pieces: list[tuple[float, float]] = []
    anchor = bounds[0]
    for edge in bounds[1:]:
        if edge - anchor >= min_piece_s:
            pieces.append((anchor, edge))
            anchor = edge
    if anchor < bounds[-1]:
        if pieces:
            pieces[-1] = (pieces[-1][0], bounds[-1])
        else:
            pieces.append((anchor, bounds[-1]))
    return pieces or [(start, end)]


def build_centroids(
    turns: Sequence[Turn],
    embed: Callable[[Sequence[tuple[float, float]]], Any],
    *,
    min_turn_s: float = CENTROID_MIN_TURN_S,
    min_samples: int = CENTROID_MIN_SAMPLES,
) -> dict[str, np.ndarray]:
    """One unit-length centroid per speaker, from that speaker's longer turns."""
    by_label: dict[str, list[tuple[float, float]]] = {}
    for start, end, label in sorted(turns):
        if end - start >= min_turn_s:
            by_label.setdefault(label, []).append((start, end))
    usable = {k: v for k, v in by_label.items() if len(v) >= min_samples}
    if len(usable) < 2:
        return {}
    spans = [span for group in usable.values() for span in group]
    matrix = np.asarray(embed(spans), dtype=np.float64)
    centroids: dict[str, np.ndarray] = {}
    offset = 0
    for label, group in usable.items():
        rows = matrix[offset : offset + len(group)]
        offset += len(group)
        mean = rows.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-9:
            centroids[label] = mean / norm
    return centroids


def _assign(
    row: np.ndarray, centroids: dict[str, np.ndarray], fallback: str, margin: float
) -> str:
    scores = sorted(
        ((float(np.dot(row, vector)), label) for label, vector in centroids.items()),
        reverse=True,
    )
    if len(scores) < 2:
        return fallback
    best, best_label = scores[0]
    second = scores[1][0]
    if not math.isfinite(best) or best - second < margin:
        return fallback
    return best_label


def split_turns_by_voice(
    turns: Sequence[Turn],
    units: Sequence[dict[str, Any]],
    embed: Callable[[Sequence[tuple[float, float]]], Any],
    *,
    margin: float = DEFAULT_MARGIN,
    min_turn_s: float = MIN_TURN_S,
    min_piece_s: float = MIN_PIECE_S,
    centroids: dict[str, np.ndarray] | None = None,
) -> tuple[list[Turn], int]:
    """Return ``(turns, splits)`` with voices separated inside long turns.

    ``embed(spans)`` returns an ``[N, D]`` matrix of L2-normalised vectors, one per span
    (``voxweave.voiceembed.embed_segments``). Every failure path returns the diarizer's
    turns untouched: a guessed split is worse than a merged line.
    """
    if not turns:
        return [], 0
    ordered = sorted(turns)
    if centroids is None:
        try:
            centroids = build_centroids(ordered, embed)
        except Exception as exc:  # noqa: BLE001 - optional refinement
            log.debug("voice split: no centroids (%s)", exc)
            return list(ordered), 0
    if len(centroids) < 2:
        return list(ordered), 0

    out: list[Turn] = []
    splits = 0
    for turn in ordered:
        start, end, label = turn
        if end - start < min_turn_s:
            out.append(turn)
            continue
        pieces = _pieces(turn, units, min_piece_s)
        if len(pieces) < 2:
            out.append(turn)
            continue
        try:
            matrix = np.asarray(embed(pieces), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - optional refinement
            log.debug("voice split skipped at %.2fs: %s", start, exc)
            out.append(turn)
            continue
        labels = [_assign(row, centroids, label, margin) for row in matrix]
        if all(new == labels[0] for new in labels):
            out.append(turn)
            continue
        group_start = 0
        for i in range(1, len(pieces) + 1):
            if i < len(pieces) and labels[i] == labels[group_start]:
                continue
            p_start = pieces[group_start][0]
            p_end = pieces[i - 1][1]
            if group_start == 0:
                out.append((p_start, p_end, label))
            else:
                splits += 1
                # The centroid names *who* this piece is, so the piece joins that speaker:
                # "SPEAKER_08r2" = speaker 08, cut out of this turn. The "r2" suffix marks
                # the split for the cue layer and is stripped before the JSON is written.
                out.append((p_start, p_end, f"{labels[group_start]}r{group_start}"))
            group_start = i
    return out, splits
