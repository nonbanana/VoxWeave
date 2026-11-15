"""Guard the Qwen3-ASR load path against configs and checkpoints it cannot actually use.

Two failure modes around ``qwen_asr``'s loader are worth handling before it burns a
minute of model loading, and both were found on real Hub checkpoints:

1. **A config without the RoPE layout.** ``qwen_asr``'s rotary init calls
   ``config.rope_scaling.get(...)`` with no None guard, so a checkpoint whose text config
   omits ``rope_scaling`` dies with ``AttributeError: 'NoneType' object has no attribute
   'get'`` before a single frame of audio is read::

       jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf       -> text_config has no rope_scaling

   The value is not a free parameter: every Qwen3-ASR checkpoint shares the Qwen3
   multimodal-RoPE layout (three sections, 24/20/20), so a fine-tune of that base needs
   exactly the base's value. :func:`repair_qwen_asr_config` fills it on the *loaded*
   config object; nothing is written to the Hugging Face cache.

2. **Weights stored under someone else's naming scheme.** Checkpoints exported for a
   different runtime rename the towers (``model.`` instead of ``thinker.``,
   ``language_model.`` for ``model.``, ``multi_modal_projector`` for ``audio_tower.proj1/2``)
   and drop the tied ``lm_head``. ``transformers`` does not error on that: it reports the
   keys as unexpected, randomly initialises the rest, prints "You should probably TRAIN
   this model", and then transcribes confident nonsense. Repairs cannot fix it — the
   weights need converting — so :func:`foreign_weight_layout` is checked first and the run
   stops with a message naming the checkpoint instead of producing garbage.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger("voxweave")

# The Qwen3 multimodal-RoPE layout every Qwen3-ASR checkpoint uses (see
# Qwen/Qwen3-ASR-1.7B's config.json). Copied on use, never shared: ``mrope_section`` is a
# list, and two repaired models must not alias one layout.
QWEN3_MROPE = {
    "interleaved": True,
    "mrope_interleaved": True,
    "mrope_section": [24, 20, 20],
    "rope_type": "default",
    "type": "default",
}

# Model types whose text tower is the Qwen3 decoder with mrope. Anything else is left
# alone: an unknown architecture must not receive a RoPE layout it was not trained with.
_QWEN3_ASR_MODEL_TYPES = frozenset({"qwen3_asr"})

# Top-level prefix every loadable Qwen3-ASR checkpoint uses. The model class builds
# ``self.thinker = ...``, so the state dict must agree.
EXPECTED_WEIGHT_PREFIX = "thinker"


def _text_config(cfg: Any) -> Any | None:
    """The text tower's config, whichever nesting the publisher used.

    The Transformers-shaped checkpoints nest it (``thinker_config.text_config``); some
    flat exports keep it at the top level. Only the nested one is read: the model builds
    from ``thinker_config``, so repairing a top-level copy would change nothing.
    """
    thinker = getattr(cfg, "thinker_config", None)
    return getattr(thinker, "text_config", None) if thinker is not None else None


def repair_qwen_asr_config(cfg: Any) -> list[str]:
    """Fill the fields ``qwen_asr`` requires from ``cfg`` in place.

    Returns the dotted names that were repaired (empty when the config was already
    complete), so the caller can report exactly what it changed instead of a vague
    warning.
    """
    if getattr(cfg, "model_type", None) not in _QWEN3_ASR_MODEL_TYPES:
        return []
    text_cfg = _text_config(cfg)
    if text_cfg is None or getattr(text_cfg, "rope_scaling", None):
        return []
    # deep, not shallow: ``mrope_section`` is a list, and two repaired models must never
    # alias one layout (a caller mutating it would poison the next load).
    text_cfg.rope_scaling = copy.deepcopy(QWEN3_MROPE)
    return ["thinker_config.text_config.rope_scaling"]


def repaired_qwen_asr_config(local_path: Any) -> Any | None:
    """Load ``local_path``'s config and repair it, or return None when nothing to do.

    None means "load the model the ordinary way": either the config is already complete
    or it could not be inspected, and in both cases handing a config object to
    ``from_pretrained`` would change behaviour we have no reason to change.
    """
    try:
        import qwen_asr  # noqa: F401  registers the qwen3_asr config/model classes
        from transformers import AutoConfig
    except ImportError:  # pragma: no cover - transformers ships with every ASR extra
        return None
    try:
        cfg = AutoConfig.from_pretrained(str(local_path), local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - a custom architecture must not fail the run
        log.debug("could not inspect %s for a config repair: %s", local_path, exc)
        return None
    repaired = repair_qwen_asr_config(cfg)
    if not repaired:
        return None
    log.warning(
        "%s ships a config without %s; applying the Qwen3-ASR default RoPE layout",
        getattr(cfg, "_name_or_path", "") or str(local_path),
        ", ".join(repaired),
    )
    return cfg


def _safetensors_keys(shard: Path) -> list[str]:
    """Tensor names in one ``.safetensors`` file, from its header only (no tensor read)."""
    with shard.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return []
        import struct

        (size,) = struct.unpack("<Q", raw)
        if size <= 0 or size > 100_000_000:  # implausible header: refuse to read
            return []
        header = json.loads(fh.read(size))
    return [key for key in header if key != "__metadata__"]


def weight_keys(snapshot: Path | str) -> list[str]:
    """Every tensor name in a model directory, from the index when one exists."""
    snapshot = Path(snapshot)
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, KeyError, ValueError):
            return []
        return list(weight_map)
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        return []
    keys: list[str] = []
    for shard in shards:
        keys.extend(_safetensors_keys(shard))
    return keys


def foreign_weight_layout(snapshot: Path | str) -> str | None:
    """The dominant top-level weight prefix, when it is not the one the loader expects.

    Returns None for a checkpoint that loads normally (including one whose weights cannot
    be inspected — absence of evidence must not block a run).
    """
    keys = weight_keys(snapshot)
    if not keys:
        return None
    prefixes = Counter(key.split(".", 1)[0] for key in keys)
    prefix, _count = prefixes.most_common(1)[0]
    if prefix == EXPECTED_WEIGHT_PREFIX:
        return None
    return prefix
