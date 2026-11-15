"""Convert a Qwen3-ASR checkpoint into the layout ``qwen_asr``'s loader expects.

Several published fine-tunes are exported by a runtime other than Transformers, so their
``model.safetensors`` uses that runtime's names::

    on disk                        what the loader builds
    -----------------------------  -----------------------------------
    model.language_model.layers.*  thinker.model.layers.*
    model.multi_modal_projector.linear_1.*   thinker.audio_tower.proj1.*
    model.multi_modal_projector.linear_2.*   thinker.audio_tower.proj2.*
    (no lm_head)                   thinker.lm_head.weight

``transformers`` does not fail on that. It reports every key as unexpected, random-
initialises the rest, prints "You should probably TRAIN this model" and then transcribes
confident nonsense, which is why the caller refuses such a checkpoint unless it has been
converted first (see :mod:`voxweave.asr_config`).

The conversion is a rename plus one synthesised tensor: these fine-tunes tie the output
layer to the input embeddings (``tie_word_embeddings``), so ``lm_head`` is the embedding
matrix itself. Text, tokenizer and processor files are copied unchanged, so nothing else
about the model changes.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

log = logging.getLogger("voxweave")

# Longest prefixes first: ``model.multi_modal_projector.linear_1.`` must win over the
# generic ``model.`` rule below it.
RENAMES: tuple[tuple[str, str], ...] = (
    ("model.language_model.", "thinker.model."),
    ("model.multi_modal_projector.linear_1.", "thinker.audio_tower.proj1."),
    ("model.multi_modal_projector.linear_2.", "thinker.audio_tower.proj2."),
)
# Everything else under the foreign prefix is a straight ``model.`` -> ``thinker.``.
FOREIGN_PREFIX = "model."
TARGET_PREFIX = "thinker."

LM_HEAD = "thinker.lm_head.weight"
EMBED_TOKENS = "thinker.model.embed_tokens.weight"

# Files the convert must not copy: the weights are rewritten, the index describes the old
# shards, and the config is repaired on the way out.
_SKIP_SUFFIXES = (".safetensors",)
_SKIP_NAMES = frozenset({"model.safetensors.index.json"})


def rename_key(key: str) -> str:
    """One weight name in the loader's layout. Unknown prefixes are returned unchanged."""
    for old, new in RENAMES:
        if key.startswith(old):
            return new + key[len(old) :]
    if key.startswith(FOREIGN_PREFIX):
        return TARGET_PREFIX + key[len(FOREIGN_PREFIX) :]
    return key


def convert_state_dict(state: Mapping[str, object]) -> dict[str, object]:
    """Rename every tensor and restore the tied ``lm_head``.

    Raises ``ValueError`` when a name still does not start with ``thinker.``: shipping a
    half-renamed checkpoint would load as random weights again, which is the failure this
    module exists to prevent.
    """
    out = {rename_key(key): value for key, value in state.items()}
    if LM_HEAD not in out:
        embed = out.get(EMBED_TOKENS)
        if embed is not None:
            # tie_word_embeddings: the output projection IS the embedding matrix. Clone it
            # because safetensors refuses to save two names that share storage -- and the
            # copy is only made once, at conversion time.
            out[LM_HEAD] = embed.clone() if hasattr(embed, "clone") else embed
    stray = sorted(key for key in out if not key.startswith(TARGET_PREFIX))
    if stray:
        raise ValueError(
            "cannot convert this checkpoint: unrecognised weight names "
            f"(e.g. {stray[0]!r}); the export uses a naming scheme this converter"
            " does not know"
        )
    return out


def _copy_auxiliary_files(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        if item.is_dir() or item.name in _SKIP_NAMES or item.suffix in _SKIP_SUFFIXES:
            continue
        shutil.copy2(item, dst / item.name)


def _ensure_preprocessor_config(dst: Path) -> None:
    """Give AutoProcessor the file name it actually looks for.

    Some exports ship the audio front end as ``processor_config.json`` with the feature
    extractor nested under ``feature_extractor``. AutoProcessor only reads
    ``preprocessor_config.json`` (flat), and without it silently builds a bare tokenizer,
    which then fails later with ``Qwen2TokenizerFast has no attribute audio_token``.
    """
    target = dst / "preprocessor_config.json"
    source = dst / "processor_config.json"
    if target.is_file() or not source.is_file():
        return
    data = json.loads(source.read_text(encoding="utf-8"))
    nested = data.pop("feature_extractor", None)
    merged = {**nested, **data} if isinstance(nested, dict) else data
    target.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


# Token ids the config names, and the tokenizer attribute the processor reads them by.
# Qwen3-ASR's processor does ``self.audio_token = self.tokenizer.audio_token``, and that
# attribute only exists when ``extra_special_tokens`` names it.
_SPECIAL_ID_FIELDS = (
    ("audio_token", "audio_token_id"),
    ("audio_bos_token", "audio_bos_token_id"),
    ("audio_eos_token", "audio_eos_token_id"),
    ("image_token", "image_token_id"),
    ("video_token", "video_token_id"),
)
# Fallback for the ids a config does not name: the Qwen3 special tokens are fixed strings,
# and a checkpoint either carries them in its vocabulary or does not need them.
_SPECIAL_BY_CONTENT = {
    "audio_token": "<|audio_pad|>",
    "audio_bos_token": "<|audio_start|>",
    "audio_eos_token": "<|audio_end|>",
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_bos_token": "<|vision_start|>",
    "vision_eos_token": "<|vision_end|>",
}


def _ensure_extra_special_tokens(dst: Path) -> None:
    """Rebuild ``extra_special_tokens`` from the config's ids and the tokenizer's vocabulary.

    A checkpoint exported by another runtime keeps ``tokenizer.json`` (so the pad/bos
    tokens exist) but omits the ``extra_special_tokens`` mapping transformers uses to set
    attributes such as ``tokenizer.audio_token``. Without it the processor constructor dies
    with ``Qwen2TokenizerFast has no attribute audio_token`` after the weights already
    loaded, which looks like a model bug and is not one.
    """
    tc_path, cfg_path, tj_path = (
        dst / "tokenizer_config.json",
        dst / "config.json",
        dst / "tokenizer.json",
    )
    if not (tc_path.is_file() and cfg_path.is_file() and tj_path.is_file()):
        return
    tc = json.loads(tc_path.read_text(encoding="utf-8"))
    if tc.get("extra_special_tokens"):
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    added = {
        token["id"]: token["content"]
        for token in json.loads(tj_path.read_text(encoding="utf-8")).get("added_tokens", [])
    }
    known = set(added.values())
    extra = {
        name: added[tid]
        for name, field in _SPECIAL_ID_FIELDS
        if isinstance(tid := cfg.get(field), int) and tid in added
    }
    for name, content in _SPECIAL_BY_CONTENT.items():
        if name not in extra and content in known:
            extra[name] = content
    if not extra:
        return
    tc["extra_special_tokens"] = extra
    tc_path.write_text(json.dumps(tc, indent=2) + "\n", encoding="utf-8")


def _read_shards(src: Path) -> dict[str, object]:
    from safetensors.torch import load_file

    index = src / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = [p.name for p in sorted(src.glob("*.safetensors"))]
    state: dict[str, object] = {}
    for shard in shards:
        log.info("convert: reading %s", shard)
        state.update(load_file(str(src / shard)))
    return state


def convert_snapshot(src: Path | str, dst: Path | str) -> Path:
    """Write a loadable copy of ``src`` into ``dst`` (created, replaced if present).

    Returns ``dst``. The weights land in one ``model.safetensors``; a sharded export is
    merged on the way through, which is fine because the result is a local cache, not a
    published model.
    """
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    from voxweave.asr_config import nest_flat_thinker, repair_qwen_asr_config

    state = convert_state_dict(_read_shards(src))
    dst.mkdir(parents=True, exist_ok=True)
    _copy_auxiliary_files(src, dst)
    _ensure_preprocessor_config(dst)
    _ensure_extra_special_tokens(dst)

    # The config must describe the nested thinker the weights now match, or the repaired
    # architecture and the tensor shapes disagree.
    config_path = src / "config.json"
    if config_path.is_file():
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(src), local_files_only=True)
        nest_flat_thinker(cfg)
        repair_qwen_asr_config(cfg)
        cfg.save_pretrained(str(dst))

    log.info("convert: writing %d tensors", len(state))
    save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})
    # Only a finished conversion may be reused: a half-written directory would load as a
    # broken checkpoint that looks cached.
    (dst / ".complete").write_text("ok\n", encoding="utf-8")
    return dst


def converted_dir(root: Path | str, model_id: str) -> Path:
    """Where a converted copy of ``model_id`` lives under ``root``."""
    return Path(root) / ("converted--" + model_id.replace("/", "--"))


def needs_conversion(keys: Iterable[str]) -> bool:
    """True when the checkpoint's weights are not already in the loader's layout."""
    for key in keys:
        if not key.startswith(TARGET_PREFIX):
            return True
    return False
