"""Guards for Qwen3-ASR checkpoints whose config or weight layout the loader cannot use.

Both cases come from real Hub checkpoints, not from synthetic tests:

* a text config without ``rope_scaling``, which ``qwen_asr`` dereferences unguarded;
* weights renamed for another runtime, which transformers silently random-initialises.
"""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest

from voxweave.asr_config import (
    EXPECTED_WEIGHT_PREFIX,
    QWEN3_MROPE,
    foreign_weight_layout,
    repair_qwen_asr_config,
    repaired_qwen_asr_config,
    weight_keys,
)


def _cfg(*, rope: dict | None, model_type: str = "qwen3_asr"):
    text = SimpleNamespace(hidden_size=2048, num_hidden_layers=28)
    if rope is not None:
        text.rope_scaling = rope
    return SimpleNamespace(
        model_type=model_type, thinker_config=SimpleNamespace(text_config=text)
    )


def test_fills_a_missing_rope_scaling() -> None:
    cfg = _cfg(rope=None)
    assert repair_qwen_asr_config(cfg) == ["thinker_config.text_config.rope_scaling"]
    assert cfg.thinker_config.text_config.rope_scaling == QWEN3_MROPE


def test_leaves_a_complete_config_alone() -> None:
    original = {"mrope_section": [16, 24, 24], "type": "default"}
    cfg = _cfg(rope=dict(original))
    assert repair_qwen_asr_config(cfg) == []
    assert cfg.thinker_config.text_config.rope_scaling == original


def test_never_repairs_another_architecture() -> None:
    cfg = _cfg(rope=None, model_type="whisper")
    assert repair_qwen_asr_config(cfg) == []
    assert not hasattr(cfg.thinker_config.text_config, "rope_scaling")


def test_repaired_default_is_not_shared() -> None:
    """Two repaired models must not alias one layout: mutating one would poison the
    next load."""
    first, second = _cfg(rope=None), _cfg(rope=None)
    repair_qwen_asr_config(first)
    repair_qwen_asr_config(second)
    first.thinker_config.text_config.rope_scaling["mrope_section"].append(99)
    assert QWEN3_MROPE["mrope_section"] == [24, 20, 20]
    assert second.thinker_config.text_config.rope_scaling["mrope_section"] == [24, 20, 20]


def test_a_directory_without_a_config_falls_through(tmp_path) -> None:
    assert repaired_qwen_asr_config(tmp_path) is None


def _write_shard(path, keys: list[str]) -> None:
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_accepts_the_expected_layout(tmp_path) -> None:
    _write_shard(tmp_path / "model.safetensors", ["thinker.model.norm.weight"])
    assert weight_keys(tmp_path) == ["thinker.model.norm.weight"]
    assert foreign_weight_layout(tmp_path) is None


def test_flags_weights_renamed_for_another_runtime(tmp_path) -> None:
    """The shape of jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf: 'model.' instead of
    'thinker.', which transformers would happily random-initialise."""
    _write_shard(
        tmp_path / "model.safetensors",
        ["model.audio_tower.conv2d1.weight", "model.language_model.layers.0.mlp.up_proj.weight"],
    )
    assert foreign_weight_layout(tmp_path) == "model"


def test_reads_the_index_when_a_sharded_checkpoint_has_one(tmp_path) -> None:
    index = {
        "weight_map": {
            "thinker.model.norm.weight": "a.safetensors",
            "thinker.audio_tower.conv2d1.weight": "b.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    assert len(weight_keys(tmp_path)) == 2
    assert foreign_weight_layout(tmp_path) is None


def test_uninspectable_weights_do_not_block_a_run(tmp_path) -> None:
    assert weight_keys(tmp_path) == []
    assert foreign_weight_layout(tmp_path) is None


@pytest.mark.parametrize("prefix", ["model", "language_model"])
def test_every_foreign_prefix_is_reported(tmp_path, prefix: str) -> None:
    _write_shard(tmp_path / "model.safetensors", [f"{prefix}.layers.0.mlp.up_proj.weight"])
    assert foreign_weight_layout(tmp_path) == prefix
    assert EXPECTED_WEIGHT_PREFIX == "thinker"
