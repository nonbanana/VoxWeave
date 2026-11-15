"""Converting a foreign Qwen3-ASR checkpoint into the layout the loader builds.

The rename map is the whole point here: a wrong or missing entry means transformers
random-initialises part of the model and transcribes nonsense, which is silent. These
tests pin every mapping and the tied ``lm_head`` that the export omits.
"""

from __future__ import annotations

import pytest

from voxweave.asr_convert import (
    EMBED_TOKENS,
    LM_HEAD,
    convert_state_dict,
    converted_dir,
    needs_conversion,
    rename_key,
)


@pytest.mark.parametrize(
    ("foreign", "expected"),
    [
        ("model.language_model.layers.0.mlp.up_proj.weight", "thinker.model.layers.0.mlp.up_proj.weight"),
        ("model.language_model.embed_tokens.weight", EMBED_TOKENS),
        (
            "model.multi_modal_projector.linear_1.weight",
            "thinker.audio_tower.proj1.weight",
        ),
        (
            "model.multi_modal_projector.linear_2.bias",
            "thinker.audio_tower.proj2.bias",
        ),
        ("model.audio_tower.conv2d1.weight", "thinker.audio_tower.conv2d1.weight"),
        ("thinker.model.norm.weight", "thinker.model.norm.weight"),
    ],
)
def test_rename_key(foreign: str, expected: str) -> None:
    assert rename_key(foreign) == expected


def test_the_longest_prefix_wins() -> None:
    """A generic 'model.' rule must not swallow the projector's two-way split."""
    key = "model.multi_modal_projector.linear_1.weight"
    assert rename_key(key) == "thinker.audio_tower.proj1.weight"
    assert "multi_modal_projector" not in rename_key(key)


def test_synthesises_the_tied_lm_head() -> None:
    state = {
        "model.language_model.embed_tokens.weight": "EMBED",
        "model.audio_tower.conv2d1.weight": "AUDIO",
    }
    out = convert_state_dict(state)
    assert out[LM_HEAD] == "EMBED"
    assert out[EMBED_TOKENS] == "EMBED"


def test_keeps_a_present_lm_head() -> None:
    state = {"model.lm_head.weight": "HEAD", "model.language_model.embed_tokens.weight": "EMBED"}
    out = convert_state_dict(state)
    assert out[LM_HEAD] == "HEAD"


def test_refuses_a_naming_scheme_it_does_not_know() -> None:
    """Half-converted output is the silent failure this module exists to prevent."""
    with pytest.raises(ValueError, match="unrecognised weight names"):
        convert_state_dict({"encoder.blocks.0.weight": "X"})


def test_needs_conversion_detects_both_layouts() -> None:
    assert needs_conversion(["model.audio_tower.conv2d1.weight"]) is True
    assert needs_conversion(["thinker.audio_tower.conv2d1.weight"]) is False
    assert needs_conversion([]) is False


def test_converted_dir_is_stable_and_filesystem_safe(tmp_path) -> None:
    a = converted_dir(tmp_path, "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
    b = converted_dir(tmp_path, "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf")
    assert a == b
    assert "/" not in a.name
    assert a.parent == tmp_path
