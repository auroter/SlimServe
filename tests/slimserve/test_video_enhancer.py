# SPDX-License-Identifier: Apache-2.0
"""The prompt enhancer's model-free pieces: the chat template against
transformers' rendering (tests/slimserve/data/ltx25_enhancer_template.json),
upstream's response cleanup, the no-repeat-n-gram ban, and the Gemma-4 E2B
layer wiring (K/V sharing, RoPE flavours, masks) on a synthetic config. The
generation itself is checked against transformers by
perf/ltx25_harness/n11_enhancer_parity.py."""

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video.ltx25 import enhancer  # noqa: E402

FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "ltx25_enhancer_template.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["user"][:12])
def test_chat_text_matches_transformers_rendering(case):
    assert "<bos>" + enhancer.chat_text(case["system"], case["user"]) == case["text"]


def test_system_prompts_are_upstreams():
    t2v = enhancer.system_prompt("t2v")
    assert t2v.startswith("You are given a user's short text-to-video request.")
    assert "audio" in t2v.lower()
    assert enhancer.system_prompt("i2v") != t2v


def test_clean_response_matches_upstream():
    assert enhancer.clean_response("“A man’s dog” — runs") == ("A man's dog\" - runs")
    assert enhancer.clean_response("...\n\n*** Caption: ok") == "Caption: ok"
    assert enhancer.clean_response("123 456") == "123 456"  # no letters: unchanged


def test_banned_tokens_complete_a_seen_ngram_only():
    seq = [1, 2, 3, 4, 5, 9, 2, 3, 4]
    assert enhancer.banned_tokens(seq, 4) == {5}  # (2 3 4) was followed by 5
    assert enhancer.banned_tokens(seq, 5) == set()  # (9 2 3 4) never occurred before
    assert enhancer.banned_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4], 5) == {5}
    assert enhancer.banned_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert enhancer.banned_tokens([1, 2], 5) == set()
    assert enhancer.banned_tokens([1, 2, 3, 4, 5, 6], 0) == set()


def _toy() -> enhancer.PromptEnhancer:
    """A PromptEnhancer with the E2B config and no weights."""
    e = enhancer.PromptEnhancer()
    kinds = (["sliding_attention"] * 4 + ["full_attention"]) * 7
    e.cfg = {
        "layer_types": kinds,
        "num_kv_shared_layers": 20,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 512,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    }
    return e


def test_kv_sharing_maps_the_tail_onto_the_last_unshared_layer_of_its_kind():
    e = _toy()
    assert [e._kv_source(i) for i in range(15)] == list(range(15))
    tail = {e._kv_source(i) for i in range(15, 35)}
    assert tail == {13, 14}  # last sliding (13) and last global (14) before the tail
    assert e._kv_source(19) == 14 and e._kv_source(20) == 13


def test_rotary_flavours():
    e = _toy()
    pos = mx.arange(8).astype(mx.float32)
    hd, cos, sin = e._rotary(True, pos)
    assert hd == 256 and cos.shape == (1, 1, 8, 128)
    hd, cos, sin = e._rotary(False, pos)
    assert hd == 512 and cos.shape == (1, 1, 8, 256)
    # proportional RoPE: only the first quarter of the head (64 angles) rotates
    c = np.array(cos)[0, 0]
    assert np.allclose(c[:, 64:], 1.0) and not np.allclose(c[1, :64], 1.0)


def test_masks_are_causal_and_windowed():
    e = _toy()
    e.cfg["sliding_window"] = 3
    m = np.array(e._mask(mx.arange(5).astype(mx.float32), 5, True))
    visible = np.isfinite(m)
    assert visible.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [False, True, True, True, False],
        [False, False, True, True, True],
    ]
    # one decode step at position 7 over 8 cached keys
    step = np.isfinite(np.array(e._mask(mx.array([7.0]), 8, True)))
    assert step.tolist() == [[False] * 5 + [True] * 3]
    assert e._mask(mx.array([7.0]), 8, False) is None
    full = np.isfinite(np.array(e._mask(mx.arange(3).astype(mx.float32), 3, False)))
    assert full.tolist() == [[True, False, False], [True, True, False], [True] * 3]


def test_checkpoint_registry_names_the_enhancer_next_to_its_configs():
    from slimserve.video.ltx25 import checkpoints

    p = checkpoints.path_of("enhancer", Path("/root"))
    assert p == Path("/root/prompt_enhancer/gemma-4-E2B-it/model.safetensors")
