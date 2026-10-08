# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 prompt enhancer: Gemma-4 E2B-it rewrites a request into the model's
caption style.

Upstream (ltx_core text_encoders/gemma/encoders/base_encoder.py `enhance_t2v`,
ltx_pipelines utils/helpers.py `generate_enhanced_prompt`): the 2.5 text
encoder is a fine-tune that cannot generate, so `--prompt-enhancer-gemma-root`
names a separate generative instruct Gemma (the README's choice is
google/gemma-4-E2B-it). The system prompt (prompts/gemma4_t2v_system_prompt.txt,
upstream's file verbatim) and `user prompt: <request>` go through Gemma's chat
template; decoding is greedy with no repeated 5-gram and at most 600 new tokens
(GEMMA4_ENHANCE_GENERATION_KWARGS); the answer is cleaned of curly quotes and
leading non-letters (`clean_response`). Greedy decoding is deterministic:
upstream's `seed` has no effect on it.

The model (transformers modeling_gemma4.py, text path): 35 decoder layers,
hidden 1536, 8 query heads over 1 K/V head, head dim 256 on sliding-window
(512) layers and 512 on every fifth, global, layer whose RoPE rotates only the
first quarter of the head ("proportional"). The last 20 layers carry no K/V
projections and reuse the K/V of the last non-shared layer of their kind
(`num_kv_shared_layers`), and their MLP is twice as wide. Every layer adds a
per-layer input: the token's own 256-dim per-layer embedding plus a projection
of the token embedding, gated by the stream (`hidden_size_per_layer_input`).
The LM head is the tied embedding with a tanh soft cap of 30.

Only the language model is loaded; the vision and audio towers in the file
are not on the text-to-video path. Upstream's I2V variant shows Gemma the
conditioning still through its vision tower, which is not ported: an
image-to-video request is enhanced from its text with the T2V system prompt.

Precision follows the text encoder: GEMM operands fp16, stream and norms fp32
(upstream bf16 throughout). Checked token for token against transformers on
the CPU by perf/ltx25_harness/n11_enhancer_parity.py.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints
from slimserve.video.ltx25.text import _gelu, _rms

MAX_NEW_TOKENS = 600
NO_REPEAT_NGRAM = 5
_PREFIX = "model.language_model."
_PROMPTS = Path(__file__).parent / "prompts"
# ltx_pipelines.utils.helpers._UNICODE_REPLACEMENTS
_REPLACEMENTS = str.maketrans("‘’“”—– ′−", "''\"\"-- '-")


def system_prompt(kind: str = "t2v") -> str:
    return (_PROMPTS / f"gemma4_{kind}_system_prompt.txt").read_text()


def chat_text(system: str, user: str) -> str:
    """Gemma-4's chat template (chat_template.jinja) for a system turn, a user
    turn and an open model turn, as transformers renders it with
    add_generation_prompt=True (both contents `| trim`med); the leading <bos>
    is added by `tokenize`."""
    return (
        f"<|turn>system\n{system.strip()}<turn|>\n"
        f"<|turn>user\n{user.strip()}<turn|>\n<|turn>model\n"
    )


def clean_response(text: str) -> str:
    """Upstream clean_response: ASCII quotes and dashes, drop a leading run of
    non-letters."""
    text = text.translate(_REPLACEMENTS)
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[i:]
    return text


def banned_tokens(seq: list[int], n: int) -> set[int]:
    """Tokens that would complete an n-gram already present in `seq`
    (transformers NoRepeatNGramLogitsProcessor over prompt + generation)."""
    if n <= 0 or len(seq) < n:
        return set()
    k = n - 1
    prefix = tuple(seq[-k:])
    return {seq[j + k] for j in range(len(seq) - k) if tuple(seq[j : j + k]) == prefix}


class _FileRows:
    """Rows of a 2-D bf16 tensor read straight from a safetensors file."""

    def __init__(self, path: Path, name: str):
        header = checkpoints.read_header(path)
        info = header.tensors[name]
        if info["dtype"] != "BF16" or len(info["shape"]) != 2:
            raise ValueError(f"{name}: expected a 2-D BF16 tensor, got {info}")
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
        start, _end = info["data_offsets"]
        self.rows, self.cols = info["shape"]
        self.mm = np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
            offset=8 + n + start,
            shape=(self.rows, self.cols),
        )

    def __call__(self, ids: list[int]) -> mx.array:
        """(len(ids), cols) fp32: bf16 bits widened exactly."""
        bits = np.asarray(self.mm[np.asarray(ids)], dtype=np.uint32) << 16
        return mx.array(bits.view(np.float32))


class PromptEnhancer:
    def __init__(
        self,
        root: Path | None = None,
        operand: mx.Dtype = mx.float16,
        stream: mx.Dtype = mx.float32,
    ):
        self.root = root
        self.O, self.S = operand, stream
        self.w: dict[str, mx.array] | None = None
        self.per_layer_rows: _FileRows | None = None
        self.cfg: dict | None = None
        self.gen: dict | None = None
        self.tokenizer = None

    # ---- loading ----------------------------------------------------------
    def load(self) -> PromptEnhancer:
        if self.w is not None:
            return self
        from tokenizers import Tokenizer

        path = checkpoints.path_of("enhancer", self.root)
        self.cfg = json.loads((path.parent / "config.json").read_text())["text_config"]
        self.gen = json.loads((path.parent / "generation_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(path.parent / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        t = self.cfg
        if (
            t["enable_moe_block"]
            or t["attention_k_eq_v"]
            or t["use_bidirectional_attention"]
        ):
            raise ValueError("unsupported Gemma-4 enhancer config")
        raw = checkpoints.load_raw(path)
        keep = {
            k[len(_PREFIX) :]: raw.pop(k) for k in list(raw) if k.startswith(_PREFIX)
        }
        del raw
        if "embed_tokens_per_layer.weight" not in keep:
            raise ValueError(f"{path}: no dense Gemma-4 language model")
        # The shared-K/V tail ships k/v projections and norms the model never
        # runs (transformers drops them on load too).
        for i in range(len(t["layer_types"])):
            if self._kv_source(i) != i:
                for name in ("k_proj", "v_proj", "k_norm", "v_norm"):
                    keep.pop(f"layers.{i}.self_attn.{name}.weight", None)
        # The per-layer embedding table (262144 x 8960 bf16, 4.7 GiB) is only
        # ever gathered, about a thousand rows per prompt: its rows are read
        # from the file on use and the table never enters memory. The token
        # table is the LM head too, so it is resident (bf16, 0.8 GiB).
        keep.pop("embed_tokens_per_layer.weight")
        self.per_layer_rows = _FileRows(path, _PREFIX + "embed_tokens_per_layer.weight")
        table = keep.pop("embed_tokens.weight")
        out = checkpoints.cast_operands(keep, self.O)
        for k in out:
            if out[k].ndim == 1 and k.endswith(("norm.weight", "layer_scalar")):
                out[k] = out[k].astype(mx.float32)
        out["embed_tokens.weight"] = table
        self.w = out
        mx.eval(self.w)
        return self

    def unload(self) -> None:
        self.w = None
        self.per_layer_rows = None
        mx.clear_cache()

    def _lin(self, name: str, x: mx.array) -> mx.array:
        return x @ self.w[name + ".weight"].T

    # ---- model ------------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        """<bos> + text: the Gemma-4 tokenizer's post-processor adds nothing."""
        self.load()
        bos = self.tokenizer.token_to_id("<bos>")
        ids = self.tokenizer.encode(text).ids
        return ids if ids[:1] == [bos] else [bos, *ids]

    def _rotary(self, sliding: bool, pos: mx.array):
        """(head_dim, cos, sin) at fp32 positions `pos`; cos/sin (1, 1, n, hd/2)
        in the operand dtype. Global layers rotate partial_rotary_factor of the
        head and leave the rest (zero frequency)."""
        t = self.cfg
        hd = t["head_dim"] if sliding else t["global_head_dim"]
        rp = t["rope_parameters"]["sliding_attention" if sliding else "full_attention"]
        half = hd // 2
        angles = half if sliding else int(rp["partial_rotary_factor"] * hd // 2)
        inv = 1.0 / mx.power(
            mx.array(rp["rope_theta"], dtype=mx.float32),
            mx.arange(0, 2 * angles, 2, dtype=mx.float32) / hd,
        )
        if angles < half:
            inv = mx.concatenate([inv, mx.zeros((half - angles,), dtype=mx.float32)])
        f = pos[:, None] * inv[None, :]
        return (
            hd,
            mx.cos(f)[None, None].astype(self.O),
            mx.sin(f)[None, None].astype(self.O),
        )

    @staticmethod
    def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    def _mask(self, q_pos: mx.array, n_keys: int, sliding: bool) -> mx.array | None:
        """Additive mask (q, n_keys) for queries at absolute positions `q_pos`
        over keys at positions 0..n_keys-1: causal, and within the sliding
        window (kv > q - window) on sliding layers. None when nothing is
        hidden."""
        if q_pos.shape[0] == 1 and not sliding:
            return None  # one query at the end of a global layer sees every key
        k_idx = mx.arange(n_keys)[None, :]
        q_idx = q_pos[:, None]
        visible = k_idx <= q_idx
        if sliding:
            visible = mx.logical_and(
                visible, k_idx > q_idx - self.cfg["sliding_window"]
            )
        return mx.where(
            visible, mx.array(0.0, dtype=self.O), mx.array(-mx.inf, dtype=self.O)
        )

    def _kv_source(self, i: int) -> int:
        """The layer whose K/V layer `i` uses: itself, or for the shared tail
        the last non-shared layer of the same kind."""
        t = self.cfg
        kinds = t["layer_types"]
        first_shared = len(kinds) - t["num_kv_shared_layers"]
        if i < first_shared:
            return i
        return max(j for j in range(first_shared) if kinds[j] == kinds[i])

    def _layer(
        self,
        i: int,
        h: mx.array,
        pli: mx.array,
        flavors: dict,
        q_pos: mx.array,
        cache: list,
    ) -> mx.array:
        """One decoder layer on the stream `h` (1, n, D) with per-layer input
        `pli` (1, n, P); extends (or, for a shared layer, reads) `cache`."""
        w, t, op, S = self.w, self.cfg, self.O, self.S
        eps, heads = t["rms_norm_eps"], t["num_attention_heads"]
        sliding = t["layer_types"][i] == "sliding_attention"
        p = f"layers.{i}"
        n = h.shape[1]
        hd, cos, sin = flavors[sliding]
        x = _rms(h, w[p + ".input_layernorm.weight"], eps).astype(op)
        q = self._lin(p + ".self_attn.q_proj", x).reshape(1, n, heads, hd)
        q = self._rope(
            _rms(q, w[p + ".self_attn.q_norm.weight"], eps)
            .astype(op)
            .transpose(0, 2, 1, 3),
            cos,
            sin,
        )
        src = self._kv_source(i)
        if src == i:
            k = self._lin(p + ".self_attn.k_proj", x)
            kv_heads = k.shape[-1] // hd
            k = k.reshape(1, n, kv_heads, hd)
            v = self._lin(p + ".self_attn.v_proj", x).reshape(1, n, kv_heads, hd)
            k = self._rope(
                _rms(k, w[p + ".self_attn.k_norm.weight"], eps)
                .astype(op)
                .transpose(0, 2, 1, 3),
                cos,
                sin,
            )
            v = _rms(v, None, eps).astype(op).transpose(0, 2, 1, 3)
            if cache[i] is not None:
                k = mx.concatenate([cache[i][0], k], axis=2)
                v = mx.concatenate([cache[i][1], v], axis=2)
            cache[i] = (k, v)
        else:
            k, v = cache[src]
        length = k.shape[2]
        kv_heads = k.shape[1]
        if kv_heads != heads:
            rep = heads // kv_heads
            k = mx.broadcast_to(k[:, :, None], (1, kv_heads, rep, length, hd)).reshape(
                1, heads, length, hd
            )
            v = mx.broadcast_to(v[:, :, None], (1, kv_heads, rep, length, hd)).reshape(
                1, heads, length, hd
            )
        mask = self._mask(q_pos, length, sliding)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        y = self._lin(
            p + ".self_attn.o_proj", y.transpose(0, 2, 1, 3).reshape(1, n, heads * hd)
        )
        h = h + _rms(y, w[p + ".post_attention_layernorm.weight"], eps).astype(S)
        x = _rms(h, w[p + ".pre_feedforward_layernorm.weight"], eps).astype(op)
        y = _gelu(self._lin(p + ".mlp.gate_proj", x)) * self._lin(p + ".mlp.up_proj", x)
        y = self._lin(p + ".mlp.down_proj", y)
        h = h + _rms(y, w[p + ".post_feedforward_layernorm.weight"], eps).astype(S)
        # per-layer input: gate the stream, scale by the token's per-layer
        # embedding, project back
        g = _gelu(self._lin(p + ".per_layer_input_gate", h.astype(op))) * pli.astype(op)
        y = self._lin(p + ".per_layer_projection", g)
        h = h + _rms(y, w[p + ".post_per_layer_input_norm.weight"], eps).astype(S)
        scalar = w.get(p + ".layer_scalar")
        return h if scalar is None else h * scalar.astype(S)

    def _embed(self, ids: list[int]) -> tuple[mx.array, mx.array]:
        """(token embeddings (1, n, D) stream dtype, per-layer inputs
        (1, n, L, P) stream dtype). Upstream keeps both embed scales in the
        model dtype, bf16."""
        w, t, S, op = self.w, self.cfg, self.S, self.O
        idx = mx.array(ids)
        layers, per = len(t["layer_types"]), t["hidden_size_per_layer_input"]
        scale = mx.array(math.sqrt(t["hidden_size"])).astype(mx.bfloat16).astype(S)
        h = w["embed_tokens.weight"][idx][None].astype(S) * scale
        pscale = mx.array(math.sqrt(per)).astype(mx.bfloat16).astype(S)
        ple = self.per_layer_rows(ids)[None].astype(S) * pscale
        ple = ple.reshape(1, -1, layers, per)
        proj = self._lin("per_layer_model_projection", h.astype(op)).astype(S)
        proj = proj * (t["hidden_size"] ** -0.5)
        proj = _rms(
            proj.reshape(1, -1, layers, per),
            w["per_layer_projection_norm.weight"],
            t["rms_norm_eps"],
        )
        return h, (proj + ple) * (2.0**-0.5)

    def _forward(self, ids: list[int], start: int, cache: list) -> mx.array:
        """Run tokens `ids` at positions start.. through the tower, extending
        `cache`; returns the final-normed stream (1, n, D)."""
        w, t = self.w, self.cfg
        n = len(ids)
        pos = mx.arange(start, start + n).astype(mx.float32)
        flavors = {True: self._rotary(True, pos), False: self._rotary(False, pos)}
        h, pli = self._embed(ids)
        for i in range(len(t["layer_types"])):
            h = self._layer(i, h, pli[:, :, i], flavors, pos, cache)
            if n > 1:
                mx.eval(h)
        return _rms(h, w["norm.weight"], t["rms_norm_eps"])

    def _logits(self, h_last: mx.array) -> mx.array:
        """Tied-embedding head on the final-normed last position with the
        final logit soft cap; fp32 (vocab,)."""
        t = self.cfg
        x = h_last[0, -1:].astype(self.O)
        logits = (x @ self.w["embed_tokens.weight"].astype(self.O).T).astype(
            mx.float32
        )[0]
        cap = t.get("final_logit_softcapping")
        return cap * mx.tanh(logits / cap) if cap else logits

    def generate(
        self,
        ids: list[int],
        max_new_tokens: int = MAX_NEW_TOKENS,
        no_repeat_ngram_size: int = NO_REPEAT_NGRAM,
        eos: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Greedy decoding with a K/V cache (transformers generate with
        do_sample=False and no_repeat_ngram_size). Returns the new ids without
        the stop token."""
        self.load()
        if eos is None:
            e = self.gen.get("eos_token_id", self.cfg["eos_token_id"])
            eos = tuple(e) if isinstance(e, list) else (e,)
        cache: list = [None] * len(self.cfg["layer_types"])
        seq = list(ids)
        logits = self._logits(self._forward(seq, 0, cache))
        out: list[int] = []
        for _ in range(max_new_tokens):
            mx.eval(logits)
            lg = np.array(logits)
            for b in banned_tokens(seq, no_repeat_ngram_size):
                lg[b] = -np.inf
            nxt = int(np.argmax(lg))
            if nxt in eos:
                break
            out.append(nxt)
            seq.append(nxt)
            logits = self._logits(self._forward([nxt], len(seq) - 1, cache))
        return out

    # ---- public -----------------------------------------------------------
    def enhance(self, prompt: str, seed: int = 42) -> str:
        """Upstream generate_enhanced_prompt for a text request. `seed` is
        accepted for API parity; greedy decoding does not use it."""
        self.load()
        text = chat_text(system_prompt("t2v"), f"user prompt: {prompt}")
        ids = self.tokenize(text)
        out = self.generate(ids)
        return clean_response(self.tokenizer.decode(out, skip_special_tokens=True))
