# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 diffusion video decoder (NADiffusionDecoder), Lightricks' default and
recommended decoder: sharper faces, textures and text than the conv decoder.

`vae/ltx-2.5-video-vae-bf16.safetensors`. Four deterministic stages of 3-D
neighborhood attention (kernels 3x7x7, 3x7x7, 3x5x5, 3x5x5) with pixel-shuffle
upsamples take the latent to pixel/4 resolution; eight diffusion blocks with an
11x11x11 window, AdaLN from the timestep and the stage-4 feature as context,
then run one x0 step from pure noise at t = 1. Reads against Lightricks'
`ltx_core/model/video_vae/diffusion_decoder/`; weights keep upstream names.

Neighborhood attention (NATTEN semantics: a window of fixed size centred on the
query and clamped inside the volume) runs as a Metal kernel, one thread per
(query, head), online softmax in fp32. The block-gather MLX formulation the
baseline uses is kept as `na3d_mlx` for validation only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints

G = mx.float32
Kernel = tuple[int, int, int]
NOISE_SEED_OFFSET = 30000
MAX_ELEMENTS = 1 << 31
W_CHUNKS = (
    4  # upstream's chunked_eager: the stage-5 width is processed in 4 haloed slabs
)
CHUNK_TOKENS = (
    3_300_000  # attention tokens per temporal chunk (a 768x512x121 clip runs whole)
)


@dataclass(frozen=True)
class DiffVAEConfig:
    in_channels: int
    out_channels: int
    patch: int
    head_dim: int
    channels: tuple[int, ...]
    depths: tuple[int, ...]
    kernels: tuple[Kernel, ...]  # four deterministic stages
    upsamples: tuple[tuple[Kernel, int], ...]  # (stride, channel reduction)
    stage5_kernel: Kernel
    timestep_scale: float
    t_hidden: int = 384
    t_freq: int = 256

    @classmethod
    def from_header(cls, header: checkpoints.Header) -> DiffVAEConfig:
        d = header.config()["vae"]["decoder"]
        if d.get("_class_name") != "NADiffusionDecoder":
            raise ValueError(f"not a diffusion decoder: {d.get('_class_name')!r}")
        return cls(
            in_channels=d["in_channels"],
            out_channels=d["out_channels"],
            patch=d["patch_size"],
            head_dim=d["head_dim"],
            channels=tuple(d["stage_channels"]),
            depths=tuple(d["stage_depths"]),
            kernels=tuple(tuple(k) for k in d["stage_kernels"][:4]),
            upsamples=tuple((tuple(s), r) for s, r in d["upsamples"]),
            stage5_kernel=tuple(d["stage5_kernel"]),
            timestep_scale=float(d.get("timestep_scale_multiplier", 1000.0)),
        )

    def cumulative_strides(self) -> list[Kernel]:
        out: list[Kernel] = [(1, 1, 1)]
        for (st, sh, sw), _ in self.upsamples:
            t, h, w = out[-1]
            out.append((t * st, h * sh, w * sw))
        return out

    def min_latent_shape(self) -> Kernel:
        strides = self.cumulative_strides()
        kernels = [*self.kernels, self.stage5_kernel]
        return tuple(
            max(math.ceil(k[a] / s[a]) for k, s in zip(kernels, strides))
            for a in range(3)
        )

    @property
    def ghost_frames(self) -> int:
        return (self.kernels[0][0] // 2) * 2


# ---- neighborhood attention --------------------------------------------------
def window_starts(length: int, kernel: int) -> mx.array:
    k_eff = min(kernel, length)
    return mx.clip(mx.arange(length) - k_eff // 2, 0, length - k_eff).astype(mx.int32)


def na3d_mlx(q: mx.array, k: mx.array, v: mx.array, kernel: Kernel) -> mx.array:
    """Reference: exact windows via a gather per query block.

    q, k, v (1, T, H, W, heads, hd).
    """
    _, t, h, w, heads, hd = q.shape
    out = mx.zeros_like(q)
    starts = [np.array(window_starts(n, kk)) for n, kk in zip((t, h, w), kernel)]
    keff = [min(kk, n) for kk, n in zip(kernel, (t, h, w))]
    for ti in range(t):
        st = starts[0][ti]
        for hi in range(h):
            sh = starts[1][hi]
            sw = starts[2]  # (w,)
            ks = mx.stack(
                [
                    k[0, st : st + keff[0], sh : sh + keff[1], s : s + keff[2]].reshape(
                        -1, heads, hd
                    )
                    for s in sw
                ]
            )  # (w, L, heads, hd)
            vs = mx.stack(
                [
                    v[0, st : st + keff[0], sh : sh + keff[1], s : s + keff[2]].reshape(
                        -1, heads, hd
                    )
                    for s in sw
                ]
            )
            qs = q[0, ti, hi]  # (w, heads, hd)
            scores = mx.einsum("wnd,wlnd->wnl", qs.astype(G), ks.astype(G))
            probs = mx.softmax(scores, axis=-1)
            out[0, ti, hi] = mx.einsum("wnl,wlnd->wnd", probs, vs.astype(G)).astype(
                q.dtype
            )
        mx.eval(out)
    return out


_NA_KERNELS: dict[tuple, object] = {}

# lanes per (query, head): each owns HD / LANES dims; loads are 128-byte coalesced
LANES = 8

_NA_SOURCE = """
    // LANES lanes per (query token, head); lane j owns dims [j*DL, (j+1)*DL).
    // q is pre-scaled.
    const uint T = shape[0], H = shape[1], W = shape[2], NH = shape[3];
    const uint gid = thread_position_in_grid.x;
    const uint lane = gid % LANES;
    const uint tid = gid / LANES;
    const uint total = T * H * W * NH;
    const bool live = tid < total;
    const uint head = live ? tid % NH : 0;
    const uint tok = live ? tid / NH : 0;
    const uint wi = tok % W;
    const uint hi = (tok / W) % H;
    const uint ti = tok / (W * H);
    const int kt = min((int)KT, (int)T);
    const int kh = min((int)KH, (int)H);
    const int kw = min((int)KW, (int)W);
    const int t0 = clamp((int)ti - kt / 2, 0, (int)T - kt);
    const int h0 = clamp((int)hi - kh / 2, 0, (int)H - kh);
    const int w0 = clamp((int)wi - kw / 2, 0, (int)W - kw);
    const uint stride_tok = NH * HD;
    const size_t base = (size_t)tid * HD + lane * DL;
    float qr[DL];
    for (uint d = 0; d < DL; ++d) qr[d] = (float)q[base + d];
    float m = -INFINITY, l = 0.0f;
    float acc[DL];
    for (uint d = 0; d < DL; ++d) acc[d] = 0.0f;
    for (int a = 0; a < kt; ++a) {
        for (int b = 0; b < kh; ++b) {
            const size_t row = ((size_t)(t0 + a) * H + (h0 + b)) * W + w0;
            for (int c = 0; c < kw; ++c) {
                const size_t off = (row + c) * stride_tok + head * HD + lane * DL;
                float s = 0.0f;
                for (uint d = 0; d < DL; ++d) s += qr[d] * (float)k[off + d];
                SHUFFLES                const float m_new = max(m, s);
                const float corr = exp(m - m_new);
                const float p = exp(s - m_new);
                l = l * corr + p;
                for (uint d = 0; d < DL; ++d)
                    acc[d] = acc[d] * corr + p * (float)v[off + d];
                m = m_new;
            }
        }
    }
    if (live) {
        const float inv = 1.0f / l;
        for (uint d = 0; d < DL; ++d) out[base + d] = (T_IN)(acc[d] * inv);
    }
"""


def na3d(
    q: mx.array, k: mx.array, v: mx.array, kernel: Kernel, lanes: int = LANES
) -> mx.array:
    """Metal neighborhood attention. q, k, v (1, T, H, W, heads, hd) in fp16 or fp32.

    Measured on the stage-5 shape (section 18 of the ledger): 8 lanes per query
    3.0 TF/s; 1-4 lanes, threadgroup-staged key rows, and row-batched softmax
    were all slower (the loop is latency-bound, and score arrays spill).
    """
    _, t, h, w, heads, hd = q.shape
    key = (kernel, hd, str(q.dtype), lanes)
    kern = _NA_KERNELS.get(key)
    if kern is None:
        src = _NA_SOURCE.replace(
            "SHUFFLES",
            "".join(
                f"s += simd_shuffle_xor(s, {o});\n" for o in (4, 2, 1) if o < lanes
            ),
        )
        for name, val in (
            ("KT", kernel[0]),
            ("KH", kernel[1]),
            ("KW", kernel[2]),
            ("HD", hd),
            ("LANES", lanes),
            ("DL", hd // lanes),
        ):
            src = src.replace(name, str(val))
        dt_tag = "f16" if q.dtype == mx.float16 else "f32"
        kern = mx.fast.metal_kernel(
            name=f"na3d_{kernel[0]}x{kernel[1]}x{kernel[2]}_d{hd}_l{lanes}_{dt_tag}",
            input_names=["q", "k", "v", "shape"],
            output_names=["out"],
            source=src,
        )
        _NA_KERNELS[key] = kern
    total = t * h * w * heads * lanes
    shape = mx.array([t, h, w, heads], dtype=mx.uint32)
    # MLX 0.32.2: a metal_kernel launched behind still-running MLX ops read
    # incomplete inputs 4 times out of 5 on the stage-4 volume of a 10 s clip
    # (NaN output); mx.eval on the inputs did not prevent it, a synchronize
    # does (ledger section 18). Costs a pipeline drain per call, ~1 ms.
    mx.synchronize()
    (out,) = kern(
        inputs=[q, k, v, shape],
        template=[("T_IN", q.dtype)],
        grid=((total + 255) // 256 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )
    mx.eval(out)
    mx.synchronize()
    return out


# ---- model ---------------------------------------------------------------------
def _rms(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    # fast.rms_norm accumulates in fp32 whatever the input dtype; no casts needed
    return mx.fast.rms_norm(x, weight.astype(x.dtype), eps)


def _rope_split(hd: int) -> tuple[int, int, int]:
    d_t = (hd // 4) // 2 * 2
    d_hw = (hd - d_t) // 2
    return d_t, d_hw, d_hw


def _inv_freqs(dim: int) -> mx.array:
    return mx.array(
        (1.0 / 10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim)).astype(
            np.float32
        )
    )


def _rotate(x: mx.array, angle: mx.array) -> mx.array:
    xe, xo = x[..., 0::2], x[..., 1::2]
    cos, sin = mx.cos(angle), mx.sin(angle)
    return mx.stack([xe * cos - xo * sin, xe * sin + xo * cos], axis=-1).reshape(
        x.shape
    )


_ROPE_FNS: dict[tuple, object] = {}


def _axial_rope(x: mx.array, t_pos, h_pos, w_pos, split, inv) -> mx.array:
    """Axial RoPE (t, h, w) on (1, T, H, W, heads, hd); compiled per split so the
    slicing, trig and rotation run as one fused pass instead of ~30 small ops."""
    d_t, d_h, _ = split
    fn = _ROPE_FNS.get(split)
    if fn is None:

        @mx.compile
        def fn(x, t_pos, h_pos, w_pos, inv_t, inv_h, inv_w):
            xf = x.astype(G)
            parts = [xf[..., :d_t], xf[..., d_t : d_t + d_h], xf[..., d_t + d_h :]]
            angles = [
                (t_pos[:, None] * inv_t[None, :])[None, :, None, None, None, :],
                (h_pos[:, None] * inv_h[None, :])[None, None, :, None, None, :],
                (w_pos[:, None] * inv_w[None, :])[None, None, None, :, None, :],
            ]
            return mx.concatenate(
                [_rotate(p, a) for p, a in zip(parts, angles)], axis=-1
            ).astype(x.dtype)

        _ROPE_FNS[split] = fn
    return fn(x, t_pos, h_pos, w_pos, *inv)


def _pixel_shuffle(x: mx.array, stride: Kernel) -> mx.array:
    b, t, h, w, cp = x.shape
    p1, p2, p3 = stride
    c = cp // (p1 * p2 * p3)
    x = x.reshape(b, t, h, w, c, p1, p2, p3).transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, t * p1, h * p2, w * p3, c)


def _patchify(x: mx.array, p: int) -> mx.array:
    b, c, f, h, w = x.shape
    x = x.reshape(b, c, f, h // p, p, w // p, p).transpose(0, 2, 3, 5, 1, 6, 4)
    return x.reshape(b, f, h // p, w // p, c * p * p)


def _unpatchify(tokens: mx.array, p: int, c: int) -> mx.array:
    b, f, hq, wr, _ = tokens.shape
    x = tokens.reshape(b, f, hq, wr, c, p, p).transpose(0, 4, 1, 2, 6, 3, 5)
    return x.reshape(b, c, f, hq * p, wr * p)


@mx.compile
def _silu_mul(g: mx.array, u: mx.array) -> mx.array:
    return g * mx.sigmoid(g) * u


def _timestep_embedding(t: mx.array, dim: int) -> mx.array:
    half = dim // 2
    freqs = mx.array(
        np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    )
    args = t.astype(G).reshape(-1, 1) * freqs.reshape(1, -1)
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


# fp16 operands and stream, untiled: 768x512x121 peaks at 16.3 GiB, 1536x1024x121 at
# 35.3 GiB (2026-10-02); slope 133 B per output pixel-frame, intercept ~10 GiB.
PEAK_BASE_BYTES = 10 << 30
PEAK_BYTES_PER_PIXEL_FRAME = 140


def estimate_peak_bytes(latent_shape: tuple[int, ...]) -> int:
    _, _, f, h, w = latent_shape
    return PEAK_BASE_BYTES + PEAK_BYTES_PER_PIXEL_FRAME * (8 * f - 7) * (32 * h) * (
        32 * w
    )


class DiffusionVAE:
    """`operand` is the GEMM / attention dtype, `stream` the residual dtype.

    Upstream runs this decoder entirely in bf16. fp16 operands with an fp16
    stream measure 64.5 dB / max 1 level against an all-fp32 run (fp32 stream:
    65.4 dB), at 7.5 GiB instead of 10.2 for 768x512x49; both are defaults.
    """

    def __init__(
        self,
        root: Path | None = None,
        operand: mx.Dtype = mx.float16,
        stream: mx.Dtype = mx.float16,
        attention: str = "metal",
    ):
        self.path = checkpoints.path_of("video-vae-diffusion", root)
        self.operand, self.stream, self.attention = operand, stream, attention
        self.w: dict[str, mx.array] = {}
        self.cfg: DiffVAEConfig | None = None

    def load(self) -> DiffusionVAE:
        if self.w:
            return self
        header = checkpoints.read_header(self.path)
        self.cfg = DiffVAEConfig.from_header(header)
        raw = checkpoints.load_raw(self.path)
        w = {}
        for name in list(raw):
            if not name.startswith(("decoder.", "per_channel_statistics.")):
                raw.pop(name)
                continue
            a = raw.pop(name)
            glue = name.endswith(
                ("norm.weight", "norm1.weight", "norm2.weight", "norm_out.weight")
            ) or (
                "scale_shift_table" in name
                or "per_channel" in name
                or "type_emb" in name
            )
            w[name] = a.astype(G if glue else self.operand)
        mx.eval(w)
        self.w = w
        self.split = _rope_split(self.cfg.head_dim)
        self.inv = tuple(_inv_freqs(d) for d in self.split)
        return self

    # ---- pieces ----
    def _lin(self, name: str, x: mx.array) -> mx.array:
        wt, b = self.w[name + ".weight"], self.w.get(name + ".bias")
        if (x.size // x.shape[-1]) * wt.shape[0] >= MAX_ELEMENTS:
            # MLX 0.32.2 slices/splits of arrays past 2^31 elements return wrong
            # data for the tail (section 18); callers chunk before getting here.
            raise ValueError(f"{name}: output would exceed 2^31 elements; chunk it")
        y = x.astype(self.operand) @ wt.T
        return (y if b is None else y + b).astype(self.stream)

    def _attn(
        self,
        p: str,
        y: mx.array,
        kernel: Kernel,
        w_pos: mx.array | None = None,
        t_pos: mx.array | None = None,
    ) -> mx.array:
        """y (1, T, H, W, C) already normalised/modulated
        -> attention output projected (stream dtype)."""
        cfg = self.cfg
        b, t, h, w, c = y.shape
        heads, hd = c // cfg.head_dim, cfg.head_dim
        # three projections rather than one split: the fused output passes 2^31
        # elements on long clips, where mx.split returns garbage for the tail
        wq, bq = self.w[p + ".qkv.weight"], self.w[p + ".qkv.bias"]
        yo = y.astype(self.operand)
        q, k, v = (
            (yo @ wq[i * c : (i + 1) * c].T + bq[i * c : (i + 1) * c])
            .astype(self.stream)
            .reshape(b, t, h, w, heads, hd)
            for i in range(3)
        )
        del yo
        q = _rms(q, self.w[p + ".q_norm.weight"]) * hd**-0.5
        k = _rms(k, self.w[p + ".k_norm.weight"])
        t_pos = mx.arange(t).astype(G) if t_pos is None else t_pos
        h_pos = mx.arange(h).astype(G)
        w_pos = mx.arange(w).astype(G) if w_pos is None else w_pos
        q = _axial_rope(q, t_pos, h_pos, w_pos, self.split, self.inv)
        k = _axial_rope(k, t_pos, h_pos, w_pos, self.split, self.inv)
        q, k, v = (a.astype(self.operand) for a in (q, k, v))
        mx.eval(q, k, v)  # the fp32 RoPE temporaries die here
        del y
        o = (
            na3d(q, k, v, kernel)
            if self.attention == "metal"
            else na3d_mlx(q, k, v, kernel)
        )
        mx.eval(o)
        del q, k, v
        y = self._lin(p + ".proj", o.reshape(b, t, h, w, c))
        mx.eval(y)
        return y

    def _t_chunks(self, x: mx.array, kernel: Kernel):
        """(core start, core end, slab start, slab end) temporal chunks whose
        slabs carry a full halo, so windows clamp exactly as in the whole volume."""
        t_total, h, w = x.shape[1:4]
        halo, kt = kernel[0] // 2, min(kernel[0], x.shape[1])
        per_frame = h * w
        frames = max(1, CHUNK_TOKENS // per_frame)
        step = math.ceil(t_total / math.ceil(t_total / frames))
        for c0 in range(0, t_total, step):
            c1 = min(t_total, c0 + step)
            s0, s1 = max(0, c0 - halo), min(t_total, c1 + halo)
            if s1 - s0 < kt:
                s0, s1 = (max(0, s1 - kt), s1) if s0 > 0 else (0, min(t_total, kt))
            yield c0, c1, s0, s1

    def _chunked_attention(
        self, p: str, x: mx.array, kernel: Kernel, pre, w_slabs: bool
    ) -> mx.array:
        """Attention over temporal chunks (and upstream's W slabs for stage 5);
        `pre` maps a slab of x to its normalised/modulated input."""
        cores = []
        for c0, c1, s0, s1 in self._t_chunks(x, kernel):
            xs = x[:, s0:s1]
            t_pos = mx.arange(s0, s1).astype(G)
            if w_slabs:
                halo = kernel[2] // 2
                outs = []
                for buf, w_pos, n in self._w_slabs(xs, halo):
                    o = self._attn(p, pre(buf), kernel, w_pos=w_pos, t_pos=t_pos)
                    del buf
                    o = o[:, c0 - s0 : c1 - s0, :, halo : halo + n]
                    mx.eval(o)
                    outs.append(o)
                core = mx.concatenate(outs, axis=3)
                del outs
            else:
                core = self._attn(p, pre(xs), kernel, t_pos=t_pos)[:, c0 - s0 : c1 - s0]
            mx.eval(core)
            cores.append(core)
        return cores[0] if len(cores) == 1 else mx.concatenate(cores, axis=1)

    def _swiglu(self, p: str, x: mx.array, chunks: int = 4) -> mx.array:
        """Per-token MLP in temporal chunks: the 4x hidden state stays in the
        operand dtype and never exists for the whole volume at once."""
        wg, wu, wd = (self.w[f"{p}.{n}.weight"] for n in ("w_gate", "w_up", "w_down"))
        outs = []
        step = max(1, math.ceil(x.shape[1] / chunks))
        for t0 in range(0, x.shape[1], step):
            xo = x[:, t0 : t0 + step].astype(self.operand)
            h = _silu_mul(xo @ wg.T, xo @ wu.T)
            y = (h @ wd.T).astype(self.stream)
            mx.eval(y)
            outs.append(y)
        return mx.concatenate(outs, axis=1)

    def _det_block(self, p: str, x: mx.array, kernel: Kernel) -> mx.array:
        n1 = self.w[p + ".norm1.weight"]
        x = x + self._chunked_attention(
            p + ".attn", x, kernel, lambda b: _rms(b, n1), w_slabs=False
        )
        x = x + self._swiglu(p + ".mlp", _rms(x, self.w[p + ".norm2.weight"]))
        mx.eval(x)
        return x

    def _upsample(self, i: int, x: mx.array, drop_leading: bool) -> mx.array:
        stride, _ = self.cfg.upsamples[i]
        y = _pixel_shuffle(self._lin(f"decoder.upsamples.{i}.proj", x), stride)
        return y[:, 1:] if (stride[0] == 2 and drop_leading) else y

    def _w_slabs(self, x: mx.array, halo: int):
        """Upstream build_w_slabs: W_CHUNKS slabs with a `halo` on each side."""
        width = x.shape[3]
        chunk = math.ceil(width / W_CHUNKS)
        extent = chunk + 2 * halo
        left = None
        for i in range(W_CHUNKS):
            cs, ce = i * chunk, min(width, (i + 1) * chunk)
            core = x[:, :, :, cs:ce]
            n = ce - cs
            has_right = i < W_CHUNKS - 1
            if left is None:
                lp = (
                    mx.repeat(core[:, :, :, :1], halo, axis=3)
                    if n
                    else mx.zeros_like(x[:, :, :, :halo])
                )
            else:
                lp = (
                    left
                    if left.shape[3] >= halo
                    else mx.concatenate(
                        [mx.zeros_like(x[:, :, :, : halo - left.shape[3]]), left],
                        axis=3,
                    )
                )
            parts = [lp, core]
            filled = 0
            if has_right:
                right = x[:, :, :, ce : min(width, ce + halo)]
                filled = right.shape[3]
                if filled:
                    parts.append(right)
            missing = extent - (halo + n + filled)
            if missing > 0:
                fill = core[:, :, :, -1:] if n else mx.zeros_like(x[:, :, :, :1])
                parts.append(mx.repeat(fill, missing, axis=3))
            buf = mx.concatenate(parts, axis=3)
            if has_right:
                left = x[:, :, :, ce - min(halo, n) : ce]
            yield buf, mx.arange(extent).astype(G) + float(cs - halo), n

    def _chunk_bounds(
        self, t_total: int, h: int, w: int, kernel: Kernel
    ) -> list[tuple[int, int]]:
        frames = max(1, CHUNK_TOKENS // (h * w))
        step = math.ceil(t_total / math.ceil(t_total / frames))
        return [(c0, min(t_total, c0 + step)) for c0 in range(0, t_total, step)]

    def _slab(
        self,
        chunks: list[mx.array],
        bounds: list[tuple[int, int]],
        i: int,
        kernel: Kernel,
    ):
        """Chunk i plus its temporal halo gathered from the neighbouring chunks:
        (slab, core offset, slab start frame). Exact: windows are centred and
        clamped against the volume, and every core query finds the same keys."""
        t_total = bounds[-1][1]
        halo, kt = kernel[0] // 2, min(kernel[0], t_total)
        c0, c1 = bounds[i]
        s0, s1 = max(0, c0 - halo), min(t_total, c1 + halo)
        if s1 - s0 < kt:
            s0, s1 = (max(0, s1 - kt), s1) if s0 > 0 else (0, min(t_total, kt))
        parts = []
        for (b0, b1), ch in zip(bounds, chunks):
            lo, hi = max(b0, s0), min(b1, s1)
            if lo < hi:
                parts.append(ch[:, lo - b0 : hi - b0])
        return (
            (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)),
            c0 - s0,
            s0,
        )

    def _diff_block(
        self,
        p: str,
        chunks: list[mx.array],
        bounds: list[tuple[int, int]],
        context: list[mx.array],
        mod: list[mx.array],
    ) -> list[mx.array]:
        """One stage-5 block over the chunk-resident residual stream."""
        table = self.w[p + ".scale_shift_table"]
        pm = [m + table[i].reshape(1, 1, 1, 1, -1) for i, m in enumerate(mod)]
        scale_msa, shift_msa, scale_mlp, shift_mlp = pm[0], pm[1], pm[3], pm[4]
        n1, n2 = self.w[p + ".norm1.weight"], self.w[p + ".norm2.weight"]
        kernel = self.cfg.stage5_kernel
        halo_w = kernel[2] // 2
        # 1. context injection, per chunk
        for i, (c0, c1) in enumerate(bounds):
            chunks[i] = (chunks[i] + self._lin(p + ".context_proj", context[i])).astype(
                self.stream
            )
            mx.eval(chunks[i])
        # 2. attention: read slabs from the pre-attention stream, write the new stream
        new = []
        for i, (c0, c1) in enumerate(bounds):
            slab, off, s0 = self._slab(chunks, bounds, i, kernel)
            t_pos = mx.arange(s0, s0 + slab.shape[1]).astype(G)
            outs = []
            for buf, w_pos, n in self._w_slabs(slab, halo_w):
                y = _rms(buf, n1) * (1 + scale_msa) + shift_msa
                del buf
                o = self._attn(p + ".attn", y, kernel, w_pos=w_pos, t_pos=t_pos)
                del y
                o = o[:, off : off + (c1 - c0), :, halo_w : halo_w + n]
                mx.eval(o)
                outs.append(o)
            del slab
            core = (chunks[i] + mx.concatenate(outs, axis=3)).astype(self.stream)
            del outs
            mx.eval(core)
            new.append(core)
        chunks = new
        # 3. MLP, per chunk
        for i in range(len(chunks)):
            chunks[i] = chunks[i] + self._swiglu(
                p + ".mlp", _rms(chunks[i], n2) * (1 + scale_mlp) + shift_mlp
            )
            mx.eval(chunks[i])
        return chunks

    # ---- decode ----
    def denormalize(self, z: mx.array) -> mx.array:
        s = self.w["per_channel_statistics.std-of-means"].reshape(1, -1, 1, 1, 1)
        m = self.w["per_channel_statistics.mean-of-means"].reshape(1, -1, 1, 1, 1)
        return z.astype(G) * s + m

    def _pad_to_floor(self, latent: mx.array):
        f_min, h_min, w_min = self.cfg.min_latent_shape()
        _, _, f, h, w = latent.shape
        t_pad, h_need, w_need = max(f_min - f, 0), max(h_min - h, 0), max(w_min - w, 0)
        h_b, w_b = h_need // 2, w_need // 2
        if t_pad:
            latent = mx.concatenate(
                [latent, mx.repeat(latent[:, :, -1:], t_pad, axis=2)], axis=2
            )
        if h_need:
            latent = mx.concatenate(
                [
                    mx.repeat(latent[:, :, :, :1], h_b, 3),
                    latent,
                    mx.repeat(latent[:, :, :, -1:], h_need - h_b, 3),
                ],
                axis=3,
            )
        if w_need:
            latent = mx.concatenate(
                [
                    mx.repeat(latent[..., :1], w_b, 4),
                    latent,
                    mx.repeat(latent[..., -1:], w_need - w_b, 4),
                ],
                axis=4,
            )
        return latent, h_b, w_b

    def stages_1_to_4(self, latent_padded: mx.array, tap=None) -> mx.array:
        cfg = self.cfg
        z = self.denormalize(latent_padded)
        z = mx.concatenate(
            [z, mx.repeat(z[:, :, -1:], cfg.ghost_frames, axis=2)], axis=2
        )
        x = self._lin("decoder.conv_in", z.transpose(0, 2, 3, 4, 1))
        for s in range(4):
            for i in range(cfg.depths[s]):
                x = self._det_block(f"decoder.det_stages.{s}.{i}", x, cfg.kernels[s])
            if tap:
                tap(f"s{s + 1}.out", x)
            if s < 3:
                x = self._upsample(s, x, drop_leading=True)
        # ghost crop: the replicated frames fed stage 1-4's windows and are dropped here
        strides = cfg.cumulative_strides()[3]
        keep = min(
            x.shape[1],
            max(
                x.shape[1] - cfg.ghost_frames * strides[0],
                math.ceil(cfg.stage5_kernel[0] / 2),
            ),
        )
        return x[:, :keep]

    def stage_5(
        self, x_t: mx.array, feat: mx.array, t: float = 1.0, tap=None
    ) -> mx.array:
        cfg = self.cfg
        (st3, _, _), _ = cfg.upsamples[3]
        te = _timestep_embedding(mx.array([t * cfg.timestep_scale]), cfg.t_freq)
        te = self._lin("decoder.t_embedder.mlp.0", te)
        te = self._lin("decoder.t_embedder.mlp.2", te * mx.sigmoid(te))
        mod = self._lin("decoder.shared_adaln.proj", te * mx.sigmoid(te))
        mod = [m.reshape(1, 1, 1, 1, -1) for m in mx.split(mod, 7, axis=-1)]
        tokens = _patchify(x_t, cfg.patch)
        bounds = self._chunk_bounds(
            tokens.shape[1], tokens.shape[2], tokens.shape[3], cfg.stage5_kernel
        )
        chunks = []
        for c0, c1 in bounds:
            ch = self._lin("decoder.conv_in_x_t", tokens[:, c0:c1])
            mx.eval(ch)
            chunks.append(ch)
        del tokens, x_t
        # Context frames come from upsample 3 of the stage-4 feature (temporal
        # stride 2, leading frame dropped), computed per stage-5 chunk: the
        # whole context passes 2^31 elements on long clips.
        context = []
        for c0, c1 in bounds:
            if (
                st3 == 2
            ):  # context frame j is shuffled frame j + 1, i.e. feat frame (j + 1) // 2
                a, bnd = (c0 + 1) // 2, min(feat.shape[1], (c1 + 1) // 2 + 1)
                piece = _pixel_shuffle(
                    self._lin("decoder.upsamples.3.proj", feat[:, a:bnd]),
                    cfg.upsamples[3][0],
                )[:, c0 + 1 - 2 * a : c1 + 1 - 2 * a]
            else:
                piece = _pixel_shuffle(
                    self._lin("decoder.upsamples.3.proj", feat[:, c0:c1]),
                    cfg.upsamples[3][0],
                )
            piece = piece.astype(self.operand)
            mx.eval(piece)
            if piece.shape[1] != c1 - c0:
                raise RuntimeError(f"context chunk {piece.shape} for frames {c0}:{c1}")
            context.append(piece)
        del feat
        for i in range(cfg.depths[4]):
            chunks = self._diff_block(
                f"decoder.diff_blocks.{i}", chunks, bounds, context, mod
            )
            if tap:
                tap(
                    f"s5.b{i}.out",
                    chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1),
                )
        del context
        pixels = []
        for ch in chunks:
            px = _unpatchify(
                self._lin(
                    "decoder.conv_out", _rms(ch, self.w["decoder.norm_out.weight"])
                ),
                cfg.patch,
                cfg.out_channels,
            )
            mx.eval(px)
            pixels.append(px)
        return pixels[0] if len(pixels) == 1 else mx.concatenate(pixels, axis=2)

    def decode_raw(self, latent: mx.array, seed: int = 0, tap=None) -> mx.array:
        """Normalized latent (1, 128, F, H, W)
        -> pixels (1, 3, 8F-7, 32H, 32W) fp32 in ~[-1, 1]."""
        self.load()
        cfg = self.cfg
        _, _, f, h, w = latent.shape
        padded, h_b, w_b = self._pad_to_floor(latent.astype(G))
        feat = self.stages_1_to_4(padded, tap)
        t4, h4, w4 = feat.shape[1:4]
        (st, sh, sw), _ = cfg.upsamples[3]
        canvas = (t4 * st - 1, h4 * sh * cfg.patch, w4 * sw * cfg.patch)
        noise = mx.random.normal(
            (1, cfg.out_channels, *canvas), key=mx.random.key(seed + NOISE_SEED_OFFSET)
        )
        pixels = self.stage_5(noise.astype(self.stream), feat, 1.0, tap)
        t_scale = cfg.cumulative_strides()[4][0]
        s_scale = cfg.cumulative_strides()[4][1] * cfg.patch
        f_px, h_px, w_px = (f - 1) * t_scale + 1, h * s_scale, w * s_scale
        return pixels[
            :,
            :,
            :f_px,
            h_b * s_scale : h_b * s_scale + h_px,
            w_b * s_scale : w_b * s_scale + w_px,
        ].astype(G)
