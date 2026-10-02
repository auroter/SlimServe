# SPDX-License-Identifier: Apache-2.0
"""Latent state, positions, noise and samplers for LTX-2.5.

Everything here is fp32: latents, noise, the x0 blend and the sampler steps.
Constants and formulas follow Lightricks' `ltx_pipelines` (utils/constants.py,
utils/samplers.py, components/diffusion_steps.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import mlx.core as mx

G = mx.float32

# 8 steps; front-loaded near sigma 1 (trained with the ancestral sampler).
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
ANCESTRAL_NOISE_SEED_OFFSET = 10000

VIDEO_TEMPORAL_SCALE = 8
VIDEO_SPATIAL_SCALE = 32
AUDIO_LATENTS_PER_SECOND = 16000 / 160 / 4  # sample rate / hop / downsample = 25


def video_latent_shape(num_frames: int, height: int, width: int) -> tuple[int, int, int]:
    return (num_frames + 7) // 8, height // 32, width // 32


def snap_dimensions(height: int, width: int, two_stage: bool = True) -> tuple[int, int]:
    m = 64 if two_stage else 32
    return max(m, height // m * m), max(m, width // m * m)


def audio_token_count(num_frames: int, fps: float) -> int:
    return round(num_frames / fps * AUDIO_LATENTS_PER_SECOND)


def video_positions(f: int, h: int, w: int, fps: float) -> mx.array:
    """(1, F*H*W, 3): seconds, pixel row, pixel column; midpoints, causal first frame."""
    idx = mx.arange(f).astype(G)
    starts = mx.maximum(idx * 8 + 1 - 8, 0.0)
    ends = mx.maximum((idx + 1) * 8 + 1 - 8, 0.0)
    t = (starts + ends) / 2.0 / fps
    y = mx.arange(h).astype(G) * 32 + 16.0
    x = mx.arange(w).astype(G) * 32 + 16.0
    grid = mx.stack(
        [
            mx.broadcast_to(t[:, None, None], (f, h, w)),
            mx.broadcast_to(y[None, :, None], (f, h, w)),
            mx.broadcast_to(x[None, None, :], (f, h, w)),
        ],
        axis=-1,
    )
    return grid.reshape(1, -1, 3)


def audio_positions(tokens: int) -> mx.array:
    """(1, T, 1): midpoint of each audio latent in seconds (causal)."""
    idx = mx.arange(tokens).astype(G)
    starts = mx.maximum(idx * 4 + 1 - 4, 0.0) * 160 / 16000
    ends = mx.maximum((idx + 1) * 4 + 1 - 4, 0.0) * 160 / 16000
    return ((starts + ends) / 2.0)[None, :, None]


def patchify(latent: mx.array) -> mx.array:
    b, c, f, h, w = latent.shape
    return latent.transpose(0, 2, 3, 4, 1).reshape(b, f * h * w, c)


def unpatchify(tokens: mx.array, dims: tuple[int, int, int]) -> mx.array:
    f, h, w = dims
    return tokens.reshape(tokens.shape[0], f, h, w, -1).transpose(0, 4, 1, 2, 3)


@dataclass(frozen=True)
class LatentState:
    latent: mx.array  # (B, N, C) current noisy tokens
    clean: mx.array  # (B, N, C) values restored where denoise_mask == 0
    denoise_mask: mx.array  # (B, N, 1): 1 generate, 0 preserve
    positions: mx.array
    keyframes_mask: mx.array | None = None  # (B, N, 1): tokens encoding one pixel frame
    attention_mask: mx.array | None = None
    frozen: bool = False

    @property
    def uniform(self) -> bool:
        return bool(mx.all(self.denoise_mask == 1.0).item())


def _normal(shape: tuple[int, ...], seed: int, bf16_noise: bool) -> mx.array:
    mx.random.seed(seed)
    n = mx.random.normal(shape)
    # The baseline runner rounds its noise to bf16; reproducing that gives the
    # same starting point for same-seed comparisons against it.
    return n.astype(mx.bfloat16).astype(G) if bf16_noise else n


def noised_state(
    shape: tuple[int, int, int],
    positions: mx.array,
    seed: int,
    sigma: float = 1.0,
    initial: mx.array | None = None,
    tokens_per_frame: int | None = None,
    bf16_noise: bool = False,
) -> LatentState:
    """noise * sigma + clean * (1 - sigma) on every token; conditioning is applied by the caller."""
    clean = mx.zeros(shape, dtype=G) if initial is None else initial.astype(G)
    noise = _normal(shape, seed, bf16_noise)
    kf = None
    if tokens_per_frame is not None:  # video: the causal first latent frame is one pixel frame
        b, n, _ = shape
        kf = (mx.arange(n) < tokens_per_frame).astype(G).reshape(1, n, 1)
        kf = mx.broadcast_to(kf, (b, n, 1))
    return LatentState(
        latent=noise * sigma + clean * (1.0 - sigma),
        clean=clean,
        denoise_mask=mx.ones((shape[0], shape[1], 1), dtype=G),
        positions=positions,
        keyframes_mask=kf,
    )


def blend(x0: mx.array, state: LatentState) -> mx.array:
    return x0 * state.denoise_mask + state.clean * (1.0 - state.denoise_mask)


# A denoiser maps (video_state, audio_state, video_x, audio_x, sigma) -> (video_x0, audio_x0).
Denoiser = Callable[[LatentState, LatentState, mx.array, mx.array, float], tuple[mx.array, mx.array]]


def euler_loop(
    denoise: Denoiser,
    video: LatentState,
    audio: LatentState,
    sigmas: list[float],
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[mx.array, mx.array]:
    vx, ax = video.latent, audio.latent
    for i, (s, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        v0, a0 = denoise(video, audio, vx, ax, s)
        v0, a0 = blend(v0, video), blend(a0, audio)
        dt = s_next - s
        vx = vx + (vx - v0) / s * dt
        ax = ax + (ax - a0) / s * dt
        mx.eval(vx, ax)
        if on_step:
            on_step(i, s)
    return vx, ax


def euler_ancestral_loop(
    denoise: Denoiser,
    video: LatentState,
    audio: LatentState,
    sigmas: list[float],
    noise_seed: int,
    eta: float = 1.0,
    s_noise: float = 1.0,
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[mx.array, mx.array]:
    """Rectified-flow ancestral Euler (alpha = 1 - sigma), one generator, video noise drawn first."""
    vx, ax = video.latent, audio.latent
    mx.random.seed(noise_seed)

    def step(x: mx.array, x0: mx.array, s: float, s_next: float, noise: mx.array | None) -> mx.array:
        sigma_down = s_next * (1.0 + (s_next / s - 1.0) * eta)
        r = sigma_down / s
        nxt = r * x + (1.0 - r) * x0
        if eta > 0:
            a_next, a_down = 1.0 - s_next, 1.0 - sigma_down
            coeff = max(s_next**2 - sigma_down**2 * a_next**2 / a_down**2, 0.0) ** 0.5
            nxt = (a_next / a_down) * nxt + noise * (s_noise * coeff)
        return nxt

    for i, (s, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        v0, a0 = denoise(video, audio, vx, ax, s)
        v0, a0 = blend(v0, video), blend(a0, audio)
        if s_next == 0:
            vx, ax = v0, a0
        else:
            vn = mx.random.normal(vx.shape) if eta > 0 else None
            an = mx.random.normal(ax.shape) if eta > 0 else None
            vx, ax = step(vx, v0, s, s_next, vn), step(ax, a0, s, s_next, an)
            if eta > 0:
                vx, ax = blend(vx, video), blend(ax, audio)
        mx.eval(vx, ax)
        if on_step:
            on_step(i, s)
        if s_next == 0:
            break
    return vx, ax


def with_latent(state: LatentState, latent: mx.array) -> LatentState:
    return replace(state, latent=latent)


# ---- dev (guided) ---------------------------------------------------------
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


def ltx2_schedule(steps: int, num_tokens: int, base_shift: float = 0.95, max_shift: float = 2.05,
                  terminal: float = 0.1) -> list[float]:
    """Token-count-shifted schedule (LinearQuadratic family), stretched so the last non-zero sigma is `terminal`."""
    import math

    import numpy as np

    sigmas = np.linspace(1.0, 0.0, steps + 1)
    slope = (max_shift - base_shift) / (4096 - 1024)
    shift = num_tokens * slope + base_shift - slope * 1024
    nz = sigmas != 0
    sigmas[nz] = math.exp(shift) / (math.exp(shift) + (1.0 / sigmas[nz] - 1.0))
    one_minus = 1.0 - sigmas[nz]
    scale = one_minus[-1] / (1.0 - terminal)
    if scale != 0:
        sigmas[nz] = 1.0 - one_minus / scale
    return sigmas.tolist()


@dataclass(frozen=True)
class Guidance:
    cfg: float = 3.0
    stg: float = 1.0
    modality: float = 3.0
    rescale: float = 0.7
    stg_blocks: tuple[int, ...] = (28,)

    def combine(self, cond: mx.array, uncond: mx.array, perturbed: mx.array, isolated: mx.array) -> mx.array:
        pred = cond + (self.cfg - 1) * (cond - uncond) + self.stg * (cond - perturbed) + (self.modality - 1) * (cond - isolated)
        if self.rescale:
            factor = mx.sqrt(mx.var(cond)) / (mx.sqrt(mx.var(pred)) + 1e-8)
            pred = pred * (self.rescale * factor + (1 - self.rescale))
        return pred
