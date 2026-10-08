# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 pipelines on the SlimServe engine.

`distilled` mirrors Lightricks' DistilledPipeline: stage 1 at half resolution
(8 ancestral Euler steps, no guidance), 2x latent upscale, stage 2 at full
resolution (3 Euler steps), with the same distilled transformer in both.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints, sampling
from slimserve.video.ltx25.dit import (
    DiTConfig,
    LTX25DiT,
    StepCache,
    attention_tiles,
    x0_from_velocity,
)
from slimserve.video.ltx25.sampling import LatentState

G = mx.float32
OS_RESERVE_BYTES = 24 << 30  # never plan Metal memory into the last 24 GiB
# MLX buffer cache: decode buffers are reused across steps and requests; a
# 12 GiB cap cost 0.8 s per short decode, 16 GiB 0.5 s, unbounded nothing.
CACHE_BYTES = 16 << 30
# The diffusion decoder's working set of distinct buffer sizes (three or four
# slab-shape families with their q/k/v and RoPE temporaries) is ~40 GiB; under
# a 16 GiB cache every call allocates from the OS and the decode is 25% slower
# (ledger section 22). render() lends it this much cache when there is room,
# parking the DiT (6 s to reload) if that is what makes room.
DECODE_CACHE_BYTES = 48 << 30
DEFAULT_VIDEO_GUIDANCE = sampling.Guidance(cfg=3.0)
DEFAULT_AUDIO_GUIDANCE = sampling.Guidance(cfg=7.0)
# upstream LTX_2_3_HQ_PARAMS (a plain constant, no per-version override): no STG
HQ_VIDEO_GUIDANCE = sampling.Guidance(cfg=3.0, stg=0.0, modality=3.0, rescale=0.45)
HQ_AUDIO_GUIDANCE = sampling.Guidance(cfg=7.0, stg=0.0, modality=3.0, rescale=1.0)
HQ_STEPS = 15
HQ_LORA_STAGE_1, HQ_LORA_STAGE_2 = 0.25, 0.5
RES2S_NOISE_SEED_OFFSET = 40000


@dataclass
class Timings:
    spans: dict[str, float] = field(default_factory=dict)
    steps: list[tuple[str, int, float]] = field(default_factory=list)
    memory: dict[str, tuple[int, int, int]] = field(
        default_factory=dict
    )  # peak, active, cache bytes

    def span(self, name: str):
        timings = self

        class _Span:
            def __enter__(self):
                mx.synchronize()
                mx.reset_peak_memory()
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                mx.synchronize()
                timings.spans[name] = (
                    timings.spans.get(name, 0.0) + time.perf_counter() - self.t
                )
                timings.memory[name] = (
                    mx.get_peak_memory(),
                    mx.get_active_memory(),
                    mx.get_cache_memory(),
                )

        return _Span()


@dataclass
class Result:
    video_latent: mx.array  # (1, 128, F, H, W), normalized
    audio_tokens: mx.array  # (1, T, 128)
    num_frames: int
    height: int
    width: int
    fps: float
    timings: Timings
    # DFR: the stage-2 keyframe slots (normalized (1, 128, K, H, W)) and their
    # pixel frames, for the keyframe-aware diffusion decode
    keyframes: tuple[mx.array, list[int]] | None = None
    # set when the clip length came from the duration head
    predicted_seconds: float | None = None


class Denoiser:
    """x0 prediction for one sampler step; owns the DiT call contract.

    `video_tiles` (dit.attention_tiles) tiles the video self-attention; both
    it and the sampler's `step_cache` are fast-tier, output-changing options."""

    def __init__(
        self,
        dit: LTX25DiT,
        video_text: mx.array,
        audio_text: mx.array,
        video_tiles=None,
    ):
        self.dit, self.video_text, self.audio_text = dit, video_text, audio_text
        self.video_tiles = video_tiles

    def __call__(
        self,
        video: LatentState,
        audio: LatentState,
        vx: mx.array,
        ax: mx.array,
        sigma: float,
        step_cache: StepCache | None = None,
    ):
        b = vx.shape[0]
        t = mx.full((b,), sigma, dtype=G)
        vt = None if video.uniform else (video.denoise_mask * sigma).squeeze(-1)
        at = None if audio.uniform else (audio.denoise_mask * sigma).squeeze(-1)
        v, a = self.dit(
            vx,
            ax,
            t,
            self.video_text,
            self.audio_text,
            video.positions,
            audio.positions,
            video_keyframes_mask=video.keyframes_mask,
            video_timesteps=vt,
            audio_timesteps=at,
            video_sigma=mx.zeros((b,), dtype=G) if video.frozen else None,
            audio_sigma=mx.zeros((b,), dtype=G) if audio.frozen else None,
            video_attention_mask=video.attention_mask,
            audio_attention_mask=audio.attention_mask,
            video_tiles=self.video_tiles,
            step_cache=step_cache,
        )
        return x0_from_velocity(vx, v, t if vt is None else vt), x0_from_velocity(
            ax, a, t if at is None else at
        )


class GuidedDenoiser:
    """Dev-model x0 with CFG + STG + modality guidance.

    Up to four passes (conditional, negative prompt, self-attention skipped on
    the STG blocks, audio<->video cross-attention skipped everywhere) run as
    one batch: the same FLOPs as the separate forwards, with every GEMM at the
    batch's rows. A pass whose scale is neutral in both modalities (cfg 1,
    stg 0, modality 1) is not run at all.
    """

    def __init__(
        self,
        dit: LTX25DiT,
        cond: tuple[mx.array, mx.array],
        negative: tuple[mx.array, mx.array],
        video: sampling.Guidance,
        audio: sampling.Guidance,
        batched: bool = True,
        video_tiles=None,
    ):
        self.dit, self.video_g, self.audio_g, self.batched = dit, video, audio, batched
        self.video_tiles = video_tiles
        kinds = ["cond"]
        if video.cfg != 1 or audio.cfg != 1:
            kinds.append("neg")
        stg_blocks = (video.stg_blocks if video.stg else ()) + (
            audio.stg_blocks if audio.stg else ()
        )
        if stg_blocks:
            kinds.append("stg")
        if video.modality != 1 or audio.modality != 1:
            kinds.append("mod")
        self.kinds = kinds
        self.passes = len(kinds)
        # Distinct text rows (conditional[, negative]); row map per pass.
        if "neg" in kinds:
            self.video_text = mx.concatenate([cond[0], negative[0]], axis=0)
            self.audio_text = mx.concatenate([cond[1], negative[1]], axis=0)
        else:
            self.video_text, self.audio_text = cond
        self.text_rows = mx.array([1 if k == "neg" else 0 for k in kinds])
        # The STG pass equals the conditional pass until the first STG block.
        self.share = None
        if "stg" in kinds and batched:
            self.share = (min(stg_blocks), 0, kinds.index("stg"))
        keep_stg = mx.array([0.0 if k == "stg" else 1.0 for k in kinds])
        keep_mod = mx.array([0.0 if k == "mod" else 1.0 for k in kinds])
        self.stg: dict[tuple[str, int], mx.array] = {}
        if "stg" in kinds:
            for blk in video.stg_blocks:
                self.stg[("video_self", blk)] = keep_stg
            for blk in audio.stg_blocks:
                self.stg[("audio_self", blk)] = keep_stg
        if "mod" in kinds:
            for blk in range(dit.cfg.num_layers):
                self.stg[("a2v", blk)] = keep_mod
                self.stg[("v2a", blk)] = keep_mod

    def _forward(
        self,
        video: LatentState,
        audio: LatentState,
        vx,
        ax,
        sigma: float,
        rows: slice,
        step_cache: StepCache | None = None,
    ):
        n = len(range(*rows.indices(self.passes)))
        if n == self.passes:  # distinct text rows, mapped per batch row
            vtext, atext, trows = self.video_text, self.audio_text, self.text_rows
        else:
            sel = self.text_rows[rows]
            vtext, atext, trows = (
                mx.take(self.video_text, sel, axis=0),
                mx.take(self.audio_text, sel, axis=0),
                None,
            )
        rep = lambda x: None if x is None else mx.repeat(x, n, axis=0)  # noqa: E731
        t = mx.full((n,), sigma, dtype=G)
        vt = None if video.uniform else rep((video.denoise_mask * sigma).squeeze(-1))
        at = None if audio.uniform else rep((audio.denoise_mask * sigma).squeeze(-1))
        v, a = self.dit(
            rep(vx),
            rep(ax),
            t,
            vtext,
            atext,
            rep(video.positions),
            rep(audio.positions),
            video_keyframes_mask=rep(video.keyframes_mask),
            video_timesteps=vt,
            audio_timesteps=at,
            video_attention_mask=video.attention_mask,
            audio_attention_mask=audio.attention_mask,
            stg={k: m[rows] for k, m in self.stg.items()},
            text_rows=trows,
            share_from=self.share if n == self.passes else None,
            video_tiles=self.video_tiles,
            step_cache=step_cache if n == self.passes else None,
        )
        return (
            x0_from_velocity(rep(vx), v, t if vt is None else vt),
            x0_from_velocity(rep(ax), a, t if at is None else at),
        )

    def __call__(
        self,
        video: LatentState,
        audio: LatentState,
        vx: mx.array,
        ax: mx.array,
        sigma: float,
        step_cache: StepCache | None = None,
    ):
        if self.batched:
            v0, a0 = self._forward(
                video, audio, vx, ax, sigma, slice(0, self.passes), step_cache
            )
        else:
            parts = [
                self._forward(video, audio, vx, ax, sigma, slice(i, i + 1))
                for i in range(self.passes)
            ]
            v0 = mx.concatenate([p[0] for p in parts], axis=0)
            a0 = mx.concatenate([p[1] for p in parts], axis=0)
        rows = {k: i for i, k in enumerate(self.kinds)}

        def term(x: mx.array, kind: str) -> mx.array | None:
            i = rows.get(kind)
            return None if i is None else x[i : i + 1]

        return (
            self.video_g.combine(
                v0[0:1], term(v0, "neg"), term(v0, "stg"), term(v0, "mod")
            ),
            self.audio_g.combine(
                a0[0:1], term(a0, "neg"), term(a0, "stg"), term(a0, "mod")
            ),
        )


class TiledDenoiser:
    """Upstream TiledDiffusionModel for the DFR spatial epilogue: one x0 per
    spatial tile of the window (every frame, the tile's cells widened by the
    overlap), blended back with separable trapezoid weights; a conditioning
    token rides in every tile its cell extent overlaps and is averaged over
    them. `extents` holds each token's [y0, y1) x [x0, x1) in target cells
    (generated and keyframe tokens: their cell; reference tokens: their cell
    times the downscale). Positions are shifted so the tile's generated
    tokens start at 0 in height and width, as upstream normalizes them."""

    def __init__(
        self,
        inner: Denoiser,
        f: int,
        h: int,
        w: int,
        extents: np.ndarray,
        tiles: list[sampling.SpatialTile],
    ):
        self.inner, self.f, self.h, self.w = inner, f, h, w
        self.n_gen = f * h * w
        self.extents = extents  # (N, 4): y0, y1, x0, x1
        self.tiles = tiles
        n = extents.shape[0]
        keep = np.zeros((len(tiles), n), dtype=bool)
        for i, t in enumerate(tiles):
            keep[i] = (
                (extents[:, 0] < t.h1)
                & (extents[:, 1] > t.h0)
                & (extents[:, 2] < t.w1)
                & (extents[:, 3] > t.w0)
            )
        count = keep.sum(axis=0).astype(np.float32)
        self.plans = []
        for i, t in enumerate(tiles):
            idx = np.nonzero(keep[i])[0]
            weight = np.empty(len(idx), dtype=np.float32)
            gen = idx < self.n_gen
            yy = (idx[gen] // w) % h
            xx = idx[gen] % w
            weight[gen] = t.weights[yy - t.h0, xx - t.w0]
            weight[~gen] = 1.0 / count[idx[~gen]]
            self.plans.append(
                (mx.array(idx.astype(np.int32)), mx.array(weight)[None, :, None], t)
            )

    def __call__(self, video, audio, vx, ax, sigma, step_cache=None):
        n = vx.shape[1]
        out_v = mx.zeros_like(vx)
        out_a = None
        for idx, weight, t in self.plans:

            def take(z, idx=idx):
                return None if z is None else mx.take(z, idx, axis=1)

            pos = take(video.positions) - mx.array([0.0, t.h0 * 32.0, t.w0 * 32.0])
            sub = sampling.LatentState(
                latent=take(video.latent),
                clean=take(video.clean),
                denoise_mask=take(video.denoise_mask),
                positions=pos,
                keyframes_mask=take(video.keyframes_mask),
                frozen=video.frozen,
            )
            v0, a0 = self.inner(sub, audio, take(vx), ax, sigma)
            out_v = out_v + mx.zeros((1, n, vx.shape[2]), dtype=vx.dtype).at[
                :, idx
            ].add(v0 * weight)
            out_a = a0 if out_a is None else out_a + a0
            mx.eval(out_v)
        return out_v, out_a / len(self.plans)


@dataclass(frozen=True)
class Fast:
    """The fast tier: output-changing settings a profile may stack on a
    pipeline. Every field's default is the exact pipeline.

    stage1_sigmas / stage2_sigmas: the distilled (and DFR) schedules; dev
      stage 2 takes stage2_sigmas, its stage 1 is `steps`.
    steps: dev stage-1 step count (the reference: 30).
    guidance: dev Guidance overrides applied to video and audio (a neutral
      scale drops that pass from the batch: modality 1.0 saves one of four).
    step_cache: first-block cache threshold (accumulated relative L1 of the
      block-0 residual; 0 = off). Applied to every sampler loop.
    attention_tiles: (rows, cols) spatial tiles for the stage-2 video
      self-attention, keys widened by attention_halo latent cells.
    """

    stage1_sigmas: tuple[float, ...] | None = None
    stage2_sigmas: tuple[float, ...] | None = None
    steps: int | None = None
    guidance: dict[str, float] | None = None
    step_cache: float = 0.0
    attention_tiles: tuple[int, int] | None = None
    attention_halo: int = 2

    @classmethod
    def from_config(cls, cfg: dict | None) -> Fast:
        if not cfg:
            return cls()
        out = dict(cfg)
        for key in ("stage1_sigmas", "stage2_sigmas", "attention_tiles"):
            if out.get(key) is not None:
                out[key] = tuple(out[key])
        unknown = set(out) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown fast settings: {sorted(unknown)}")
        return cls(**out)

    @property
    def active(self) -> bool:
        return self != Fast()

    def cache(self) -> StepCache | None:
        return StepCache(self.step_cache) if self.step_cache > 0 else None

    def tiles(self, f: int, h: int, w: int, total: int | None = None):
        if self.attention_tiles is None:
            return None
        return attention_tiles(
            f, h, w, self.attention_tiles, self.attention_halo, total
        )

    def sigmas1(self, default: list[float]) -> list[float]:
        return list(self.stage1_sigmas) if self.stage1_sigmas else default

    def sigmas2(self, default: list[float]) -> list[float]:
        return list(self.stage2_sigmas) if self.stage2_sigmas else default


class LTX25Engine:
    """Resident components plus the pipelines that run on them."""

    def __init__(
        self,
        root: Path | None = None,
        variant: str = "distilled",
        bf16_noise: bool = False,
        decoder: str = "diffusion",
    ):
        self.root = root or checkpoints.model_root()
        # Metal memory is wired: it cannot be compressed or swapped, and MLX's
        # buffer cache counts. Active + cache is held OS_RESERVE below RAM.
        total = int(mx.device_info()["memory_size"])
        mx.set_cache_limit(CACHE_BYTES)
        mx.set_memory_limit(total - OS_RESERVE_BYTES - CACHE_BYTES)
        self.variant = variant
        self.bf16_noise = bf16_noise
        self.dit: LTX25DiT | None = None
        self.text = None
        self.vae = None
        self.upscaler = None
        self.audio = None
        self.distilled_lora = None
        self.detail_lora = None
        self.diffvae = None
        self.duration = None
        self.enhancer = None
        self.temporal_upscaler = None
        self.decoder = decoder

    # ---- components -------------------------------------------------------
    def load_text(self):
        if self.text is None:
            from slimserve.video.ltx25.text import TextEncoder

            self.text = TextEncoder(self.root, variant=self.variant)
            self.text.load()
        return self.text

    def load_dit(self) -> LTX25DiT:
        if self.dit is None:
            weights, _connectors, cfg = checkpoints.load_dit(self.variant, self.root)
            self.dit = LTX25DiT(weights, DiTConfig.from_checkpoint(cfg))
        return self.dit

    def load_vae(self):
        if self.vae is None:
            from slimserve.video.ltx25.vae import VideoVAE

            self.vae = VideoVAE(self.root)
            self.vae.load()
        return self.vae

    def load_upscaler(self):
        if self.upscaler is None:
            from slimserve.video.ltx25.upscaler import LatentUpscaler

            self.upscaler = LatentUpscaler("spatial", self.root)
            self.upscaler.load()
        return self.upscaler

    def load_audio(self):
        if self.audio is None:
            from slimserve.video.ltx25.audio import AudioDecoder

            self.audio = AudioDecoder(self.root)
            self.audio.load()
        return self.audio

    def load_duration(self):
        if self.duration is None:
            from slimserve.video.ltx25.duration import DurationHead

            self.duration = DurationHead(self.root).load()
        return self.duration

    def load_enhancer(self):
        if self.enhancer is None:
            from slimserve.video.ltx25.enhancer import PromptEnhancer

            self.enhancer = PromptEnhancer(self.root).load()
        return self.enhancer

    def enhance(self, prompt: str, image: str | bytes | None = None) -> str:
        """Upstream's --enhance-prompt: Gemma-4 E2B-it rewrites the request into
        the caption style the model was trained on (~4.5 GiB while loaded, ~1 s
        to load, ~5 s per rewrite). With the conditioning still it describes
        what it sees (enhance_i2v)."""
        return self.load_enhancer().enhance(prompt, image)

    def unload_enhancer(self) -> None:
        if self.enhancer is not None:
            self.enhancer.unload()
            self.enhancer = None

    def resolve_frames(
        self,
        num_frames: int | None,
        video_text: mx.array,
        audio_text: mx.array,
        fps: float,
        max_num_frames: int | None,
    ) -> tuple[int, float | None]:
        """A request without a frame count gets the duration head's prediction
        (upstream resolve_num_frames: clamped to 1-20 s, snapped to 8k + 1),
        further capped at `max_num_frames` (the profile's validated envelope at
        the clip's size)."""
        if num_frames is not None:
            return num_frames, None
        frames, seconds = self.load_duration().num_frames(video_text, audio_text, fps)
        if max_num_frames is not None:
            frames = min(frames, max_num_frames)
        return frames, seconds

    def unload_text(self) -> None:
        if self.text is not None:
            self.text.unload()
            self.text = None

    def unload_dit(self) -> None:
        """Drop the transformer (44 GiB fp16); load_dit() reloads it in ~6 s."""
        if self.dit is not None:
            self.dit = None
            mx.clear_cache()

    def decode_budget(self) -> int:
        """Bytes the VAE decode may use: what is left after the resident models
        and the OS reserve. The decoder tiles to fit. Metal memory is wired, so
        overshooting this takes the machine down, not just the process."""
        total = int(mx.device_info()["memory_size"])
        free = total - mx.get_active_memory() - OS_RESERVE_BYTES - CACHE_BYTES
        return int(max(6 << 30, min(free, total // 2)))

    @staticmethod
    def _stepper(tm: Timings, on_step):
        def stepper(stage: str):
            t_last = [time.perf_counter()]

            def cb(i: int, sigma: float) -> None:
                mx.synchronize()
                now = time.perf_counter()
                tm.steps.append((stage, i, now - t_last[0]))
                t_last[0] = now
                if on_step:
                    on_step(stage, i, sigma)

            return cb

        return stepper

    # ---- image conditioning (I2V, keyframes) ----------------------------------
    @staticmethod
    def _prepare_images(
        image: str | bytes | None,
        image_strength: float,
        images: list[sampling.Still] | None,
        num_frames: int,
    ) -> list[tuple[np.ndarray, int, float]]:
        """Decode and CRF-round-trip every still once: (uint8 pixels, pixel
        frame, strength) in request order. `image` is the one-still shorthand
        (frame 0). Frames outside the clip are refused as upstream's
        assert_image_frames_in_clip."""
        from slimserve.video.ltx25 import image as image_mod

        stills = list(images or [])
        if image is not None:
            stills.insert(0, sampling.Still(image, 0, image_strength))
        out = []
        for still in stills:
            if not 0 <= still.frame < num_frames:
                raise ValueError(
                    f"image frame {still.frame} is outside the clip's "
                    f"{num_frames} frames"
                )
            if not 0.0 <= still.strength <= 1.0:
                raise ValueError("image strength must be in [0, 1]")
            crf = image_mod.DEFAULT_IMAGE_CRF if still.crf is None else still.crf
            out.append(
                (
                    image_mod.prepare_image(still.image, crf=crf),
                    still.frame,
                    still.strength,
                )
            )
        return out

    def _image_latents(
        self, prepared: list[tuple[np.ndarray, int, float]], h: int, w: int, tm: Timings
    ) -> list[tuple[mx.array, int, float]]:
        """Encode each prepared still at one stage's pixel size (h*32 x w*32)
        through the conv VAE encoder: normalized (1, 128, 1, h, w). Upstream
        re-encodes per stage (chunks/conditionings.py image_conditionings_for_chunk
        at the chunk's _pixel_hw), so both stages are conditioned. Called outside
        the stage spans: a span resets the peak-memory counter on entry."""
        if not prepared:
            return []
        from slimserve.video.ltx25 import image as image_mod

        out = []
        with tm.span("image"):
            vae = self.load_vae()
            for pixels, frame, strength in prepared:
                latent = vae.encode(
                    image_mod.conditioning_frame(pixels, h * 32, w * 32)
                )
                mx.eval(latent)
                out.append((latent, frame, strength))
        return out

    @staticmethod
    def _condition(
        video: LatentState,
        conds: list[tuple[mx.array, int, float]],
        fps: float = 24.0,
        sigma: float = 1.0,
        seed: int = 0,
        append_all: bool = False,
    ) -> LatentState:
        """Apply the encoded stills (see `sampling.condition_stills`): frame 0
        pinned into latent frame 0, other frames appended as keyframe tokens
        (noised at `sigma` from `seed` where their strength is below 1)."""
        if not conds:
            return video
        video = sampling.condition_stills(video, conds, fps, sigma, seed, append_all)
        mx.eval(video.latent, video.clean, video.denoise_mask)
        return video

    # ---- distilled --------------------------------------------------------
    def distilled(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1536,
        num_frames: int | None = 121,
        fps: float = 24.0,
        seed: int = 42,
        text_embeds: tuple[mx.array, mx.array] | None = None,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        image: str | bytes | None = None,
        image_strength: float = 1.0,
        fast: Fast | None = None,
        max_num_frames: int | None = None,
        images: list[sampling.Still] | None = None,
    ) -> Result:
        """`image` (a path or encoded image bytes) conditions the first frame
        (image-to-video) at `image_strength` in both stages; `images` are
        further stills at any pixel frame (upstream's repeatable --image: frame
        0 replaces latent frame 0, other frames become appended keyframe
        tokens). `fast` stacks the output-changing fast tier (see `Fast`).
        `num_frames` None: the duration head decides (see `resolve_frames`)."""
        fast = fast or Fast()
        tm = Timings()
        if text_embeds is None:
            with tm.span("text"):
                video_text, audio_text = self.load_text().encode(prompt)[:2]
                mx.eval(video_text, audio_text)
            if not keep_text:
                self.unload_text()
        else:
            video_text, audio_text = text_embeds
        num_frames, predicted = self.resolve_frames(
            num_frames, video_text, audio_text, fps, max_num_frames
        )
        with tm.span("load"):
            dit = self.load_dit()
            vae = self.load_vae()
            upscaler = self.load_upscaler()
        denoise = Denoiser(dit, video_text, audio_text)

        height, width = sampling.snap_dimensions(height, width, two_stage=True)
        f, h1, w1 = sampling.video_latent_shape(num_frames, height // 2, width // 2)
        audio_t = sampling.audio_token_count(num_frames, fps)
        apos = sampling.audio_positions(audio_t)

        stepper = self._stepper(tm, on_step)
        stills = self._prepare_images(image, image_strength, images, num_frames)
        cond1 = self._image_latents(stills, h1, w1, tm)

        # Stage 1: half resolution, from pure noise, ancestral Euler.
        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h1 * w1, 128),
                sampling.video_positions(f, h1, w1, fps),
                seed,
                tokens_per_frame=h1 * w1,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond1, fps, 1.0, seed)
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise
            )
            v1, a1 = sampling.euler_ancestral_loop(
                denoise,
                video,
                audio,
                fast.sigmas1(sampling.DISTILLED_SIGMAS),
                noise_seed=seed + sampling.ANCESTRAL_NOISE_SEED_OFFSET,
                on_step=stepper("stage1"),
                step_cache=fast.cache(),
            )

        # 2x latent upscale in the VAE's denormalized latent space.
        with tm.span("upscale"):
            half = sampling.unpatchify(v1[:, : f * h1 * w1], (f, h1, w1))
            up = vae.normalize(upscaler(vae.denormalize(half)))
            mx.eval(up)
        h2, w2 = h1 * 2, w1 * 2
        cond2 = self._image_latents(stills, h2, w2, tm)

        # Stage 2: full resolution refinement from sigma 0.909; ancestral too on
        # 2.5 checkpoints (upstream distilled.py), from its own noise offset.
        with tm.span("stage2"):
            sigmas2 = fast.sigmas2(sampling.STAGE_2_DISTILLED_SIGMAS)
            s0 = sigmas2[0]
            video = sampling.noised_state(
                (1, f * h2 * w2, 128),
                sampling.video_positions(f, h2, w2, fps),
                seed + 2,
                sigma=s0,
                initial=sampling.patchify(up),
                tokens_per_frame=h2 * w2,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond2, fps, s0, seed + 2)
            denoise = Denoiser(
                dit,
                video_text,
                audio_text,
                video_tiles=fast.tiles(f, h2, w2, video.latent.shape[1]),
            )
            audio = sampling.noised_state(
                (1, audio_t, 128),
                apos,
                seed + 2,
                sigma=s0,
                initial=a1,
                bf16_noise=self.bf16_noise,
            )
            v2, a2 = sampling.euler_ancestral_loop(
                denoise,
                video,
                audio,
                sigmas2,
                noise_seed=seed + sampling.ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET,
                on_step=stepper("stage2"),
                step_cache=fast.cache(),
            )

        return Result(
            sampling.unpatchify(v2[:, : f * h2 * w2], (f, h2, w2)),
            a2,
            num_frames,
            height,
            width,
            fps,
            tm,
            predicted_seconds=predicted,
        )

    # ---- DFR --------------------------------------------------------------
    def load_detail_lora(self):
        if self.detail_lora is None:
            from slimserve.video.ltx25.lora import Lora

            self.detail_lora = Lora("detail-lora", self.root).load()
        return self.detail_lora

    def dfr(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1536,
        num_frames: int | None = 121,
        fps: float = 24.0,
        seed: int = 42,
        detail_strength: float = 0.5,
        text_embeds: tuple[mx.array, mx.array] | None = None,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        image: str | bytes | None = None,
        image_strength: float = 1.0,
        fast: Fast | None = None,
        max_num_frames: int | None = None,
        temporal_upscalings: int = 0,
        spatial_upscalings: int = 1,
        images: list[sampling.Still] | None = None,
    ) -> Result:
        """Lightricks' DFRPipeline, default configuration: the distilled flow
        with one generated keyframe slot per 24/32-frame segment in stage 1,
        then a stage 2 that runs under the detailing IC-LoRA with the upscaled
        slots and the half-resolution stage-1 latent as reference tokens.
        `image` / `images` condition the clip in both stages and in every later
        window a still falls into (see `distilled`; frames are rebased by
        2**rounds after temporal rounds, as upstream rebase_image_conditionings).
        `temporal_upscalings` (0-2): each round doubles the frame rate with the
        temporal upscaler and re-denoises the canvas in keyframe-seam windows
        (see `_temporal_round`); the clip ships at fps * 2**rounds.
        `spatial_upscalings` 2 runs stage 1 at a quarter and stage 2 at half of
        the output size and adds the tiled full-resolution detailing epilogue
        (see `_spatial_epilogue`); sizes must then be multiples of 128. `fast`
        stacks the fast tier (see `Fast`)."""
        if self.variant != "distilled":
            raise ValueError("the DFR pipeline runs on the distilled transformer")
        if temporal_upscalings not in (0, 1, 2):
            raise ValueError("temporal_upscalings must be 0, 1 or 2")
        if spatial_upscalings not in (1, 2):
            raise ValueError("spatial_upscalings must be 1 or 2")
        fast = fast or Fast()
        tm = Timings()
        if text_embeds is None:
            with tm.span("text"):
                video_text, audio_text = self.load_text().encode(prompt)[:2]
                mx.eval(video_text, audio_text)
                if not keep_text:
                    self.unload_text()
        else:
            video_text, audio_text = text_embeds
        num_frames, predicted = self.resolve_frames(
            num_frames, video_text, audio_text, fps, max_num_frames
        )
        with tm.span("load"):
            dit = self.load_dit()
            vae = self.load_vae()
            upscaler = self.load_upscaler()
            lora = self.load_detail_lora()
        denoise = Denoiser(dit, video_text, audio_text)
        stepper = self._stepper(tm, on_step)

        div = 2**spatial_upscalings
        m = 32 * div
        height, width = max(m, height // m * m), max(m, width // m * m)
        canvas, _segment, slot_frames = sampling.dfr_canvas(num_frames)
        cfps = sampling.conditioning_fps(fps)
        f, h1, w1 = sampling.video_latent_shape(canvas, height // div, width // div)
        audio_t = sampling.audio_token_count(canvas, fps)
        apos = sampling.audio_positions(audio_t)
        k = len(slot_frames)
        stills = self._prepare_images(image, image_strength, images, num_frames)
        cond1 = self._image_latents(stills, h1, w1, tm)

        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h1 * w1, 128),
                sampling.video_positions(f, h1, w1, cfps),
                seed,
                tokens_per_frame=h1 * w1,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond1, cfps, 1.0, seed)
            video, slots = sampling.append_slots(
                video, slot_frames, h1, w1, cfps, None, 1.0, seed
            )
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise
            )
            v1, a1 = sampling.euler_ancestral_loop(
                denoise,
                video,
                audio,
                fast.sigmas1(sampling.DISTILLED_SIGMAS),
                noise_seed=seed + sampling.ANCESTRAL_NOISE_SEED_OFFSET,
                on_step=stepper("stage1"),
                step_cache=fast.cache(),
            )

        with tm.span("upscale"):
            half = sampling.unpatchify(v1[:, : f * h1 * w1], (f, h1, w1))
            up = vae.normalize(upscaler(vae.denormalize(half)))
            slots_up = vae.normalize(
                upscaler(
                    vae.denormalize(sampling.slots_to_latent(v1, slots, k, h1, w1))
                )
            )
            mx.eval(up, slots_up)
        h2, w2 = h1 * 2, w1 * 2
        cond2 = self._image_latents(stills, h2, w2, tm)

        with tm.span("stage2"):
            sigmas2 = fast.sigmas2(sampling.STAGE_2_DISTILLED_SIGMAS)
            s0 = sigmas2[0]
            video = sampling.noised_state(
                (1, f * h2 * w2, 128),
                sampling.video_positions(f, h2, w2, cfps),
                seed + 2,
                sigma=s0,
                initial=sampling.patchify(up),
                tokens_per_frame=h2 * w2,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond2, cfps, s0, seed + 2)
            video, slots2 = sampling.append_slots(
                video, slot_frames, h2, w2, cfps, slots_up, s0, seed + 2
            )
            video = sampling.append_reference(
                video,
                sampling.patchify(half),
                sampling.video_positions(f, h1, w1, cfps),
                lora.reference_downscale,
            )
            audio = sampling.noised_state(
                (1, audio_t, 128),
                apos,
                seed + 2,
                sigma=s0,
                initial=a1,
                bf16_noise=self.bf16_noise,
            )
            denoise = Denoiser(
                dit,
                video_text,
                audio_text,
                video_tiles=fast.tiles(f, h2, w2, video.latent.shape[1]),
            )
            lora.attach(dit, detail_strength)
            try:
                v2, _ = sampling.euler_ancestral_loop(
                    denoise,
                    video,
                    audio,
                    sigmas2,
                    noise_seed=seed + sampling.ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET,
                    on_step=stepper("stage2"),
                    step_cache=fast.cache(),
                )
            finally:
                lora.detach(dit)

        canvas_latent = sampling.unpatchify(v2[:, : f * h2 * w2], (f, h2, w2))
        slot_planes = sampling.slots_to_latent(v2, slots2, k, h2, w2)
        plane_at = {
            fr: slot_planes[:, :, i : i + 1] for i, fr in enumerate(slot_frames)
        }
        canvas_frames, playback_fps = canvas, fps
        source_duration = canvas / fps
        for round_idx in range(1, temporal_upscalings + 1):
            canvas_frames = 2 * (canvas_frames - 1) + 1
            playback_fps *= 2.0
            with tm.span(f"temporal{round_idx}"):
                canvas_latent, plane_at = self._temporal_round(
                    round_idx,
                    Denoiser(dit, video_text, audio_text),  # plain stage, no LoRA
                    vae,
                    canvas_latent,
                    plane_at,
                    canvas_frames,
                    playback_fps,
                    h2,
                    w2,
                    a1,
                    source_duration,
                    seed,
                    stills,
                    tm,
                    stepper(f"temporal{round_idx}"),
                )
        if spatial_upscalings == 2:
            with tm.span("epilogue"):
                canvas_latent, plane_at = self._spatial_epilogue(
                    dit,
                    lora,
                    vae,
                    video_text,
                    audio_text,
                    canvas_latent,
                    plane_at,
                    canvas_frames,
                    playback_fps,
                    temporal_upscalings,
                    a1,
                    source_duration,
                    seed,
                    stills,
                    detail_strength,
                    tm,
                    stepper("epilogue"),
                )
        # trim the canvas padding back off: (requested - 1) * 2**rounds + 1 frames
        num_frames = (num_frames - 1) * 2**temporal_upscalings + 1
        keep = (num_frames - 1) // 8 + 1
        latent = canvas_latent[:, :, :keep]
        # the denoised slots anchor the decode (upstream DFR always keyframe-
        # decodes); slots past the trimmed clip are dropped
        kept = sorted(fr for fr in plane_at if fr < num_frames)
        keyframes = None
        if kept:
            keyframes = (
                mx.concatenate([plane_at[fr] for fr in kept], axis=2),
                kept,
            )
            mx.eval(keyframes[0])
        return Result(
            latent,
            a1[:, : sampling.audio_token_count(num_frames, playback_fps)],
            num_frames,
            height,
            width,
            playback_fps,
            tm,
            keyframes=keyframes,
            predicted_seconds=predicted,
        )

    def load_temporal_upscaler(self):
        if self.temporal_upscaler is None:
            from slimserve.video.ltx25.upscaler import LatentUpscaler

            self.temporal_upscaler = LatentUpscaler("temporal", self.root)
            self.temporal_upscaler.load()
        return self.temporal_upscaler

    def _temporal_round(
        self,
        round_idx: int,
        denoise: Denoiser,
        vae,
        canvas_latent: mx.array,
        plane_at: dict[int, mx.array],
        canvas_frames: int,
        playback_fps: float,
        h: int,
        w: int,
        audio_tokens: mx.array,
        source_duration: float,
        seed: int,
        stills: list[tuple[np.ndarray, int, float]],
        tm: Timings,
        on_step,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        """Upstream run_one_temporal_round: x2 temporal upsample of the canvas
        (the keyframe planes move to 2x their pixel frames), then 2**round
        windows cut on the keyframe seams, each re-denoised from sigma 0.975
        (DISTILLED_SIGMAS[4:], ancestral, eta 0.5) as its own clip: the window's
        seams are pinned as near-clean anchor keyframes (strength 0.95), each
        segment midpoint gets a generated slot seeded from the nearest cell, a
        non-first window starts on the last plane before its seam with the
        previous window's cells up to the seam held fixed (the lead-in), and
        its stage-1 audio window rides along frozen. Owned cells are stitched;
        new slot planes join the carry bag for the next round and the decode."""
        up = vae.normalize(
            self.load_temporal_upscaler()(vae.denormalize(canvas_latent))
        )
        mx.eval(up)
        plane_at = {2 * p: plane for p, plane in plane_at.items()}
        seams = sorted(plane_at)
        tiles = sampling.temporal_tile_plan(seams, canvas_frames, 2**round_idx)
        cond_fps = sampling.conditioning_fps(playback_fps)
        sigmas = sampling.DISTILLED_SIGMAS[4:]
        s0 = sigmas[0]
        encoded_stills = self._image_latents(stills, h, w, tm)
        owned: list[mx.array] = []
        previous: tuple[int, mx.array] | None = None
        for tile_index, tile in enumerate(tiles):
            prefix = None
            if tile_index > 0:
                prefix = sampling.tile_prefix((tile.start - 1) * 8, plane_at)
            if prefix is None:
                tile_latent = up[:, :, tile.start : tile.end]
                pixel_start, pinned = 0, 0
            else:
                tile_latent = mx.concatenate(
                    [
                        plane_at[prefix.keyframe_position],
                        up[:, :, prefix.video_start_cell : tile.end],
                    ],
                    axis=2,
                )
                pixel_start, pinned = prefix.keyframe_position, prefix.cells
            ft = tile_latent.shape[2]
            local_frames = (ft - 1) * 8 + 1
            tile_seed = seed + 1000 * round_idx + tile_index
            video = sampling.noised_state(
                (1, ft * h * w, 128),
                sampling.video_positions(ft, h, w, cond_fps),
                tile_seed,
                sigma=s0,
                initial=sampling.patchify(tile_latent),
                tokens_per_frame=h * w,
                bf16_noise=self.bf16_noise,
            )
            # stills inside this window, at 2**round their requested frame
            # (upstream _temporal_tile_conditionings / rebase_image_conditionings)
            video = self._condition(
                video,
                sampling.rebase_stills(
                    encoded_stills,
                    2**round_idx,
                    pixel_start,
                    pixel_start + local_frames - 1,
                ),
                cond_fps,
                s0,
                tile_seed,
            )
            resume = prefix.resume_pixel if prefix else 0
            anchors = [p for p in tile.anchors if p >= resume]
            if anchors:
                video = sampling.append_anchor_keyframes(
                    video,
                    mx.concatenate([plane_at[p] for p in anchors], axis=2),
                    [p - pixel_start for p in anchors],
                    h,
                    w,
                    cond_fps,
                    s0,
                    tile_seed,
                )
            slots = None
            if tile.slots:
                local = [p - pixel_start for p in tile.slots]
                initial = mx.concatenate(
                    [
                        tile_latent[:, :, min(max(round(p / 8), 0), ft - 1)][:, :, None]
                        for p in local
                    ],
                    axis=2,
                )
                video, slots = sampling.append_slots(
                    video, local, h, w, cond_fps, initial, s0, tile_seed
                )
            if prefix is not None and previous is not None:
                base_cell, prev_latent = previous
                cells = prefix.cells - 1
                offset = prefix.video_start_cell - base_cell
                carried = mx.concatenate(
                    [
                        plane_at[prefix.keyframe_position],
                        prev_latent[:, :, offset : offset + cells],
                    ],
                    axis=2,
                )
                video = sampling.condition_latent_frame(video, carried, 1.0, 0)
            mx.eval(video.latent, video.clean, video.denoise_mask)
            audio_t = sampling.audio_token_count(local_frames, cond_fps)
            a = sampling.audio_tokens_for_tile(
                audio_tokens,
                pixel_start,
                local_frames,
                playback_fps,
                source_duration,
                cond_fps,
            )
            audio = sampling.LatentState(
                latent=a,
                clean=a,
                denoise_mask=mx.zeros((1, audio_t, 1), dtype=G),
                positions=sampling.audio_positions(audio_t),
                frozen=True,
            )
            vt, _ = sampling.euler_ancestral_loop(
                denoise,
                video,
                audio,
                sigmas,
                noise_seed=tile_seed,
                eta=sampling.TEMPORAL_ANCESTRAL_ETA,
                on_step=on_step,
            )
            tile_out = sampling.unpatchify(vt[:, : ft * h * w], (ft, h, w))
            mx.eval(tile_out)
            owned.append(tile_out[:, :, pinned:])
            previous = (prefix.video_start_cell - 1 if prefix else tile.start, tile_out)
            if slots is not None:
                planes = sampling.slots_to_latent(vt, slots, len(tile.slots), h, w)
                for i, p in enumerate(tile.slots):
                    plane_at.setdefault(p, planes[:, :, i : i + 1])
        stitched = mx.concatenate(owned, axis=2)
        if stitched.shape[2] != (canvas_frames - 1) // 8 + 1:
            raise RuntimeError("temporal round stitched the wrong number of cells")
        mx.eval(stitched)
        return stitched, dict(sorted(plane_at.items()))

    def _rebuild_keyframes(
        self, planes: dict[int, mx.array], seed: int
    ) -> dict[int, mx.array]:
        """Upstream _rebuild_epilogue_keyframes: decode each carry plane as its
        own one-frame clip, Lanczos x2 in RGB, encode back as a one-frame
        latent at the epilogue's resolution."""
        from PIL import Image

        decoder = self.load_diffvae()
        vae = self.load_vae()
        out = {}
        for i, (pos, plane) in enumerate(planes.items()):
            px = decoder.decode_raw(
                plane, seed=seed + sampling.EPILOGUE_KEYFRAME_DECODE_SEED_OFFSET + i
            )
            rgb = np.array(
                mx.clip((px[0, :, 0] + 1.0) * 0.5, 0.0, 1.0).transpose(1, 2, 0)
            )
            img = Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")
            img = img.resize(
                (img.width * 2, img.height * 2), resample=Image.Resampling.LANCZOS
            )
            big = np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
            pixels = mx.array(big).transpose(2, 0, 1)[None, :, None]  # (1, 3, 1, H, W)
            out[pos] = vae.encode(pixels)
            mx.eval(out[pos])
            del px
        return out

    def _spatial_epilogue(
        self,
        dit: LTX25DiT,
        lora,
        vae,
        video_text: mx.array,
        audio_text: mx.array,
        canvas_latent: mx.array,
        plane_at: dict[int, mx.array],
        canvas_frames: int,
        playback_fps: float,
        temporal_upscalings: int,
        audio_tokens: mx.array,
        source_duration: float,
        seed: int,
        stills: list[tuple[np.ndarray, int, float]],
        detail_strength: float,
        tm: Timings,
        on_step,
    ) -> tuple[mx.array, dict[int, mx.array]]:
        """Upstream run_spatial_epilogue (spatial_upscalings 2): the stage-2
        canvas is x2 spatially upsampled and re-detailed at the output size
        under the detailing IC-LoRA with the stage-2 latent as reference tokens,
        in keyframe-seam temporal windows (one window without temporal rounds;
        upstream's plan cannot express that and fails), each window first one
        Euler step from sigma 0.909 on 2x2 spatial tiles then the remaining
        steps on 4x4 tiles (overlap 10 cells, trapezoid blend), its carry
        keyframes pinned clean (decoded, Lanczos x2, re-encoded), a generated
        opening plane pinned at frame 0 when no still sits at frame 0, a
        non-first window starting on the plane before its seam with the lead-in
        pinned. Returns the full-resolution canvas latent and the rebuilt planes."""
        _, _, t_cells, h2, w2 = canvas_latent.shape
        h, w = 2 * h2, 2 * w2
        scale = 2**temporal_upscalings
        to_decode = dict(plane_at)
        if not any(frame * scale == 0 for _, frame, _ in stills):
            to_decode[-1] = canvas_latent[:, :, :1]
        encoded = self._rebuild_keyframes(to_decode, seed)
        opening = encoded.pop(-1, None)
        guide = canvas_latent
        up = vae.normalize(self.load_upscaler()(vae.denormalize(guide)))
        mx.eval(up)
        if temporal_upscalings == 0:
            windows = [sampling.TemporalTile(0, t_cells, 0, (t_cells - 1) * 8, (), ())]
        else:
            windows = sampling.temporal_tile_plan(
                sorted(plane_at), canvas_frames, 2**temporal_upscalings
            )
        cond_fps = sampling.conditioning_fps(playback_fps)
        sigmas = sampling.STAGE_2_DISTILLED_SIGMAS
        phases = [(sigmas[:2], sigmas[0], sampling.EPILOGUE_SPATIAL_COARSE_TILES)]
        if len(sigmas) > 2:
            phases.append((sigmas[1:], 0.0, sampling.EPILOGUE_SPATIAL_TILES))
        encoded_stills = self._image_latents(stills, h, w, tm)
        downscale = lora.reference_downscale
        stitched: list[mx.array] = []
        previous: tuple[int, mx.array] | None = None
        lora.attach(dit, detail_strength)
        try:
            for index, win in enumerate(windows):
                prefix = None
                if index > 0:
                    prefix = sampling.tile_prefix((win.start - 1) * 8, encoded)
                if prefix is None:
                    latent_in = up[:, :, win.start : win.end]
                    ref = guide[:, :, win.start : win.end]
                    origin, resume, pinned = win.pixel_start, 0, 0
                else:
                    latent_in = mx.concatenate(
                        [
                            encoded[prefix.keyframe_position],
                            up[:, :, prefix.video_start_cell : win.end],
                        ],
                        axis=2,
                    )
                    ref = guide[:, :, prefix.video_start_cell - 1 : win.end]
                    origin, resume, pinned = (
                        prefix.keyframe_position,
                        prefix.resume_pixel,
                        prefix.cells,
                    )
                ft = latent_in.shape[2]
                local_frames = (ft - 1) * 8 + 1
                audio_t = sampling.audio_token_count(local_frames, cond_fps)
                a = sampling.audio_tokens_for_tile(
                    audio_tokens,
                    origin,
                    local_frames,
                    playback_fps,
                    source_duration,
                    cond_fps,
                )
                audio = sampling.LatentState(
                    latent=a,
                    clean=a,
                    denoise_mask=mx.zeros((1, audio_t, 1), dtype=G),
                    positions=sampling.audio_positions(audio_t),
                    frozen=True,
                )
                window_seed = seed + 100 * index
                anchors = [
                    (pos - origin, plane)
                    for pos, plane in encoded.items()
                    if origin <= pos <= win.pixel_end and pos >= resume
                ]
                if opening is not None and prefix is None:
                    anchors.append((0, opening))
                latent = latent_in
                for phase_sigmas, noise_scale, n_tiles in phases:
                    video = sampling.noised_state(
                        (1, ft * h * w, 128),
                        sampling.video_positions(ft, h, w, cond_fps),
                        window_seed + sampling.EPILOGUE_NOISE_SEED_OFFSET,
                        sigma=noise_scale,
                        initial=sampling.patchify(latent),
                        tokens_per_frame=h * w,
                        bf16_noise=self.bf16_noise,
                    )
                    yy, xx = np.divmod(np.arange(ft * h * w) % (h * w), w)
                    extents = [
                        np.stack([yy, yy + 1, xx, xx + 1], axis=1).astype(np.float32)
                    ]
                    # stills inside this window (upstream
                    # _encode_epilogue_window_images: rebased by 2**rounds,
                    # window-local, none before the resume pixel)
                    window_stills = sampling.rebase_stills(
                        encoded_stills, scale, origin, win.pixel_end, resume
                    )
                    before = video.latent.shape[1]
                    video = self._condition(
                        video,
                        window_stills,
                        cond_fps,
                        noise_scale,
                        window_seed + sampling.EPILOGUE_NOISE_SEED_OFFSET,
                    )
                    for _ in range((video.latent.shape[1] - before) // (h * w)):
                        extents.append(extents[0][: h * w])
                    for local, plane in anchors:
                        video = sampling.append_anchor_keyframes(
                            video,
                            plane,
                            [local],
                            h,
                            w,
                            cond_fps,
                            noise_scale,
                            window_seed,
                            strength=sampling.EPILOGUE_KEYFRAME_STRENGTH,
                        )
                        extents.append(extents[0][: h * w])
                    video = sampling.append_reference(
                        video,
                        sampling.patchify(ref),
                        sampling.video_positions(ft, h2, w2, cond_fps),
                        downscale,
                    )
                    ry, rx = np.divmod(np.arange(ft * h2 * w2) % (h2 * w2), w2)
                    extents.append(
                        np.stack(
                            [
                                ry * downscale,
                                (ry + 1) * downscale,
                                rx * downscale,
                                (rx + 1) * downscale,
                            ],
                            axis=1,
                        ).astype(np.float32)
                    )
                    if prefix is not None and previous is not None:
                        base_cell, prev_latent = previous
                        cells = prefix.cells - 1
                        offset = prefix.video_start_cell - base_cell
                        carried = mx.concatenate(
                            [
                                encoded[prefix.keyframe_position],
                                prev_latent[:, :, offset : offset + cells],
                            ],
                            axis=2,
                        )
                        video = sampling.condition_latent_frame(video, carried, 1.0, 0)
                    mx.eval(video.latent, video.clean, video.denoise_mask)
                    tiles = sampling.spatial_tiles(
                        h, w, n_tiles, sampling.EPILOGUE_SPATIAL_OVERLAP
                    )
                    den = TiledDenoiser(
                        Denoiser(dit, video_text, audio_text),
                        ft,
                        h,
                        w,
                        np.concatenate(extents, axis=0),
                        tiles,
                    )
                    vt, _ = sampling.euler_loop(
                        den, video, audio, phase_sigmas, on_step=on_step
                    )
                    latent = sampling.unpatchify(vt[:, : ft * h * w], (ft, h, w))
                    mx.eval(latent)
                previous = (
                    prefix.video_start_cell - 1 if prefix else win.start,
                    latent,
                )
                stitched.append(latent[:, :, pinned:])
        finally:
            lora.detach(dit)
        out = mx.concatenate(stitched, axis=2)
        if out.shape[2] != t_cells:
            raise RuntimeError(
                "the spatial epilogue stitched the wrong number of cells"
            )
        mx.eval(out)
        return out, encoded

    # ---- dev --------------------------------------------------------------
    def load_distilled_lora(self):
        if self.distilled_lora is None:
            from slimserve.video.ltx25.lora import Lora

            self.distilled_lora = Lora("distilled-lora", self.root).load()
        return self.distilled_lora

    def dev(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1536,
        num_frames: int | None = 121,
        fps: float = 24.0,
        seed: int = 42,
        negative_prompt: str | None = None,
        steps: int = 30,
        video_guidance: sampling.Guidance = DEFAULT_VIDEO_GUIDANCE,
        audio_guidance: sampling.Guidance = DEFAULT_AUDIO_GUIDANCE,
        batched: bool = True,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        image: str | bytes | None = None,
        image_strength: float = 1.0,
        fast: Fast | None = None,
        max_num_frames: int | None = None,
        images: list[sampling.Still] | None = None,
        keyframes_only: bool = False,
    ) -> Result:
        """Lightricks' TI2VidTwoStagesPipeline: guided dev stage 1 at half
        resolution, 2x latent upscale, 3-step stage 2 with the distilled LoRA
        and no guidance. Audio is taken from stage 1, as upstream. `image` /
        `images` condition the clip in both stages (see `distilled`). `fast`
        stacks the fast tier (see `Fast`). `keyframes_only` is the
        KeyframeInterpolationPipeline: every still is appended as keyframe
        tokens (frame 0 too) and the audio is re-noised and refined in stage 2."""
        if self.variant != "dev":
            raise ValueError("the dev pipeline needs LTX25Engine(variant='dev')")
        fast = fast or Fast()
        if fast.steps:
            steps = fast.steps
        if fast.guidance:
            video_guidance = replace(video_guidance, **fast.guidance)
            audio_guidance = replace(audio_guidance, **fast.guidance)
        tm = Timings()
        with tm.span("text"):
            text = self.load_text()
            cond = text.encode(prompt)[:2]
            neg = text.encode(
                sampling.DEFAULT_NEGATIVE_PROMPT
                if negative_prompt is None
                else negative_prompt
            )[:2]
            mx.eval(cond, neg)
            num_frames, predicted = self.resolve_frames(
                num_frames, cond[0], cond[1], fps, max_num_frames
            )
            if not keep_text:
                self.unload_text()
        with tm.span("load"):
            dit = self.load_dit()
            vae = self.load_vae()
            upscaler = self.load_upscaler()
            lora = self.load_distilled_lora()

        height, width = sampling.snap_dimensions(height, width, two_stage=True)
        f, h1, w1 = sampling.video_latent_shape(num_frames, height // 2, width // 2)
        audio_t = sampling.audio_token_count(num_frames, fps)
        apos = sampling.audio_positions(audio_t)
        stepper = self._stepper(tm, on_step)
        stills = self._prepare_images(image, image_strength, images, num_frames)
        cond1 = self._image_latents(stills, h1, w1, tm)

        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h1 * w1, 128),
                sampling.video_positions(f, h1, w1, fps),
                seed,
                tokens_per_frame=h1 * w1,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond1, fps, 1.0, seed, keyframes_only)
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise
            )
            guided = GuidedDenoiser(
                dit, cond, neg, video_guidance, audio_guidance, batched
            )
            v1, a1 = sampling.euler_loop(
                guided,
                video,
                audio,
                # Upstream calls scheduler.execute(steps) without a latent, so the
                # shift is the 4096-token anchor (2.05) at every resolution.
                sampling.ltx2_schedule(steps, 4096),
                on_step=stepper("stage1"),
                step_cache=fast.cache(),
            )

        with tm.span("upscale"):
            half = sampling.unpatchify(v1[:, : f * h1 * w1], (f, h1, w1))
            up = vae.normalize(upscaler(vae.denormalize(half)))
            mx.eval(up)
        h2, w2 = h1 * 2, w1 * 2
        cond2 = self._image_latents(stills, h2, w2, tm)

        with tm.span("stage2"):
            sigmas2 = fast.sigmas2(sampling.STAGE_2_DISTILLED_SIGMAS)
            s0 = sigmas2[0]
            video = sampling.noised_state(
                (1, f * h2 * w2, 128),
                sampling.video_positions(f, h2, w2, fps),
                seed + 2,
                sigma=s0,
                initial=sampling.patchify(up),
                tokens_per_frame=h2 * w2,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond2, fps, s0, seed + 2, keyframes_only)
            if keyframes_only:
                # Upstream keyframe_interpolation.py re-noises the stage-1 audio
                # to sigma 0.909 and refines it with the video; it ships from
                # stage 2.
                audio = sampling.noised_state(
                    (1, audio_t, 128),
                    apos,
                    seed + 2,
                    sigma=s0,
                    initial=a1,
                    bf16_noise=self.bf16_noise,
                )
            else:
                # Upstream ti2vid_two_stages.py refines stage 2 with
                # freeze_audio=True: the stage-1 audio stays clean (sigma 0 for
                # its tokens, its AdaLN and the cross-attention gates) and is
                # what ships.
                audio = sampling.LatentState(
                    latent=a1,
                    clean=a1,
                    denoise_mask=mx.zeros((1, audio_t, 1), dtype=G),
                    positions=apos,
                    frozen=True,
                )
            lora.attach(dit)
            try:
                v2, a2 = sampling.euler_loop(
                    Denoiser(
                        dit,
                        *cond,
                        video_tiles=fast.tiles(f, h2, w2, video.latent.shape[1]),
                    ),
                    video,
                    audio,
                    sigmas2,
                    on_step=stepper("stage2"),
                    step_cache=fast.cache(),
                )
            finally:
                lora.detach(dit)

        return Result(
            sampling.unpatchify(v2[:, : f * h2 * w2], (f, h2, w2)),
            a2 if keyframes_only else a1,
            num_frames,
            height,
            width,
            fps,
            tm,
            predicted_seconds=predicted,
        )

    # ---- keyframe interpolation ---------------------------------------------
    def keyframes(
        self,
        prompt: str,
        images: list[sampling.Still],
        **kwargs,
    ) -> Result:
        """Lightricks' KeyframeInterpolationPipeline: the dev flow with every
        still appended as a keyframe token block at its pixel frame (frame 0
        included; image_conditionings_by_adding_guiding_latent) and the audio
        re-noised and refined in stage 2. Takes `dev`'s other arguments."""
        if not images:
            raise ValueError("keyframe interpolation needs at least one still")
        if kwargs.get("image") is not None:
            raise ValueError("pass the stills as `images` (frame indices)")
        return self.dev(prompt, images=images, keyframes_only=True, **kwargs)

    # ---- one stage ------------------------------------------------------------
    def one_stage(
        self,
        prompt: str,
        height: int = 512,
        width: int = 768,
        num_frames: int | None = 121,
        fps: float = 24.0,
        seed: int = 42,
        negative_prompt: str | None = None,
        steps: int = 30,
        video_guidance: sampling.Guidance = DEFAULT_VIDEO_GUIDANCE,
        audio_guidance: sampling.Guidance = DEFAULT_AUDIO_GUIDANCE,
        batched: bool = True,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        image: str | bytes | None = None,
        image_strength: float = 1.0,
        fast: Fast | None = None,
        max_num_frames: int | None = None,
        images: list[sampling.Still] | None = None,
    ) -> Result:
        """Lightricks' TI2VidOneStagePipeline: one guided dev stage at the
        output size (sizes snap to 32), no upsampler, audio from the same
        stage. Upstream keeps it for prototyping; the two-stage flows are the
        production paths. `fast` applies its steps, guidance and step cache."""
        if self.variant != "dev":
            raise ValueError("the one-stage pipeline needs LTX25Engine(variant='dev')")
        fast = fast or Fast()
        if fast.steps:
            steps = fast.steps
        if fast.guidance:
            video_guidance = replace(video_guidance, **fast.guidance)
            audio_guidance = replace(audio_guidance, **fast.guidance)
        tm = Timings()
        with tm.span("text"):
            text = self.load_text()
            cond = text.encode(prompt)[:2]
            neg = text.encode(
                sampling.DEFAULT_NEGATIVE_PROMPT
                if negative_prompt is None
                else negative_prompt
            )[:2]
            mx.eval(cond, neg)
            num_frames, predicted = self.resolve_frames(
                num_frames, cond[0], cond[1], fps, max_num_frames
            )
            if not keep_text:
                self.unload_text()
        with tm.span("load"):
            dit = self.load_dit()
            self.load_vae()

        height, width = sampling.snap_dimensions(height, width, two_stage=False)
        f, h, w = sampling.video_latent_shape(num_frames, height, width)
        audio_t = sampling.audio_token_count(num_frames, fps)
        stepper = self._stepper(tm, on_step)
        stills = self._prepare_images(image, image_strength, images, num_frames)
        conds = self._image_latents(stills, h, w, tm)

        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h * w, 128),
                sampling.video_positions(f, h, w, fps),
                seed,
                tokens_per_frame=h * w,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, conds, fps, 1.0, seed)
            audio = sampling.noised_state(
                (1, audio_t, 128),
                sampling.audio_positions(audio_t),
                seed + 1,
                bf16_noise=self.bf16_noise,
            )
            guided = GuidedDenoiser(
                dit, cond, neg, video_guidance, audio_guidance, batched
            )
            v, a = sampling.euler_loop(
                guided,
                video,
                audio,
                sampling.ltx2_schedule(steps, 4096),
                on_step=stepper("stage1"),
                step_cache=fast.cache(),
            )
        return Result(
            sampling.unpatchify(v[:, : f * h * w], (f, h, w)),
            a,
            num_frames,
            height,
            width,
            fps,
            tm,
            predicted_seconds=predicted,
        )

    # ---- hq (res_2s) --------------------------------------------------------
    def hq(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1536,
        num_frames: int | None = 121,
        fps: float = 24.0,
        seed: int = 42,
        negative_prompt: str | None = None,
        steps: int = HQ_STEPS,
        video_guidance: sampling.Guidance = HQ_VIDEO_GUIDANCE,
        audio_guidance: sampling.Guidance = HQ_AUDIO_GUIDANCE,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        image: str | bytes | None = None,
        image_strength: float = 1.0,
        max_num_frames: int | None = None,
        images: list[sampling.Still] | None = None,
    ) -> Result:
        """Lightricks' TI2VidTwoStagesHQPipeline: the dev transformer with the
        distilled LoRA at 0.25, 15 guided res_2s steps (CFG 3 / 7, no STG,
        modality 3, rescale 0.45 / 1.0) on the token-count-shifted schedule at
        half resolution; 2x latent upscale; 3 res_2s steps at full resolution
        with the LoRA at 0.5, audio re-noised and refined alongside (it ships
        from stage 2). The SDE noise streams are seeded from the request seed
        (upstream leaves them at their default seed)."""
        if self.variant != "dev":
            raise ValueError("the HQ pipeline needs LTX25Engine(variant='dev')")
        tm = Timings()
        with tm.span("text"):
            text = self.load_text()
            cond = text.encode(prompt)[:2]
            neg = text.encode(
                sampling.DEFAULT_NEGATIVE_PROMPT
                if negative_prompt is None
                else negative_prompt
            )[:2]
            mx.eval(cond, neg)
            num_frames, predicted = self.resolve_frames(
                num_frames, cond[0], cond[1], fps, max_num_frames
            )
            if not keep_text:
                self.unload_text()
        with tm.span("load"):
            dit = self.load_dit()
            vae = self.load_vae()
            upscaler = self.load_upscaler()
            lora = self.load_distilled_lora()

        height, width = sampling.snap_dimensions(height, width, two_stage=True)
        f, h1, w1 = sampling.video_latent_shape(num_frames, height // 2, width // 2)
        audio_t = sampling.audio_token_count(num_frames, fps)
        apos = sampling.audio_positions(audio_t)
        stepper = self._stepper(tm, on_step)
        stills = self._prepare_images(image, image_strength, images, num_frames)
        cond1 = self._image_latents(stills, h1, w1, tm)

        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h1 * w1, 128),
                sampling.video_positions(f, h1, w1, fps),
                seed,
                tokens_per_frame=h1 * w1,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond1, fps, 1.0, seed)
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise
            )
            guided = GuidedDenoiser(dit, cond, neg, video_guidance, audio_guidance)
            lora.attach(dit, HQ_LORA_STAGE_1)
            try:
                v1, a1 = sampling.res2s_loop(
                    guided,
                    video,
                    audio,
                    # upstream hands the scheduler the stage-1 latent: the shift
                    # follows its token count (the dev pipeline's does not)
                    sampling.ltx2_schedule(steps, f * h1 * w1),
                    noise_seed=seed + RES2S_NOISE_SEED_OFFSET,
                    on_step=stepper("stage1"),
                )
                mx.eval(v1, a1)
            finally:
                lora.detach(dit)

        with tm.span("upscale"):
            half = sampling.unpatchify(v1[:, : f * h1 * w1], (f, h1, w1))
            up = vae.normalize(upscaler(vae.denormalize(half)))
            mx.eval(up)
        h2, w2 = h1 * 2, w1 * 2
        cond2 = self._image_latents(stills, h2, w2, tm)

        with tm.span("stage2"):
            sigmas2 = list(sampling.STAGE_2_DISTILLED_SIGMAS)
            s0 = sigmas2[0]
            video = sampling.noised_state(
                (1, f * h2 * w2, 128),
                sampling.video_positions(f, h2, w2, fps),
                seed + 2,
                sigma=s0,
                initial=sampling.patchify(up),
                tokens_per_frame=h2 * w2,
                bf16_noise=self.bf16_noise,
            )
            video = self._condition(video, cond2, fps, s0, seed + 2)
            audio = sampling.noised_state(
                (1, audio_t, 128),
                apos,
                seed + 2,
                sigma=s0,
                initial=a1,
                bf16_noise=self.bf16_noise,
            )
            lora.attach(dit, HQ_LORA_STAGE_2)
            try:
                v2, a2 = sampling.res2s_loop(
                    Denoiser(dit, *cond),
                    video,
                    audio,
                    sigmas2,
                    noise_seed=seed + RES2S_NOISE_SEED_OFFSET + 1,
                    on_step=stepper("stage2"),
                )
                mx.eval(v2, a2)
            finally:
                lora.detach(dit)

        return Result(
            sampling.unpatchify(v2[:, : f * h2 * w2], (f, h2, w2)),
            a2,
            num_frames,
            height,
            width,
            fps,
            tm,
            predicted_seconds=predicted,
        )

    # ---- decode -----------------------------------------------------------
    def load_diffvae(self):
        if self.diffvae is None:
            from slimserve.video.ltx25.diffvae import DiffusionVAE

            self.diffvae = DiffusionVAE(self.root).load()
        return self.diffvae

    def render(
        self,
        result: Result,
        path: str | Path,
        seed: int = 42,
        decoder: str | None = None,
    ) -> Path:
        """Decode and mux. `decoder`: "diffusion" (Lightricks' default: sharper
        faces, textures and text; ~4x the decode time) or "conv"."""
        from slimserve.video.ltx25 import mux

        decoder = decoder or self.decoder
        tm = result.timings
        with tm.span("vae_decode"):
            from slimserve.video.ltx25 import vae as vae_mod

            if decoder == "diffusion":
                from slimserve.video.ltx25 import diffvae as diff_mod

                need = diff_mod.estimate_peak_bytes(result.video_latent.shape)
                total = int(mx.device_info()["memory_size"])

                def room() -> int:
                    return total - int(mx.get_active_memory()) - OS_RESERVE_BYTES - need

                # Make room for the decoder and its buffer cache: first the text
                # encoder (28 GiB, 4.8 s to reload), then the DiT (44 GiB, 6 s).
                if room() < DECODE_CACHE_BYTES:
                    self.unload_text()
                if room() < DECODE_CACHE_BYTES:
                    self.unload_dit()
                if room() < CACHE_BYTES:
                    raise MemoryError(
                        f"the diffusion decoder needs ~{need >> 30} GiB "
                        "for this clip and "
                        f"{max(room(), 0) >> 30} GiB are free "
                        "next to the resident models; "
                        "use decoder=conv or a smaller clip"
                    )
                cache = int(min(DECODE_CACHE_BYTES, room()))
                mx.set_cache_limit(cache)
                mx.set_memory_limit(total - OS_RESERVE_BYTES - cache)
                try:
                    pixels = self.load_diffvae().decode_raw(
                        result.video_latent, seed=seed, keyframes=result.keyframes
                    )
                    mx.eval(pixels)
                finally:
                    mx.set_cache_limit(CACHE_BYTES)
                    mx.set_memory_limit(total - OS_RESERVE_BYTES - CACHE_BYTES)
                frames = vae_mod.to_uint8(pixels)
                del pixels
            else:
                vae = self.load_vae()
                # An untiled decode is exact and fastest. If the resident text
                # encoder is what stands in the way, release it (4.8 s to
                # reload) rather than blend tiles.
                if (
                    vae_mod.estimate_peak_bytes(result.video_latent.shape, None)
                    > self.decode_budget()
                ):
                    self.unload_text()
                frames = vae.decode(
                    result.video_latent,
                    frame_rate=result.fps,
                    budget_bytes=self.decode_budget(),
                )
        with tm.span("audio_decode"):
            waveform, sample_rate = self.load_audio().decode(result.audio_tokens)
        with tm.span("mux"):
            frames = frames[: result.num_frames]
            mux.write_mp4(str(path), frames, result.fps, waveform, sample_rate)
        return Path(path)
