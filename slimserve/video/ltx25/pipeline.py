# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 pipelines on the SlimServe engine.

`distilled` mirrors Lightricks' DistilledPipeline: stage 1 at half resolution
(8 ancestral Euler steps, no guidance), 2x latent upscale, stage 2 at full
resolution (3 Euler steps), with the same distilled transformer in both.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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
    video_latent: mx.array | None  # (1, 128, F, H, W), normalized; None: audio only
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
    # A2Vid / DubIt: the source waveform ((channels, samples) float, rate) that
    # ships in place of the decoded audio tokens (upstream replace_chunks_audio:
    # the samples covering the clip's frames)
    source_audio: tuple[np.ndarray, int] | None = None


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
        step: int | None = None,
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
        self.last: tuple[mx.array, mx.array] | None = None  # for skip_step
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
        video: LatentState | None,
        audio: LatentState,
        vx,
        ax,
        sigma: float,
        rows: slice,
        step_cache: StepCache | None = None,
        run_video: bool = True,
        run_audio: bool = True,
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
        no_video = video is None  # T2A: an audio-only forward
        vt = (
            None
            if no_video or video.uniform
            else rep((video.denoise_mask * sigma).squeeze(-1))
        )
        at = None if audio.uniform else rep((audio.denoise_mask * sigma).squeeze(-1))
        v, a = self.dit(
            None if no_video else rep(vx),
            rep(ax),
            t,
            None if no_video else vtext,
            atext,
            None if no_video else rep(video.positions),
            rep(audio.positions),
            video_keyframes_mask=None if no_video else rep(video.keyframes_mask),
            video_timesteps=vt,
            audio_timesteps=at,
            video_attention_mask=None if no_video else video.attention_mask,
            audio_attention_mask=audio.attention_mask,
            stg={k: m[rows] for k, m in self.stg.items()},
            text_rows=trows,
            share_from=self.share if n == self.passes and not no_video else None,
            video_tiles=None if no_video else self.video_tiles,
            step_cache=step_cache if n == self.passes else None,
            run_video=run_video and not no_video,
            run_audio=run_audio,
        )
        return (
            None
            if v is None
            else x0_from_velocity(rep(vx), v, t if vt is None else vt),
            None
            if a is None
            else x0_from_velocity(rep(ax), a, t if at is None else at),
        )

    def __call__(
        self,
        video: LatentState,
        audio: LatentState,
        vx: mx.array,
        ax: mx.array,
        sigma: float,
        step_cache: StepCache | None = None,
        step: int | None = None,
    ):
        # upstream _guided_denoise: a modality whose guider skips this step
        # keeps its last x0 (both skipped: no forward at all)
        run_v, run_a = not self.video_g.skips(step), not self.audio_g.skips(step)
        if video is None:
            run_v = False
        if not (run_v or run_a):
            if self.last is None:
                raise ValueError("skip_step cannot skip the first step")
            return self.last
        if self.batched:
            v0, a0 = self._forward(
                video,
                audio,
                vx,
                ax,
                sigma,
                slice(0, self.passes),
                step_cache,
                run_v,
                run_a,
            )
        else:
            parts = [
                self._forward(
                    video, audio, vx, ax, sigma, slice(i, i + 1), None, run_v, run_a
                )
                for i in range(self.passes)
            ]
            cat = lambda k: (  # noqa: E731
                None
                if parts[0][k] is None
                else mx.concatenate([p[k] for p in parts], axis=0)
            )
            v0, a0 = cat(0), cat(1)
        rows = {k: i for i, k in enumerate(self.kinds)}

        def term(x: mx.array, kind: str) -> mx.array | None:
            i = rows.get(kind)
            return None if i is None else x[i : i + 1]

        def guided(x: mx.array | None, g: sampling.Guidance, last) -> mx.array:
            if x is None:
                if last is None:
                    raise ValueError("skip_step cannot skip the first step")
                return last
            return g.combine(x[0:1], term(x, "neg"), term(x, "stg"), term(x, "mod"))

        prev = self.last or (None, None)
        self.last = (
            None if video is None else guided(v0, self.video_g, prev[0]),
            guided(a0, self.audio_g, prev[1]),
        )
        return self.last


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

    def __call__(self, video, audio, vx, ax, sigma, step_cache=None, step=None):
        n = vx.shape[1]
        out_v = mx.zeros_like(vx)
        out_a = None
        for idx, weight, t in self.plans:

            def take(z, idx=idx):
                return None if z is None else mx.take(z, idx, axis=1)

            pos = take(video.positions) - mx.array([0.0, t.h0 * 32.0, t.w0 * 32.0])
            mask = video.attention_mask
            if mask is not None:
                mask = mx.take(mx.take(mask, idx, axis=1), idx, axis=2)
            sub = sampling.LatentState(
                latent=take(video.latent),
                clean=take(video.clean),
                denoise_mask=take(video.denoise_mask),
                positions=pos,
                keyframes_mask=take(video.keyframes_mask),
                attention_mask=mask,
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
        self.user_lora_cache: dict[str, Any] = {}  # path -> loaded Lora

    # ---- user LoRAs (upstream --lora PATH [STRENGTH]) ------------------------
    @contextmanager
    def user_loras(self, specs: list[tuple[str, float]] | None) -> Iterator[None]:
        """Attach user adapters to the transformer for one request: every stage
        of every pipeline runs with them (upstream passes `loras` to each
        DiffusionStage, the official distilled / detailing adapters on top).
        Loaded files are kept by path."""
        if not specs:
            yield
            return
        from slimserve.video.ltx25.lora import Lora

        dit = self.load_dit()
        attached = []
        try:
            for path, strength in specs:
                key = str(Path(path).expanduser().resolve())
                lora = self.user_lora_cache.get(key)
                if lora is None:
                    lora = self.user_lora_cache[key] = Lora.from_path(key).load()
                lora.attach(dit, strength)
                attached.append(lora)
            yield
        finally:
            for lora in attached:
                lora.detach(dit)

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

    def load_audio(self, encoder: bool = False):
        """The audio VAE decoder + vocoder; `encoder` adds the VAE encoder (the
        editing pipelines' source audio) to the resident set."""
        if self.audio is None or (encoder and not self.audio.with_encoder):
            from slimserve.video.ltx25.audio import AudioDecoder

            self.audio = AudioDecoder(self.root, encoder=encoder)
            self.audio.load()
        return self.audio

    # ---- source clips (retake, IC-LoRA, A2Vid, DubIt) ----------------------------
    SOURCE_ENCODE_TILING = ((768, 64), (80, 24))  # upstream TileSizeConfig.default()

    def _source_video_latent(
        self,
        path: str,
        info,
        height: int,
        width: int,
        tm: Timings,
        start_time: float = 0.0,
        num_frames: int | None = None,
        fps: float | None = None,
    ) -> mx.array:
        """Upstream video_latent_from_file: the frames in [start_time, start_time
        + num_frames / fps) decoded (`media.read_frames`), resized to fill and
        center-cropped to height x width and mapped to [-1, 1] as the stills
        are, tiled-encoded (frames 80/24, 768/64 px), the latent length
        conformed to the clip's (trimmed, or zero-padded at the end)."""
        from slimserve.video.ltx25 import image as image_mod
        from slimserve.video.ltx25 import media
        from slimserve.video.ltx25.vae import Tiling

        fps = fps or info.fps
        num_frames = num_frames or info.frames
        with tm.span("encode_video"):
            frames = media.read_frames(path, info, start_time, num_frames / fps)
            if frames.shape[0] == 0:
                raise ValueError(f"{path}: no frames from {start_time:.3f} s")
            pixels = np.stack(
                [
                    image_mod.resize_and_center_crop(
                        f.astype(np.float32), height, width
                    )
                    for f in frames
                ]
            )
            pixels = mx.array((pixels / 127.5 - 1.0).transpose(3, 0, 1, 2))[None]
            spatial, temporal = self.SOURCE_ENCODE_TILING
            latent = self.load_vae().encode_tiled(
                pixels, Tiling(spatial=spatial, temporal=temporal)
            )
            want = (num_frames - 1) // 8 + 1
            latent = latent[:, :, :want]
            if latent.shape[2] < want:
                pad = mx.zeros((1, 128, want - latent.shape[2], *latent.shape[3:]))
                latent = mx.concatenate([latent, pad], axis=2)
            mx.eval(latent)
            del pixels, frames
        return latent

    def _source_audio_tokens(
        self,
        path: str,
        info,
        num_frames: int,
        fps: float,
        tm: Timings,
        start_time: float = 0.0,
    ) -> mx.array | None:
        """Upstream audio_latent_from_file: the stream's samples in the window,
        resampled to 16 kHz, log-mel, the VAE encoder, conformed to the clip's
        token count; None without an audio stream."""
        from slimserve.video.ltx25 import media

        audio = media.read_audio(path, info, start_time, num_frames / fps)
        if audio is None:
            return None
        with tm.span("encode_audio"):
            wav, rate = audio
            tokens = self.load_audio(encoder=True).encode(wav, rate)
            want = sampling.audio_token_count(num_frames, fps)
            tokens = tokens[:, :want]
            if tokens.shape[1] < want:
                tokens = mx.concatenate(
                    [tokens, mx.zeros((1, want - tokens.shape[1], 128))], axis=1
                )
            mx.eval(tokens)
        return tokens

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

    # ---- generated keyframe slots (upstream --num-generated-keyframes) --------
    @staticmethod
    def _slot_frames(generated_keyframes, num_frames: int) -> list[int]:
        """Upstream resolve_generated_keyframes: an int asks for that many
        evenly spaced interior pixel frames (linspace over [0, F - 1] rounded,
        endpoints dropped); a sequence gives the frames; 0 / empty is off."""
        if isinstance(generated_keyframes, bool):
            raise ValueError("generated_keyframes is a count or a list of frames")
        if isinstance(generated_keyframes, int):
            n = generated_keyframes
            if n < 0:
                raise ValueError("generated_keyframes must be non-negative")
            if n == 0:
                return []
            if num_frames < n + 2:
                raise ValueError(
                    f"{n} generated keyframes need at least {n + 2} frames, "
                    f"got {num_frames}"
                )
            return [
                int(x) for x in np.rint(np.linspace(0, num_frames - 1, n + 2))[1:-1]
            ]
        frames = sorted({int(x) for x in generated_keyframes})
        if frames and (frames[0] < 0 or frames[-1] >= num_frames):
            raise ValueError(f"generated keyframes must lie in [0, {num_frames})")
        return frames

    def _slots_stage1(self, video, frames, h, w, fps, seed):
        if not frames:
            return video, None
        return sampling.append_slots(video, frames, h, w, fps, None, 1.0, seed)

    def _slots_upscaled(self, v1, slots, frames, h1, w1):
        """The stage-1 slot planes, x2 upscaled, as stage 2's initial content."""
        if slots is None:
            return None
        vae, upscaler = self.load_vae(), self.load_upscaler()
        planes = sampling.slots_to_latent(v1, slots, len(frames), h1, w1)
        up = vae.normalize(upscaler(vae.denormalize(planes)))
        mx.eval(up)
        return up

    def _slots_stage2(self, video, frames, h, w, fps, initial, sigma, seed):
        if not frames:
            return video, None
        return sampling.append_slots(video, frames, h, w, fps, initial, sigma, seed)

    @staticmethod
    def _decode_keyframes(v, slots, frames, h, w, num_frames, decode_with_keyframes):
        """The denoised slot planes for the keyframe-aware decode (upstream
        decode_keyframes_from_slots), or None."""
        if slots is None or not decode_with_keyframes:
            return None
        planes = sampling.slots_to_latent(v, slots, len(frames), h, w)
        kept = [i for i, fr in enumerate(frames) if 0 <= fr < num_frames]
        if not kept:
            return None
        planes = mx.concatenate([planes[:, :, i : i + 1] for i in kept], axis=2)
        mx.eval(planes)
        return planes, [frames[i] for i in kept]

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
        generated_keyframes: int | list[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> Result:
        """`image` (a path or encoded image bytes) conditions the first frame
        (image-to-video) at `image_strength` in both stages; `images` are
        further stills at any pixel frame (upstream's repeatable --image: frame
        0 replaces latent frame 0, other frames become appended keyframe
        tokens). `fast` stacks the output-changing fast tier (see `Fast`).
        `num_frames` None: the duration head decides (see `resolve_frames`).
        `generated_keyframes` (upstream --num-generated-keyframes): that many
        evenly spaced generated keyframe slots (or the frames given) ride in
        both stages as in DFR; `decode_with_keyframes` anchors the diffusion
        decode on them."""
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
            slot_frames = self._slot_frames(generated_keyframes, num_frames)
            video, slots = self._slots_stage1(video, slot_frames, h1, w1, fps, seed)
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
            slots_up = self._slots_upscaled(v1, slots, slot_frames, h1, w1)
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
            video, slots2 = self._slots_stage2(
                video, slot_frames, h2, w2, fps, slots_up, s0, seed + 2
            )
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
            keyframes=self._decode_keyframes(
                v2, slots2, slot_frames, h2, w2, num_frames, decode_with_keyframes
            ),
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
        source_audio: tuple[mx.array, np.ndarray, int] | None = None,
        generated_keyframes: int | list[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> Result:
        """Lightricks' TI2VidTwoStagesPipeline: guided dev stage 1 at half
        resolution, 2x latent upscale, 3-step stage 2 with the distilled LoRA
        and no guidance. Audio is taken from stage 1, as upstream. `image` /
        `images` condition the clip in both stages (see `distilled`). `fast`
        stacks the fast tier (see `Fast`). `keyframes_only` is the
        KeyframeInterpolationPipeline: every still is appended as keyframe
        tokens (frame 0 too) and the audio is re-noised and refined in stage 2.
        `source_audio` (tokens (1, T, 128), waveform, rate) is the A2Vid
        pipeline: the encoded source audio rides frozen through both stages
        (audio guidance off) and the source waveform ships."""
        if self.variant != "dev":
            raise ValueError("the dev pipeline needs LTX25Engine(variant='dev')")
        if source_audio is not None:
            audio_guidance = sampling.Guidance(
                cfg=1.0, stg=0.0, modality=1.0, rescale=0.0
            )
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
            slot_frames = self._slot_frames(generated_keyframes, num_frames)
            video, slots = self._slots_stage1(video, slot_frames, h1, w1, fps, seed)
            if source_audio is not None:
                tokens = source_audio[0][:, :audio_t]
                if tokens.shape[1] < audio_t:
                    tokens = mx.concatenate(
                        [tokens, mx.zeros((1, audio_t - tokens.shape[1], 128))], axis=1
                    )
                audio = sampling.LatentState(
                    latent=tokens,
                    clean=tokens,
                    denoise_mask=mx.zeros((1, audio_t, 1), dtype=G),
                    positions=apos,
                    frozen=True,
                )
            else:
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
            slots_up = self._slots_upscaled(v1, slots, slot_frames, h1, w1)
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
            video, slots2 = self._slots_stage2(
                video, slot_frames, h2, w2, fps, slots_up, s0, seed + 2
            )
            if keyframes_only and source_audio is None:
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
            a2 if keyframes_only and source_audio is None else a1,
            num_frames,
            height,
            width,
            fps,
            tm,
            keyframes=self._decode_keyframes(
                v2, slots2, slot_frames, h2, w2, num_frames, decode_with_keyframes
            ),
            predicted_seconds=predicted,
            source_audio=None if source_audio is None else source_audio[1:],
        )

    # ---- audio to video ---------------------------------------------------------
    def a2vid(
        self,
        prompt: str,
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        num_frames: int | None = None,
        fps: float = 24.0,
        **kwargs,
    ) -> Result:
        """Lightricks' A2VidPipelineTwoStage: the dev flow driven by an audio
        file. The stream's samples from `audio_start_time` (at most
        `audio_max_duration` s) are encoded through the audio VAE and ride
        frozen through both stages (video guidance only); the clip length
        follows the audio when `num_frames` is not given (int(duration * fps),
        at most 1024, snapped to 8k + 1); the source waveform ships, cut to the
        clip. Takes `dev`'s other arguments."""
        from slimserve.video.ltx25 import media

        info = media.probe_audio(audio_path)
        audio = media.read_audio(audio_path, info, audio_start_time, audio_max_duration)
        if audio is None:
            raise ValueError(f"{audio_path}: no audio from {audio_start_time:.3f} s")
        wav, rate = audio
        if num_frames is None:
            raw = int(wav.shape[1] / rate * fps)
            num_frames = max(1, min(raw, 1024))
            cap = kwargs.get("max_num_frames")
            if cap is not None:  # the profile's envelope at this size
                num_frames = min(num_frames, cap)
        num_frames = max(1, (num_frames - 1) // 8 * 8 + 1)
        tm_tokens = self.load_audio(encoder=True).encode(wav, rate)
        mx.eval(tm_tokens)
        return self.dev(
            prompt,
            num_frames=num_frames,
            fps=fps,
            source_audio=(tm_tokens, wav, rate),
            **kwargs,
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
        generated_keyframes: int | list[int] = 0,
        decode_with_keyframes: bool = False,
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
            slot_frames = self._slot_frames(generated_keyframes, num_frames)
            video, slots = self._slots_stage1(video, slot_frames, h, w, fps, seed)
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
            keyframes=self._decode_keyframes(
                v, slots, slot_frames, h, w, num_frames, decode_with_keyframes
            ),
            predicted_seconds=predicted,
        )

    # ---- retake -----------------------------------------------------------------
    def retake(
        self,
        prompt: str,
        video_path: str,
        start_time: float,
        end_time: float,
        seed: int = 42,
        negative_prompt: str | None = None,
        steps: int = 40,
        video_guidance: sampling.Guidance = DEFAULT_VIDEO_GUIDANCE,
        audio_guidance: sampling.Guidance = DEFAULT_AUDIO_GUIDANCE,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        fast: Fast | None = None,
        batched: bool = True,
    ) -> Result:
        """Lightricks' RetakePipeline: the source clip (its own size, frame
        count and rate; 8k + 1 frames, sides multiples of 32) is encoded to
        video and audio latents, and only the tokens whose time span overlaps
        [start_time, end_time) are regenerated from the prompt
        (TemporalRegionMask), the rest stays the source. On the distilled
        transformer (upstream's CLI): the 8 distilled sigmas, plain Euler, no
        guidance; on the dev transformer: `steps` guided steps with the
        negative prompt. A source without audio gets its audio generated over
        the whole clip; `regenerate_video` / `regenerate_audio` False keeps
        that modality frozen. `fast` applies its step cache and sigma override."""
        from slimserve.video.ltx25 import media

        if start_time >= end_time:
            raise ValueError("start_time must be less than end_time")
        fast = fast or Fast()
        if fast.steps:
            steps = fast.steps
        if fast.guidance:
            video_guidance = replace(video_guidance, **fast.guidance)
            audio_guidance = replace(audio_guidance, **fast.guidance)
        info = media.probe(video_path)
        if (info.frames - 1) % 8:
            snapped = (info.frames - 1) // 8 * 8 + 1
            raise ValueError(
                f"the source has {info.frames} frames; retake needs 8k + 1 "
                f"(trim it to {snapped})"
            )
        if info.width % 32 or info.height % 32:
            raise ValueError(
                f"the source is {info.width}x{info.height}; sides must be "
                "multiples of 32"
            )
        guided = self.variant == "dev"
        tm = Timings()
        with tm.span("text"):
            text = self.load_text()
            cond = text.encode(prompt)[:2]
            neg = None
            if guided:
                neg = text.encode(
                    sampling.DEFAULT_NEGATIVE_PROMPT
                    if negative_prompt is None
                    else negative_prompt
                )[:2]
                mx.eval(cond, neg)
            else:
                mx.eval(cond)
            if not keep_text:
                self.unload_text()
        with tm.span("load"):
            dit = self.load_dit()
            self.load_vae()
        num_frames, fps = info.frames, info.fps
        f, h, w = sampling.video_latent_shape(num_frames, info.height, info.width)
        source = self._source_video_latent(
            video_path, info, info.height, info.width, tm
        )
        audio_src = self._source_audio_tokens(video_path, info, num_frames, fps, tm)
        audio_t = sampling.audio_token_count(num_frames, fps)
        stepper = self._stepper(tm, on_step)

        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h * w, 128),
                sampling.video_positions(f, h, w, fps),
                seed,
                initial=sampling.patchify(source),
                tokens_per_frame=h * w,
                bf16_noise=self.bf16_noise,
            )
            if regenerate_video:
                video = sampling.region_mask(video, start_time, end_time, fps, h * w)
            else:
                video = replace(
                    video,
                    latent=video.clean,
                    denoise_mask=mx.zeros_like(video.denoise_mask),
                    frozen=True,
                )
            audio = sampling.noised_state(
                (1, audio_t, 128),
                sampling.audio_positions(audio_t),
                seed + 1,
                initial=audio_src,
                bf16_noise=self.bf16_noise,
            )
            if audio_src is None:
                pass  # no audio track: generated from scratch over the clip
            elif regenerate_audio:
                audio = sampling.region_mask(audio, start_time, end_time, fps, None)
            else:
                audio = replace(
                    audio,
                    latent=audio.clean,
                    denoise_mask=mx.zeros_like(audio.denoise_mask),
                    frozen=True,
                )
            mx.eval(video.latent, video.denoise_mask, audio.latent, audio.denoise_mask)
            if guided:
                denoise = GuidedDenoiser(
                    dit, cond, neg, video_guidance, audio_guidance, batched
                )
                sigmas = sampling.ltx2_schedule(steps, 4096)
            else:
                denoise = Denoiser(
                    dit, *cond, video_tiles=fast.tiles(f, h, w, video.latent.shape[1])
                )
                sigmas = fast.sigmas1(sampling.DISTILLED_SIGMAS)
            # upstream's stage runs its default loop here: plain Euler
            v, a = sampling.euler_loop(
                denoise,
                video,
                audio,
                sigmas,
                on_step=stepper("stage1"),
                step_cache=fast.cache(),
            )
        return Result(
            sampling.unpatchify(v, (f, h, w)),
            a,
            num_frames,
            info.height,
            info.width,
            fps,
            tm,
        )

    # ---- IC-LoRA (video-to-video) ---------------------------------------------
    IC_LORA_TILE = (1024, 1536)  # upstream LTX_2_PARAMS stage 2 (height, width)

    def _reference_tokens(
        self,
        path: str,
        info,
        height: int,
        width: int,
        num_frames: int,
        downscale: int,
        temporal_scale: int,
        tm: Timings,
    ) -> tuple[mx.array, tuple[int, int, int]]:
        """Upstream append_ic_lora_reference_video_conditionings: the first
        num_frames frames of the reference (by index), resized to fill and
        center-cropped to the stage's size over `downscale`, frame 0 then every
        `temporal_scale`th frame kept, encoded untiled. Returns the patchified
        tokens and the latent (f, h, w)."""
        from slimserve.video.ltx25 import image as image_mod
        from slimserve.video.ltx25 import media

        if height % downscale or width % downscale:
            raise ValueError(
                f"{width}x{height} must be divisible by the adapter's "
                f"reference_downscale_factor {downscale}"
            )
        rh, rw = height // downscale, width // downscale
        with tm.span("reference"):
            frames = media.read_frames(path, info, 0.0, num_frames / info.fps)[
                :num_frames
            ]
            if frames.shape[0] == 0:
                raise ValueError(f"{path}: no frames")
            if temporal_scale > 1:
                frames = frames[[0, *range(1, frames.shape[0], temporal_scale)]]
            pixels = np.stack(
                [
                    image_mod.resize_and_center_crop(fr.astype(np.float32), rh, rw)
                    for fr in frames
                ]
            )
            pixels = mx.array((pixels / 127.5 - 1.0).transpose(3, 0, 1, 2))[None]
            latent = self.load_vae().encode(pixels)
            mx.eval(latent)
            del pixels, frames
        _, _, f, h, w = latent.shape
        return sampling.patchify(latent), (f, h, w)

    def ic_lora(
        self,
        prompt: str,
        video_conditioning: list[tuple[str, float]],
        loras: list[tuple[str, float]],
        height: int = 1024,
        width: int = 1536,
        num_frames: int | None = 121,
        fps: float = 24.0,
        seed: int = 42,
        images: list[sampling.Still] | None = None,
        image: str | bytes | None = None,
        image_strength: float = 1.0,
        attention_strength: float = 1.0,
        attention_mask: str | None = None,
        skip_stage_2: bool = False,
        stage_2_ic_lora: bool = False,
        tile: bool = False,
        tile_height: int | None = None,
        tile_width: int | None = None,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        fast: Fast | None = None,
        max_num_frames: int | None = None,
        generated_keyframes: int | list[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> Result:
        """Lightricks' ICLoraPipeline, the CLI's two-stage recipe: stage 1 at
        half resolution on the distilled transformer under the IC-LoRA
        adapters (`loras`: path, strength; their metadata sets the reference
        downscale and temporal factors), the 8 distilled sigmas, plain Euler,
        with each reference video (`video_conditioning`: path, strength)
        encoded at the stage's size and appended as clean reference tokens
        (VideoConditionByReferenceLatent); then the 2x latent upscale and a
        3-sigma stage 2 on the bare checkpoint (`stage_2_ic_lora` keeps the
        adapters and references there too; `skip_stage_2` ships stage 1 at
        half size). `attention_strength` (0-1) and `attention_mask` (a
        grayscale mask video, per region) scale how strongly the reference
        tokens and the target attend to each other. `tile` runs each
        transformer call over pinned `tile_height` x `tile_width` windows
        (default 1024x1536, half overlap) blended with trapezoids; a tiled
        full-resolution stage needs `stage_2_ic_lora`. Stills condition as
        on `distilled`."""
        from slimserve.video.ltx25 import media
        from slimserve.video.ltx25.lora import Lora

        if self.variant != "distilled":
            raise ValueError("the IC-LoRA pipeline runs on the distilled transformer")
        if not video_conditioning:
            raise ValueError("ic_lora needs at least one reference video")
        if not loras:
            raise ValueError(
                "ic_lora needs the IC-LoRA adapter (--lora PATH [STRENGTH])"
            )
        if not 0.0 <= attention_strength <= 1.0:
            raise ValueError("attention_strength must be in [0, 1]")
        fast = fast or Fast()
        adapters = []
        downscale, temporal_scale = 1, 1
        for path, strength in loras:
            key = str(Path(path).expanduser().resolve())
            lora = self.user_lora_cache.get(key)
            if lora is None:
                lora = self.user_lora_cache[key] = Lora.from_path(key).load()
            for name, have, got in (
                ("reference_downscale_factor", downscale, lora.reference_downscale),
                (
                    "reference_temporal_scale_factor",
                    temporal_scale,
                    lora.reference_temporal_scale,
                ),
            ):
                if got != 1 and have not in (1, got):
                    raise ValueError(f"conflicting {name} values in the adapters")
            downscale = max(downscale, lora.reference_downscale)
            temporal_scale = max(temporal_scale, lora.reference_temporal_scale)
            adapters.append((lora, strength))
        if tile:
            th = tile_height or self.IC_LORA_TILE[0]
            tw = tile_width or self.IC_LORA_TILE[1]
            if th < 64 or tw < 64 or th % 32 or tw % 32:
                raise ValueError("tile sizes must be multiples of 32, at least 64")
            tiled_stage_2 = not skip_stage_2 and (height > th or width > tw)
            if tiled_stage_2 and not stage_2_ic_lora:
                raise ValueError(
                    "a tiled full-resolution stage needs the IC-LoRA on it "
                    "(stage_2_ic_lora)"
                )
        elif tile_height is not None or tile_width is not None:
            raise ValueError("tile_height / tile_width need tile")
        tm = Timings()
        with tm.span("text"):
            video_text, audio_text = self.load_text().encode(prompt)[:2]
            mx.eval(video_text, audio_text)
            if not keep_text:
                self.unload_text()
        num_frames, predicted = self.resolve_frames(
            num_frames, video_text, audio_text, fps, max_num_frames
        )
        with tm.span("load"):
            dit = self.load_dit()
            vae = self.load_vae()
            upscaler = None if skip_stage_2 else self.load_upscaler()
        height, width = sampling.snap_dimensions(
            height, width, two_stage=not skip_stage_2
        )
        infos = {path: media.probe(path) for path, _ in video_conditioning}
        mask_video = None
        if attention_mask is not None:
            # upstream _load_mask_video: decoded at the stage-1 size, grey, [0, 1]
            minfo = media.probe(attention_mask)
            frames = media.read_frames(attention_mask, minfo)[:num_frames]
            from slimserve.video.ltx25 import image as image_mod

            mask_video = np.stack(
                [
                    image_mod.resize_and_center_crop(
                        fr.astype(np.float32), height // 2, width // 2
                    ).mean(axis=-1)
                    / 255.0
                    for fr in frames
                ]
            ).clip(0.0, 1.0)
        f, h1, w1 = sampling.video_latent_shape(num_frames, height // 2, width // 2)
        audio_t = sampling.audio_token_count(num_frames, fps)
        apos = sampling.audio_positions(audio_t)
        stepper = self._stepper(tm, on_step)
        stills = self._prepare_images(image, image_strength, images, num_frames)

        def conditioned(video, h, w, sigma, stage_seed, with_reference):
            """The stage's conditioned state and, for the tiled stages, every
            token's cell extent on the target grid (generated and keyframe
            tokens: their cell; reference tokens: their cell times the
            downscale, as the DFR epilogue's extents)."""
            n_noisy = f * h * w
            yy, xx = np.divmod(np.arange(n_noisy) % (h * w), w)
            extents = [np.stack([yy, yy + 1, xx, xx + 1], axis=1).astype(np.float32)]
            video = self._condition(
                video, self._image_latents(stills, h, w, tm), fps, sigma, stage_seed
            )
            for _ in range((video.latent.shape[1] - n_noisy) // (h * w)):
                extents.append(extents[0][: h * w])
            if with_reference:
                for path, strength in video_conditioning:
                    tokens, (rf, rh, rw) = self._reference_tokens(
                        path,
                        infos[path],
                        h * 32,
                        w * 32,
                        num_frames,
                        downscale,
                        temporal_scale,
                        tm,
                    )
                    weights = None
                    if mask_video is not None:
                        weights = sampling.mask_video_to_tokens(mask_video, rf, rh, rw)
                        weights = weights * attention_strength
                    elif attention_strength < 1.0:
                        weights = attention_strength
                    video = sampling.append_reference(
                        video,
                        tokens,
                        sampling.video_positions(rf, rh, rw, fps),
                        downscale,
                        strength,
                        temporal_scale,
                        fps,
                        weights,
                        n_noisy,
                    )
                    ry, rx = np.divmod(np.arange(rf * rh * rw) % (rh * rw), rw)
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
            mx.eval(video.latent, video.clean, video.positions)
            return video, np.concatenate(extents)

        def tiles_for(h, w):
            if not tile:
                return None
            th_c, tw_c = th // 32, tw // 32
            return sampling.spatial_tiles_by_size(
                h, w, th_c, tw_c, int(th * 0.5) // 32, int(tw * 0.5) // 32
            )

        def run(stage, video, extents, audio, sigmas, h, w, lora_on, stage_tiles):
            den = Denoiser(
                dit,
                video_text,
                audio_text,
                video_tiles=None
                if stage_tiles
                else fast.tiles(f, h, w, video.latent.shape[1]),
            )
            if stage_tiles is not None:
                den = TiledDenoiser(den, f, h, w, extents, stage_tiles)
            for lora, strength in adapters if lora_on else ():
                lora.attach(dit, strength)
            try:
                return sampling.euler_loop(
                    den,
                    video,
                    audio,
                    sigmas,
                    on_step=stepper(stage),
                    step_cache=fast.cache() if stage_tiles is None else None,
                )
            finally:
                for lora, _ in adapters if lora_on else ():
                    lora.detach(dit)

        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h1 * w1, 128),
                sampling.video_positions(f, h1, w1, fps),
                seed,
                tokens_per_frame=h1 * w1,
                bf16_noise=self.bf16_noise,
            )
            video, extents = conditioned(video, h1, w1, 1.0, seed, True)
            slot_frames = self._slot_frames(generated_keyframes, num_frames)
            video, slots = self._slots_stage1(video, slot_frames, h1, w1, fps, seed)
            if slots is not None:
                extents = np.concatenate(
                    [extents, *([extents[: h1 * w1]] * len(slot_frames))]
                )
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise
            )
            v1, a1 = run(
                "stage1",
                video,
                extents,
                audio,
                fast.sigmas1(sampling.DISTILLED_SIGMAS),
                h1,
                w1,
                True,
                tiles_for(h1, w1),
            )
        if skip_stage_2:
            return Result(
                sampling.unpatchify(v1[:, : f * h1 * w1], (f, h1, w1)),
                a1,
                num_frames,
                height // 2,
                width // 2,
                fps,
                tm,
                keyframes=self._decode_keyframes(
                    v1, slots, slot_frames, h1, w1, num_frames, decode_with_keyframes
                ),
                predicted_seconds=predicted,
            )
        with tm.span("upscale"):
            half = sampling.unpatchify(v1[:, : f * h1 * w1], (f, h1, w1))
            up = vae.normalize(upscaler(vae.denormalize(half)))
            mx.eval(up)
            slots_up = self._slots_upscaled(v1, slots, slot_frames, h1, w1)
        h2, w2 = h1 * 2, w1 * 2
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
            video, extents = conditioned(video, h2, w2, s0, seed + 2, stage_2_ic_lora)
            video, slots2 = self._slots_stage2(
                video, slot_frames, h2, w2, fps, slots_up, s0, seed + 2
            )
            if slots2 is not None:
                extents = np.concatenate(
                    [extents, *([extents[: h2 * w2]] * len(slot_frames))]
                )
            audio = sampling.noised_state(
                (1, audio_t, 128),
                apos,
                seed + 2,
                sigma=s0,
                initial=a1,
                bf16_noise=self.bf16_noise,
            )
            v2, a2 = run(
                "stage2",
                video,
                extents,
                audio,
                sigmas2,
                h2,
                w2,
                stage_2_ic_lora,
                tiles_for(h2, w2),
            )
        return Result(
            sampling.unpatchify(v2[:, : f * h2 * w2], (f, h2, w2)),
            a2,
            num_frames,
            height,
            width,
            fps,
            tm,
            keyframes=self._decode_keyframes(
                v2, slots2, slot_frames, h2, w2, num_frames, decode_with_keyframes
            ),
            predicted_seconds=predicted,
        )

    # ---- Dub-It ------------------------------------------------------------------
    def dubit(
        self,
        prompt: str,
        reference_video: str,
        loras: list[tuple[str, float]],
        height: int = 1024,
        width: int = 1536,
        seed: int = 42,
        reference_strength: float = 1.0,
        images: list[sampling.Still] | None = None,
        image: str | bytes | None = None,
        image_strength: float = 1.0,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        fast: Fast | None = None,
        generated_keyframes: int | list[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> Result:
        """Lightricks' DubItPipeline: the distilled transformer with one Dub-It
        IC-LoRA in both stages; the reference clip (its frame count, snapped
        to 8k + 1, and rate are the output's) is encoded with the default
        tiling at each stage's size and appended as video reference tokens;
        its audio is encoded and appended after the target audio as frozen
        reference tokens with negative times (stage 1), and stage 2 freezes
        the stage-1 audio and appends it as its own reference. The new speech
        is generated by the model in stage 1 (the prompt's words, the
        reference's voice and scene) and the picture re-rendered to it, lips
        included; the stage-1 audio ships."""
        from slimserve.video.ltx25 import media
        from slimserve.video.ltx25.lora import Lora

        if self.variant != "distilled":
            raise ValueError("Dub-It runs on the distilled transformer")
        if len(loras) != 1:
            raise ValueError("Dub-It takes exactly one adapter (the Dub-It IC-LoRA)")
        fast = fast or Fast()
        key = str(Path(loras[0][0]).expanduser().resolve())
        lora = self.user_lora_cache.get(key)
        if lora is None:
            lora = self.user_lora_cache[key] = Lora.from_path(key).load()
        lora_strength = loras[0][1]
        downscale = lora.reference_downscale
        info = media.probe(reference_video)
        num_frames = max(1, (info.frames - 1) // 8 * 8 + 1)
        fps = info.fps
        tm = Timings()
        with tm.span("text"):
            video_text, audio_text = self.load_text().encode(prompt)[:2]
            mx.eval(video_text, audio_text)
            if not keep_text:
                self.unload_text()
        with tm.span("load"):
            dit = self.load_dit()
            vae = self.load_vae()
            upscaler = self.load_upscaler()
        height, width = sampling.snap_dimensions(height, width, two_stage=True)
        f, h1, w1 = sampling.video_latent_shape(num_frames, height // 2, width // 2)
        audio_t = sampling.audio_token_count(num_frames, fps)
        apos = sampling.audio_positions(audio_t)
        stepper = self._stepper(tm, on_step)
        stills = self._prepare_images(image, image_strength, images, num_frames)
        ref_audio = self._source_audio_tokens(
            reference_video, info, num_frames, fps, tm
        )
        if ref_audio is None:
            raise ValueError(f"no audio stream in {reference_video}")

        def with_reference(video, h, w, sigma, stage_seed):
            video = self._condition(
                video, self._image_latents(stills, h, w, tm), fps, sigma, stage_seed
            )
            if (h * 32) % downscale or (w * 32) % downscale:
                raise ValueError(
                    f"{w * 32}x{h * 32} must be divisible by the adapter's "
                    f"reference_downscale_factor {downscale}"
                )
            latent = self._source_video_latent(
                reference_video,
                info,
                h * 32 // downscale,
                w * 32 // downscale,
                tm,
                0.0,
                num_frames,
                fps,
            )
            _, _, rf, rh, rw = latent.shape
            video = sampling.append_reference(
                video,
                sampling.patchify(latent),
                sampling.video_positions(rf, rh, rw, fps),
                downscale,
                reference_strength,
            )
            mx.eval(video.latent, video.clean, video.positions)
            return video

        lora.attach(dit, lora_strength)
        try:
            with tm.span("stage1"):
                video = sampling.noised_state(
                    (1, f * h1 * w1, 128),
                    sampling.video_positions(f, h1, w1, fps),
                    seed,
                    tokens_per_frame=h1 * w1,
                    bf16_noise=self.bf16_noise,
                )
                video = with_reference(video, h1, w1, 1.0, seed)
                slot_frames = self._slot_frames(generated_keyframes, num_frames)
                video, slots = self._slots_stage1(video, slot_frames, h1, w1, fps, seed)
                audio = sampling.noised_state(
                    (1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise
                )
                audio = sampling.append_audio_reference(audio, ref_audio)
                mx.eval(audio.latent, audio.positions)
                v1, a1 = sampling.euler_loop(
                    Denoiser(dit, video_text, audio_text),
                    video,
                    audio,
                    fast.sigmas1(sampling.DISTILLED_SIGMAS),
                    on_step=stepper("stage1"),
                    step_cache=fast.cache(),
                )
                a1 = a1[:, :audio_t]
            with tm.span("upscale"):
                half = sampling.unpatchify(v1[:, : f * h1 * w1], (f, h1, w1))
                up = vae.normalize(upscaler(vae.denormalize(half)))
                mx.eval(up)
                slots_up = self._slots_upscaled(v1, slots, slot_frames, h1, w1)
            h2, w2 = h1 * 2, w1 * 2
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
                video = with_reference(video, h2, w2, s0, seed + 2)
                video, slots2 = self._slots_stage2(
                    video, slot_frames, h2, w2, fps, slots_up, s0, seed + 2
                )
                # stage-1 audio frozen, and appended as its own reference
                audio = sampling.LatentState(
                    latent=a1,
                    clean=a1,
                    denoise_mask=mx.zeros((1, audio_t, 1), dtype=G),
                    positions=apos,
                    frozen=True,
                )
                audio = sampling.append_audio_reference(audio, a1)
                mx.eval(audio.latent, audio.positions)
                v2, _ = sampling.euler_loop(
                    Denoiser(
                        dit,
                        video_text,
                        audio_text,
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
            a1,
            num_frames,
            height,
            width,
            fps,
            tm,
            keyframes=self._decode_keyframes(
                v2, slots2, slot_frames, h2, w2, num_frames, decode_with_keyframes
            ),
        )

    # ---- text to audio ----------------------------------------------------------
    def t2a(
        self,
        prompt: str,
        num_frames: int | None = None,
        fps: float = 24.0,
        seed: int = 42,
        negative_prompt: str | None = None,
        steps: int = 30,
        audio_guidance: sampling.Guidance = DEFAULT_AUDIO_GUIDANCE,
        batched: bool = True,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
        fast: Fast | None = None,
        max_num_frames: int | None = None,
    ) -> Result:
        """Lightricks' T2AOneStagePipeline: audio only, one guided stage on the
        dev transformer with no video modality (the blocks run their audio
        half: self-attention, text cross-attention, feed-forward); the audio
        length is `num_frames` / `fps` (the duration head's pick from the
        audio connector's tokens when not given); the audio guider as the
        guided profiles (CFG 7, STG on block 28, no modality term). Ships a
        WAV."""
        if self.variant != "dev":
            raise ValueError("the T2A pipeline needs LTX25Engine(variant='dev')")
        fast = fast or Fast()
        if fast.steps:
            steps = fast.steps
        if fast.guidance:
            audio_guidance = replace(audio_guidance, **fast.guidance)
        audio_guidance = replace(audio_guidance, modality=1.0)  # no video to isolate
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
            predicted = None
            if num_frames is None:
                num_frames, predicted = self.load_duration().num_frames(
                    None, cond[1], fps
                )
                if max_num_frames is not None:
                    num_frames = min(num_frames, max_num_frames)
            if not keep_text:
                self.unload_text()
        with tm.span("load"):
            dit = self.load_dit()
        audio_t = sampling.audio_token_count(num_frames, fps)
        stepper = self._stepper(tm, on_step)
        with tm.span("stage1"):
            audio = sampling.noised_state(
                (1, audio_t, 128),
                sampling.audio_positions(audio_t),
                seed + 1,
                bf16_noise=self.bf16_noise,
            )
            # the video guider is absent upstream: neutral here so no video
            # pass is planned
            guided = GuidedDenoiser(
                dit,
                cond,
                neg,
                sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0, rescale=0.0),
                audio_guidance,
                batched,
            )
            a = sampling.euler_loop_audio(
                guided,
                audio,
                sampling.ltx2_schedule(steps, 4096),
                on_step=stepper("stage1"),
            )
        return Result(None, a, num_frames, 0, 0, fps, tm, predicted_seconds=predicted)

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
        lora_stage_1: float = HQ_LORA_STAGE_1,
        lora_stage_2: float = HQ_LORA_STAGE_2,
        generated_keyframes: int | list[int] = 0,
        decode_with_keyframes: bool = False,
    ) -> Result:
        """Lightricks' TI2VidTwoStagesHQPipeline: the dev transformer with the
        distilled LoRA at 0.25, 15 guided res_2s steps (CFG 3 / 7, no STG,
        modality 3, rescale 0.45 / 1.0) on the token-count-shifted schedule at
        half resolution; 2x latent upscale; 3 res_2s steps at full resolution
        with the LoRA at 0.5, audio re-noised and refined alongside (it ships
        from stage 2). The SDE noise streams are seeded from the request seed
        (upstream leaves them at their default seed). `lora_stage_1` /
        `lora_stage_2` are upstream's --distilled-lora-strength-stage-1 / -2."""
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
            slot_frames = self._slot_frames(generated_keyframes, num_frames)
            video, slots = self._slots_stage1(video, slot_frames, h1, w1, fps, seed)
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise
            )
            guided = GuidedDenoiser(dit, cond, neg, video_guidance, audio_guidance)
            lora.attach(dit, lora_stage_1)
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
            slots_up = self._slots_upscaled(v1, slots, slot_frames, h1, w1)
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
            video, slots2 = self._slots_stage2(
                video, slot_frames, h2, w2, fps, slots_up, s0, seed + 2
            )
            audio = sampling.noised_state(
                (1, audio_t, 128),
                apos,
                seed + 2,
                sigma=s0,
                initial=a1,
                bf16_noise=self.bf16_noise,
            )
            lora.attach(dit, lora_stage_2)
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
            keyframes=self._decode_keyframes(
                v2, slots2, slot_frames, h2, w2, num_frames, decode_with_keyframes
            ),
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
        if result.video_latent is None:  # T2A: a WAV (upstream encode_audio)
            with tm.span("audio_decode"):
                waveform, sample_rate = self.load_audio().decode(result.audio_tokens)
            with tm.span("mux"):
                path = Path(path)
                if path.suffix.lower() != ".wav":
                    path = path.with_suffix(".wav")
                mux.write_wav(str(path), waveform, sample_rate)
            return path
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
            if result.source_audio is not None:
                waveform, sample_rate = result.source_audio
                end = round(result.num_frames / result.fps * sample_rate)
                waveform = waveform[..., :end]
            else:
                waveform, sample_rate = self.load_audio().decode(result.audio_tokens)
        with tm.span("mux"):
            frames = frames[: result.num_frames]
            mux.write_mp4(str(path), frames, result.fps, waveform, sample_rate)
        return Path(path)
