# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 pipelines on the SlimServe engine.

`distilled` mirrors Lightricks' DistilledPipeline: stage 1 at half resolution
(8 ancestral Euler steps, no guidance), 2x latent upscale, stage 2 at full
resolution (3 Euler steps), with the same distilled transformer in both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx

from slimserve.video.ltx25 import checkpoints, sampling
from slimserve.video.ltx25.dit import DiTConfig, LTX25DiT, x0_from_velocity
from slimserve.video.ltx25.sampling import LatentState

G = mx.float32
OS_RESERVE_BYTES = 24 << 30  # never plan Metal memory into the last 24 GiB


@dataclass
class Timings:
    spans: dict[str, float] = field(default_factory=dict)
    steps: list[tuple[str, int, float]] = field(default_factory=list)
    memory: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # peak, active, cache bytes

    def span(self, name: str):
        timings = self

        class _Span:
            def __enter__(self):
                mx.synchronize()
                mx.reset_peak_memory()
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                mx.synchronize()
                timings.spans[name] = timings.spans.get(name, 0.0) + time.perf_counter() - self.t
                timings.memory[name] = (mx.get_peak_memory(), mx.get_active_memory(), mx.get_cache_memory())

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


class Denoiser:
    """x0 prediction for one sampler step; owns the DiT call contract."""

    def __init__(self, dit: LTX25DiT, video_text: mx.array, audio_text: mx.array):
        self.dit, self.video_text, self.audio_text = dit, video_text, audio_text

    def __call__(self, video: LatentState, audio: LatentState, vx: mx.array, ax: mx.array, sigma: float):
        b = vx.shape[0]
        t = mx.full((b,), sigma, dtype=G)
        vt = None if video.uniform else (video.denoise_mask * sigma).squeeze(-1)
        at = None if audio.uniform else (audio.denoise_mask * sigma).squeeze(-1)
        v, a = self.dit(
            vx, ax, t, self.video_text, self.audio_text, video.positions, audio.positions,
            video_keyframes_mask=video.keyframes_mask,
            video_timesteps=vt, audio_timesteps=at,
            video_sigma=mx.zeros((b,), dtype=G) if video.frozen else None,
            audio_sigma=mx.zeros((b,), dtype=G) if audio.frozen else None,
            video_attention_mask=video.attention_mask, audio_attention_mask=audio.attention_mask,
        )
        return x0_from_velocity(vx, v, t if vt is None else vt), x0_from_velocity(ax, a, t if at is None else at)


class GuidedDenoiser:
    """Dev-model x0 with CFG + STG + modality guidance.

    The four passes (conditional, negative prompt, self-attention skipped on
    the STG blocks, audio<->video cross-attention skipped everywhere) run as
    one batch of four: the same FLOPs as four forwards, with every GEMM at 4x
    the rows.
    """

    PASSES = 4

    def __init__(self, dit: LTX25DiT, cond: tuple[mx.array, mx.array], negative: tuple[mx.array, mx.array],
                 video: sampling.Guidance, audio: sampling.Guidance, batched: bool = True):
        self.dit, self.video_g, self.audio_g, self.batched = dit, video, audio, batched
        self.video_text = mx.concatenate([cond[0], negative[0], cond[0], cond[0]], axis=0)
        self.audio_text = mx.concatenate([cond[1], negative[1], cond[1], cond[1]], axis=0)
        keep_stg = mx.array([1.0, 1.0, 0.0, 1.0])
        keep_mod = mx.array([1.0, 1.0, 1.0, 0.0])
        self.stg: dict[tuple[str, int], mx.array] = {}
        for blk in video.stg_blocks:
            self.stg[("video_self", blk)] = keep_stg
        for blk in audio.stg_blocks:
            self.stg[("audio_self", blk)] = keep_stg
        for blk in range(dit.cfg.num_layers):
            self.stg[("a2v", blk)] = keep_mod
            self.stg[("v2a", blk)] = keep_mod

    def _forward(self, video: LatentState, audio: LatentState, vx, ax, sigma: float, rows: slice):
        n = len(range(*rows.indices(self.PASSES)))
        rep = lambda x: None if x is None else mx.repeat(x, n, axis=0)  # noqa: E731
        t = mx.full((n,), sigma, dtype=G)
        vt = None if video.uniform else rep((video.denoise_mask * sigma).squeeze(-1))
        at = None if audio.uniform else rep((audio.denoise_mask * sigma).squeeze(-1))
        v, a = self.dit(
            rep(vx), rep(ax), t, self.video_text[rows], self.audio_text[rows],
            rep(video.positions), rep(audio.positions),
            video_keyframes_mask=rep(video.keyframes_mask), video_timesteps=vt, audio_timesteps=at,
            video_attention_mask=video.attention_mask, audio_attention_mask=audio.attention_mask,
            stg={k: m[rows] for k, m in self.stg.items()},
        )
        return (x0_from_velocity(rep(vx), v, t if vt is None else vt),
                x0_from_velocity(rep(ax), a, t if at is None else at))

    def __call__(self, video: LatentState, audio: LatentState, vx: mx.array, ax: mx.array, sigma: float):
        if self.batched:
            v0, a0 = self._forward(video, audio, vx, ax, sigma, slice(0, 4))
        else:
            parts = [self._forward(video, audio, vx, ax, sigma, slice(i, i + 1)) for i in range(4)]
            v0 = mx.concatenate([p[0] for p in parts], axis=0)
            a0 = mx.concatenate([p[1] for p in parts], axis=0)
        return (self.video_g.combine(v0[0:1], v0[1:2], v0[2:3], v0[3:4]),
                self.audio_g.combine(a0[0:1], a0[1:2], a0[2:3], a0[3:4]))


class LTX25Engine:
    """Resident components plus the pipelines that run on them."""

    def __init__(self, root: Path | None = None, variant: str = "distilled", bf16_noise: bool = False):
        self.root = root or checkpoints.model_root()
        total = int(mx.device_info()["memory_size"])
        mx.set_memory_limit(total - OS_RESERVE_BYTES)
        mx.set_cache_limit(8 << 30)
        self.variant = variant
        self.bf16_noise = bf16_noise
        self.dit: LTX25DiT | None = None
        self.text = None
        self.vae = None
        self.upscaler = None
        self.audio = None
        self.distilled_lora = None

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

    def unload_text(self) -> None:
        if self.text is not None:
            self.text.unload()
            self.text = None

    def decode_budget(self) -> int:
        """Bytes the VAE decode may use: what is left after the resident models
        and the OS reserve. The decoder tiles to fit. Metal memory is wired, so
        overshooting this takes the machine down, not just the process."""
        total = int(mx.device_info()["memory_size"])
        free = total - mx.get_active_memory() - OS_RESERVE_BYTES
        return int(max(4 << 30, min(free, total // 2)))

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

    # ---- distilled --------------------------------------------------------
    def distilled(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1536,
        num_frames: int = 121,
        fps: float = 24.0,
        seed: int = 42,
        text_embeds: tuple[mx.array, mx.array] | None = None,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
    ) -> Result:
        tm = Timings()
        if text_embeds is None:
            with tm.span("text"):
                video_text, audio_text = self.load_text().encode(prompt)[:2]
                mx.eval(video_text, audio_text)
            if not keep_text:
                self.unload_text()
        else:
            video_text, audio_text = text_embeds
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

        # Stage 1: half resolution, from pure noise, ancestral Euler.
        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h1 * w1, 128), sampling.video_positions(f, h1, w1, fps), seed,
                tokens_per_frame=h1 * w1, bf16_noise=self.bf16_noise)
            audio = sampling.noised_state((1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise)
            v1, a1 = sampling.euler_ancestral_loop(
                denoise, video, audio, sampling.DISTILLED_SIGMAS,
                noise_seed=seed + sampling.ANCESTRAL_NOISE_SEED_OFFSET, on_step=stepper("stage1"))

        # 2x latent upscale in the VAE's denormalized latent space.
        with tm.span("upscale"):
            half = sampling.unpatchify(v1, (f, h1, w1))
            up = vae.normalize(upscaler(vae.denormalize(half)))
            mx.eval(up)

        # Stage 2: full resolution refinement from sigma 0.909, plain Euler.
        with tm.span("stage2"):
            h2, w2 = h1 * 2, w1 * 2
            s0 = sampling.STAGE_2_DISTILLED_SIGMAS[0]
            video = sampling.noised_state(
                (1, f * h2 * w2, 128), sampling.video_positions(f, h2, w2, fps), seed + 2, sigma=s0,
                initial=sampling.patchify(up), tokens_per_frame=h2 * w2, bf16_noise=self.bf16_noise)
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 2, sigma=s0, initial=a1, bf16_noise=self.bf16_noise)
            v2, a2 = sampling.euler_loop(
                denoise, video, audio, sampling.STAGE_2_DISTILLED_SIGMAS, on_step=stepper("stage2"))

        return Result(sampling.unpatchify(v2, (f, h2, w2)), a2, num_frames, height, width, fps, tm)

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
        num_frames: int = 121,
        fps: float = 24.0,
        seed: int = 42,
        negative_prompt: str | None = None,
        steps: int = 30,
        video_guidance: sampling.Guidance = sampling.Guidance(cfg=3.0),
        audio_guidance: sampling.Guidance = sampling.Guidance(cfg=7.0),
        batched: bool = True,
        keep_text: bool = True,
        on_step: Callable[[str, int, float], None] | None = None,
    ) -> Result:
        """Lightricks' TI2VidTwoStagesPipeline: guided dev stage 1 at half
        resolution, 2x latent upscale, 3-step stage 2 with the distilled LoRA
        and no guidance. Audio is taken from stage 1, as upstream."""
        if self.variant != "dev":
            raise ValueError("the dev pipeline needs LTX25Engine(variant='dev')")
        tm = Timings()
        with tm.span("text"):
            text = self.load_text()
            cond = text.encode(prompt)[:2]
            neg = text.encode(sampling.DEFAULT_NEGATIVE_PROMPT if negative_prompt is None else negative_prompt)[:2]
            mx.eval(cond, neg)
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

        with tm.span("stage1"):
            video = sampling.noised_state(
                (1, f * h1 * w1, 128), sampling.video_positions(f, h1, w1, fps), seed,
                tokens_per_frame=h1 * w1, bf16_noise=self.bf16_noise)
            audio = sampling.noised_state((1, audio_t, 128), apos, seed + 1, bf16_noise=self.bf16_noise)
            guided = GuidedDenoiser(dit, cond, neg, video_guidance, audio_guidance, batched)
            v1, a1 = sampling.euler_loop(
                guided, video, audio, sampling.ltx2_schedule(steps, f * h1 * w1), on_step=stepper("stage1"))

        with tm.span("upscale"):
            up = vae.normalize(upscaler(vae.denormalize(sampling.unpatchify(v1, (f, h1, w1)))))
            mx.eval(up)

        with tm.span("stage2"):
            h2, w2 = h1 * 2, w1 * 2
            s0 = sampling.STAGE_2_DISTILLED_SIGMAS[0]
            video = sampling.noised_state(
                (1, f * h2 * w2, 128), sampling.video_positions(f, h2, w2, fps), seed + 2, sigma=s0,
                initial=sampling.patchify(up), tokens_per_frame=h2 * w2, bf16_noise=self.bf16_noise)
            audio = sampling.noised_state(
                (1, audio_t, 128), apos, seed + 2, sigma=s0, initial=a1, bf16_noise=self.bf16_noise)
            lora.attach(dit)
            try:
                v2, _ = sampling.euler_loop(
                    Denoiser(dit, *cond), video, audio, sampling.STAGE_2_DISTILLED_SIGMAS, on_step=stepper("stage2"))
            finally:
                lora.detach(dit)

        return Result(sampling.unpatchify(v2, (f, h2, w2)), a1, num_frames, height, width, fps, tm)

    # ---- decode -----------------------------------------------------------
    def render(self, result: Result, path: str | Path, seed: int = 42) -> Path:
        from slimserve.video.ltx25 import mux

        tm = result.timings
        with tm.span("vae_decode"):
            frames = self.load_vae().decode(
                result.video_latent, frame_rate=result.fps, budget_bytes=self.decode_budget())
        with tm.span("audio_decode"):
            waveform, sample_rate = self.load_audio().decode(result.audio_tokens)
        with tm.span("mux"):
            frames = frames[: result.num_frames]
            mux.write_mp4(str(path), frames, result.fps, waveform, sample_rate)
        return Path(path)
