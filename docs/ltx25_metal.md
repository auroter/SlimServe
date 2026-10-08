# LTX-2.5 video generation on Apple Silicon

SlimServe serves Lightricks' LTX-2.5 (22B audio+video diffusion transformer)
on Apple Silicon through its own engine, `slimserve/video/ltx25/`. It loads
the official Lightricks safetensors directly, runs on MLX, and produces an
H.264 + AAC mp4. Three profiles, one per upstream pipeline:

| profile | pipeline | what it is for |
| --- | --- | --- |
| `ltx25-distilled` | DistilledPipeline: 8 + 3 steps, no guidance | fast iteration |
| `ltx25-dev` | TI2VidTwoStages: 30 guided steps (CFG 3 / STG 1 / modality 3), then 3 with the distilled LoRA | quality |
| `ltx25-dfr` | DFRPipeline: distilled flow with generated keyframe slots and the detailing IC-LoRA | Lightricks' production path |

Distilled versus dev is a quality-versus-time choice for the person asking,
not something the server picks.

## Use

```
slimserve ltx25-distilled --dry-run                 # show the resolved plan
slimserve ltx25-distilled -y                        # fetch weights if needed, then serve on :8000
slimserve ltx25-dev -p "A red fox trotting through a snowy pine forest at dawn" --output fox.mp4
slimserve ltx25-dfr -p "..." --size 768x512 --seconds 5 --seed 7
slimserve ltx25-distilled -p "the fox turns and runs" --image fox.png     # image-to-video
slimserve ltx25-distilled-fast -p "..."                                   # the fast tier (see below)
slimserve ltx25-hq -p "..."                                               # Lightricks' HQ preset (res_2s)
slimserve ltx25-dev -p "a cat watches rain" --enhance-prompt                # Gemma rewrites the prompt first
```

The weights are gated: the Hugging Face token on the machine must have
accepted `Lightricks/LTX-2.5` and, for `ltx25-dfr`,
`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`. The prompt enhancer
(`google/gemma-4-E2B-it`, 10.3 GB, every profile) is public. `ffmpeg` must be
on PATH.

Serving API (a clip takes minutes, so it is job-shaped):

```
POST   /v1/videos               {"prompt": "...", "size": "1536x1024", "seconds": 5, "seed": 42,
                                 "enhance_prompt": false,
                                 "image": "<base64 or data: URL>", "image_strength": 1.0}
GET    /v1/videos/<id>          status (queued | in_progress | completed | failed), progress, timings
GET    /v1/videos/<id>/content  the mp4
DELETE /v1/videos/<id>
GET    /health, /v1/models
```

`ltx25-dfr` takes two more options (upstream's `--temporal-upscalings` and
`--spatial-upscalings`; CLI flags of the same names): `temporal_upscalings`
1 or 2 doubles the frame rate per round (the canvas is x2 temporally
upsampled and re-denoised in keyframe-seam windows; a 5 s clip at 24 fps
ships at 48 or 96 fps) and `spatial_upscalings` 2 runs the first two stages
at a quarter and half of the output size and adds the tiled full-resolution
detailing epilogue (sizes must be multiples of 128; the way to a sharp
2048x1024 from the same base as a 1024x512 clip).

`"wait": true` holds the POST open until the clip is done. A request with
neither `seconds` nor `num_frames` gets its length from the model's duration
head (Lightricks' auto-duration: the clip the prompt implies, 1-20 s, snapped
to the 8k + 1 frame grid, capped at the profile's envelope for that size); the
job reports the pick as `num_frames` and `predicted_seconds`. `negative_prompt`
is accepted by `ltx25-dev` and `ltx25-hq` (the guided pipelines). `decoder` is `diffusion` (default, Lightricks'
recommended decoder: sharper faces, textures and text) or `conv` (about 4x
faster decode); the CLI flag is `--decoder`. Width and height are multiples of 64, frame
counts are 8k + 1. Requests larger than the profile's validated clip
(1536x1024x121, 24,576 latent tokens) are refused with a 400: that envelope is
what was measured to fit in memory. There is one GPU, so requests run one at
a time in arrival order with the weights resident; up to 16 may wait (then
429). Finished clips are kept under `$SLIMSERVE_VIDEO_DIR`
(default `~/.cache/slimserve/videos`), the most recent 32.

### Prompt enhancement

`enhance_prompt: true` (CLI `--enhance-prompt`; off by default, as upstream's
`--enhance-prompt`) rewrites the request into the long caption style the
model was trained on before anything else runs. LTX-2.5's own text encoder is
a fine-tune that cannot generate, so Lightricks' pipelines do this with a
separate generative instruct Gemma (`--prompt-enhancer-gemma-root`); the
profiles fetch the one their README names, `google/gemma-4-E2B-it`, and run
its language model in MLX (`enhancer.py`: 35 layers with per-layer input
embeddings, a shared-K/V tail, 512-token sliding and global attention). It is
loaded for the rewrite and dropped again (4.3 GiB in fp16, 0.8 s to load; the
4.7 GiB per-layer embedding table is never loaded, its rows are read from the
file), so the measured resident set is unchanged. The recipe is
upstream's exactly: its T2V system prompt (`prompts/`), `user prompt: ...`
through Gemma's chat template, greedy decoding with no repeated 5-gram, at
most 600 new tokens, curly quotes and a leading non-letter run cleaned off. A
rewrite takes about 5 s (a 160-220 token caption) and is reported as the job's
`enhanced_prompt` and `enhance` span; the CLI prints it.

Checked against transformers' fp32 Gemma-4 on the CPU
(`perf/ltx25_harness/n11_enhancer_parity.py`): same prompt tokens, first-step
logits within 2.6e-4 (relative), and teacher-forced logits at every one of 218
generated positions with the same argmax (median 3.2e-4, max 2.9e-3). Free
running, one of two test prompts reproduced the reference caption token for
token; the other followed it for 118 tokens and then took the other side of a
0.01-logit tie (`,` 17.879 vs ` with` 17.869 in the reference), which is below
the fp16-vs-fp32 error and gives an equally valid caption. Greedy decoding is
deterministic on a given machine.

An image request is enhanced with the still, as upstream's `enhance_i2v`:
Gemma gets the I2V system prompt and sees the image before `User Raw Input
Prompt: ...`. The still is decoded like the conditioning frame and scaled to a
896 long side with torch's uint8 bilinear arithmetic (`image.py
resize_bilinear_uint8`, bit-exact: the vision tower turns the one-level
rounding differences of a float resize into a 10% change of the soft tokens),
fitted by Gemma's PIL image processor onto a 48-pixel grid of at most 2520
patches (bicubic), and encoded by the vision tower (16 layers, 2-D RoPE,
clamped linears) into at most 280 soft tokens that take the place of the
`<|image|>` placeholders. The tower runs with fp32 operands (0.2 s per still;
in fp16 its output is off by 5e-3, in fp32 by 4e-6). On the I2V reference
(a snowy forest with a fox, a prompt about a woman and a window) ours is
identical to transformers for all 191 generated tokens: Gemma describes what
is in the image, then folds the request in.

### Image-to-video

`image` makes a still the clip's first frame, on every pipeline. The engine
follows Lightricks' `ltx_pipelines` conditioning path exactly:

- The still is decoded to sRGB (EXIF orientation applied), then re-compressed
  once as a single H.264 frame (libx264, preset veryfast, yuv420p) at CRF 18
  and decoded back (upstream `media_io/decode.py preprocess`; CRF 18 is
  `LTX_2_4_IMAGE_CRF`, the value for 2.4+ checkpoints, not the 2.0-era 33).
  The model was trained on video frames that carry H.264 compression
  statistics; a pristine PNG conditions it off-distribution and the first
  frame then "pops". Odd edges are trimmed to even sizes first, as the codec
  requires. The round trip runs through the `ffmpeg` binary (as the muxer
  does), with bilinear chroma scaling on both legs like PyAV's default.
- It is resized to fill the target (scale = max(H/src_h, W/src_w), ceil'd
  sizes, bilinear, `align_corners=False`, no antialias), center-cropped and
  mapped to [-1, 1] by `x / 127.5 - 1` (`resize.py resize_and_center_crop`,
  `range_map.py normalize_images`).
- Both stages are conditioned (upstream
  `chunks/conditionings.py image_conditionings_for_chunk` encodes at each
  chunk's own pixel size): in stage 1 the still is encoded at the half
  resolution through the conv VAE encoder, in stage 2 at the full
  resolution. Each encoding is written into the latent state's clean tokens
  at latent frame 0 with denoise mask `1 - image_strength`
  (`ltx_core latent_cond.py _apply_condition_by_latent_index`), and the
  noised state is `lerp(clean, noised, mask)` so a strength-1 frame starts
  clean and stays clean through the sampler; the transformer sees per-token
  timesteps `mask * sigma`. In DFR this happens before the keyframe slots
  and the reference tokens are appended.

CLI: `--image PATH` and `--image-strength` (0-1, default 1.0). API: `image`
is the encoded still (PNG, JPEG, ...) as base64, optionally a
`data:image/png;base64,...` URL, with `image_strength`; undecodable payloads
are a 400. The encode is timed as the `image` span (both stages summed).

### The fast tier

`ltx25-distilled-fast`, `ltx25-dev-fast` and `ltx25-dfr-fast` are the same
engine with output-changing settings stacked on: they are for iterating, and
for anyone who judges a clip by eye rather than against a reference. Reviewed
at 1536x1024x121: no meaningful difference from the exact clips in any mode;
dev (exact or fast) is clearly the best of the three in quality and in prompt
adherence, so `ltx25-dev-fast` is the profile to reach for first. The
exact profiles stay the reference the fast ones are measured against. At
1536x1024x121 on the M1 Ultra (ledger section 24):

| profile | exact | fast | what is on |
| --- | ---: | ---: | --- |
| distilled | 413 s | **278 s** (229 s with the opt-in 5-step stage 1) | conv decoder; 2-step stage 2 |
| dev | 1513 s | **911 s** | 20 steps; first-block step cache 0.10; conv decoder (all four guidance passes kept) |
| DFR | 681 s | **421 s** | 2-step stage 2; 2x2-tiled stage-2 attention; conv decoder |

`ltx25-hq` (Lightricks' HQ preset, res_2s) takes 1902 s at the same size: half
the steps of dev but two model evaluations per step and the distilled LoRA
in both stages.

Every lever was measured alone first, with a PSNR against the exact clip and a
look at the frames. Kept: fewer refinement steps (30-33 dB, the same shot),
the conv decoder (35 dB), on dev fewer steps and the step cache
(21-23 dB each, the same shot at sheet scale), on DFR tiled attention (28 dB,
no seam: the reference tokens anchor every tile). Rejected: dropping dev's
modality guidance pass (its footsteps come out 4 dB softer on top of the 3 dB
that fewer steps already cost; a less-work-less-punch tradeoff, as with fine
detail, so the profile keeps all four passes), tiled attention
on distilled and dev (a seam through faces at the frame centre), the step
cache on the ancestral schedules (the block-0 residual moves 22-70% per step,
so it never fires), and 5-step distilled stage 1 is left out of the default
because it changes the composition (a different sample, not a worse one).
The settings are the profile's `engine.fast` block (`pipeline.Fast`); a
profile's request envelope and API are otherwise identical.

## Measured (M1 Ultra 64-core GPU, 128 GiB, macOS 15.7.2, seed 42)

Baseline is dgrauet/ltx-2-mlx v0.15.12, the fastest LTX-2.5 path on a Mac
before this work, in its recommended q8 pack and its bf16 pack. Wall time is
a cold process, prompt to mp4, including model loads.

| clip | pipeline | baseline q8 | baseline bf16 | SlimServe | vs q8 |
| --- | --- | ---: | ---: | ---: | ---: |
| 768x512x121 | distilled, conv decoder | 116.9 s | 202.4 s | **86 s** cold, **71 s** resident | 1.36x |
| 768x512x121 | dev, 30 steps | 606.9 s | | **343 s** | 1.77x |
| 768x512x121 | DFR | 170.9 s | | **125 s** | 1.37x |
| 1536x1024x121 | distilled | 504.9 s (tiled attention and decode; untiled dies in decode) | | **343 s**, untiled | 1.47x |

The baseline rows use its conv decoder; with the diffusion decoder (now the
default, ledger section 18) add about 23 s to a 768x512 clip and 127 s to an
HD one. SlimServe runs the unquantized official weights; the q8 baseline is an 8-bit
pack. Per-forward accuracy against an fp32 run of the same weights on
identical inputs: SlimServe rel-L2 0.0069, the baseline as shipped 0.042.
Evidence, per-op profiles and every intermediate number are in
`perf/ltx25_metal_campaign.md`.

## How it is fast

1. **Precision path.** On M1-M4 the only matrix-multiply-accumulate the GPU
   has is fp16. The baseline's fp32 AdaLN tables silently promote its whole
   residual stream to fp32, so every GEMM runs at a third of the fp16 rate.
   The engine keeps the residual stream, norms, AdaLN, RoPE tables, x0
   recovery and the model boundary in fp32 and casts to fp16 exactly at each
   GEMM and attention operand. The shipped bf16 weights convert to fp16
   exactly. This is 2.4x on the transformer and more accurate than the
   baseline. Nothing is quantized.
2. **Split-K GEMM.** MLX's fp16 GEMM falls from 19 to 10 TF/s when K reaches
   16384 (the FFN's second linear), and to 6 TF/s at 24k rows. Summing eight
   K=2048 GEMMs stays at 18-19 TF/s: 1.8 s per forward at 6k tokens, about
   17 s per forward at HD.
3. **Fused glue.** AdaLN modulate + cast, gated residual, GELU and the
   per-head gate are compiled chains, one kernel pass each. Bit-identical.
4. **Slab VAE.** MLX's conv3d scratch scales with the whole clip. Every
   convolution runs in temporal slabs, with the activation and padding done
   inside the slab. Exact output; the 121-frame decode went from about 60 GiB
   to 10 GiB and HD decodes untiled in 34 s at 27 GiB.
5. **Batched, de-duplicated guidance.** The dev pipeline's four passes per
   step run as one batch; the STG pass is forked from the conditional pass at
   block 28 instead of recomputed, and the three conditional-text passes
   share their text key/value projections. 10.9 to 9.1 s per step, exact.
6. **Runtime adapters.** The rank-450 distilled LoRA and the detailing
   IC-LoRA are applied as low-rank terms next to the base GEMM. No second
   39 GiB transformer, no reload between stages, exact detach.
7. **Diffusion decoder with a Metal neighborhood-attention kernel.** The
   recommended decoder's 3-D neighborhood attention has no fast MLX op; the
   baseline's formulation took 69 s for 49 frames. One Metal kernel (eight
   lanes per query, fp32 online softmax) decodes the same in 9 s, exact to
   1e-7 against a reference, and the 121-frame clip in 30 s.
8. **Shape-aware convolutions.** MLX's conv3d runs 1024-channel layers on
   small grids at 1.5 TF/s; those run as per-tap split-K GEMMs instead (the
   upscaler 2.5 to 0.45 s), chosen per shape by a persisted measurement.
9. **Compiled filters.** The vocoder's 12-tap anti-aliasing filters are
   compiled multiply-add chains instead of 200 dispatch-bound convolutions
   (audio 2.5 to 0.36 s).
10. **Text path.** Gemma-4 12B runs on the prompt's real tokens only (padding
   is masked and replaced by registers in the connector either way): 0.85 s
   per prompt.

Where it stands against the hardware: the M1 Ultra's measured fp16 MMA
ceiling is 19.4 TF/s (register-resident, section 17 of the ledger). MLX's
GEMM runs at 95-99% of it and its attention at 78%, so what is left on this chip is
small; the larger remaining wins are opt-in (step caching, sparse attention)
and the M5's int8 path. See the ledger's kernel sections for what was tried.

## Memory, and why the engine is strict about it

Metal memory is wired: it cannot be compressed or swapped. Overshooting does
not fail an allocation, it starves the OS (on 2026-10-02 four concurrent
model-loading jobs panicked this machine). The engine therefore:

- caps MLX active memory at RAM - 24 GiB - 16 GiB cache and the buffer cache
  at 16 GiB;
- loads each checkpoint in 4 GiB chunks, dropping the bf16 source as it casts;
- plans the VAE decode from a measured model (5 GiB + 135 B per output
  pixel-frame in fp32) against what the resident models leave, releases the
  text encoder first if that is what makes an untiled decode fit, and tiles
  only after that;
- refuses requests beyond the validated envelope.

Resident set while serving: distilled about 71 GiB (transformer 40, Gemma 28,
VAE/audio 3), dev about 80 GiB (plus the 8 GiB LoRA). The profiles are gated
to 128 GiB machines because that is what was validated.

Development rule (HANDOFF.md): one model-loading process at a time, through
`perf/ltx25_harness/gpu_run.py`.

## Layout

| file | contents |
| --- | --- |
| `checkpoints.py` | official file set, safetensors headers, load-time dtype policy |
| `dit.py` | the transformer forward (upstream weight names), split-K, fused glue, LoRA hook, STG masks, first-block step cache, tiled self-attention |
| `text.py` | Gemma-4 tower, multi-layer feature projection, the two connectors |
| `sampling.py` | latent state, positions, noise, Euler / ancestral Euler, dev schedule and guider, DFR canvas and slots |
| `pipeline.py` | `LTX25Engine`: distilled, dev, dfr, render; memory budgeting; `Fast` (the fast tier's settings) |
| `lora.py` | runtime low-rank adapters |
| `vae.py`, `upscaler.py` | conv VAE (slab conv3d, tiling planner), latent upscalers |
| `image.py` | image-to-video still: decode, CRF-18 round trip, upstream resize/crop/normalize |
| `duration.py` | the duration head (auto clip length from the prompt) |
| `enhancer.py`, `prompts/` | the prompt enhancer: Gemma-4 E2B-it language model and vision tower, greedy decoding, upstream's system prompts |
| `audio.py`, `mux.py` | audio VAE + vocoder + bandwidth extension; ffmpeg mux |
| `../server.py`, `../cli.py` | the job queue and HTTP API; `slimserve` integration |

## Not implemented yet

- Image conditioning beyond the first frame (keyframes at other frame
  indices, video-to-video reference conditioning); the I2V first-frame path
  is wired but its end-to-end output has not yet been compared against
  upstream on this machine.
- M3+/M5 variants: native bf16 and the M5 int8 path are unverified on
  hardware and are separate profile records when they exist.
