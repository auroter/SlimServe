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
```

The weights are gated: the Hugging Face token on the machine must have
accepted `Lightricks/LTX-2.5` and, for `ltx25-dfr`,
`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`. `ffmpeg` must be on
PATH.

Serving API (a clip takes minutes, so it is job-shaped):

```
POST   /v1/videos               {"prompt": "...", "size": "1536x1024", "seconds": 5, "seed": 42}
GET    /v1/videos/<id>          status (queued | in_progress | completed | failed), progress, timings
GET    /v1/videos/<id>/content  the mp4
DELETE /v1/videos/<id>
GET    /health, /v1/models
```

`"wait": true` holds the POST open until the clip is done. `negative_prompt`
is accepted by `ltx25-dev` only. `decoder` is `diffusion` (default, Lightricks'
recommended decoder: sharper faces, textures and text) or `conv` (about 4x
faster decode); the CLI flag is `--decoder`. Width and height are multiples of 64, frame
counts are 8k + 1. Requests larger than the profile's validated clip
(1536x1024x121, 24,576 latent tokens) are refused with a 400: that envelope is
what was measured to fit in memory. There is one GPU, so requests run one at
a time in arrival order with the weights resident; up to 16 may wait (then
429). Finished clips are kept under `$SLIMSERVE_VIDEO_DIR`
(default `~/.cache/slimserve/videos`), the most recent 32.

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
| `dit.py` | the transformer forward (upstream weight names), split-K, fused glue, LoRA hook, STG masks |
| `text.py` | Gemma-4 tower, multi-layer feature projection, the two connectors |
| `sampling.py` | latent state, positions, noise, Euler / ancestral Euler, dev schedule and guider, DFR canvas and slots |
| `pipeline.py` | `LTX25Engine`: distilled, dev, dfr, render; memory budgeting |
| `lora.py` | runtime low-rank adapters |
| `vae.py`, `upscaler.py` | conv VAE (slab conv3d, tiling planner), latent upscalers |
| `audio.py`, `mux.py` | audio VAE + vocoder + bandwidth extension; ffmpeg mux |
| `../server.py`, `../cli.py` | the job queue and HTTP API; `slimserve` integration |

## Not implemented yet

- Image-to-video conditioning (the VAE encoder is ported and parity-checked;
  the conditioning path is not wired).
- DFR temporal rounds and the second spatial epilogue; the duration head;
  the prompt enhancer; the res_2s sampler (HQ pipeline).
- M3+/M5 variants: native bf16 and the M5 int8 path are unverified on
  hardware and are separate profile records when they exist.
