# LTX-2.5 on Apple Silicon: campaign plan (drafted 2026-10-01)

Companion: `perf/ltx25_metal_research.md` (survey with URLs). This is the plan of
record; decisions below were made with Sean on 2026-10-01 and are not open.

## 1. Decisions

**Scope: the whole M line, one kernel codebase.** Bring-up and the first
profile are on the M1 Ultra 128 GiB Studio. Kernels are written once with
per-chip variants (tile shapes, bf16 native on M3+, int8 tensor path on M5),
the way llama.cpp handles it. Platform remains part of profile identity.

**Three pipeline ids, same DiT, same kernels.**

| id (working) | pipeline | stage 1 | stage 2 | position |
| --- | --- | --- | --- | --- |
| `ltx25-distilled` | DistilledPipeline | 8 forwards, half res, CFG 1 | 2x latent upscale, 3 steps, distilled LoRA | fast / iterate |
| `ltx25-dev` | TI2VidTwoStages | 30 steps x 4 forwards (CFG 3.0, STG 1.0, modality 3.0), half res | same | quality |
| `ltx25-dfr` | DFRPipeline | distilled + generated keyframe slots | upscale, spatial-detailing epilogue with the official detailing IC-LoRA (strength 0.5), optional temporal rounds (+8 steps each) | production (Lightricks' label) |

Dev and distilled are the same architecture and shapes; dev's 4-way guidance
batch is just M=4x rows. DFR adds a 0.33 GB IC-LoRA and tiling logic, no new
kernels. Default output 1536x1024x121 @ 24 fps (stage 1 at 768x512).

**Precision on M1-M4: no quantization.** bf16 checkpoints as shipped, fp16
MMA operands with fp32 accumulate, fp32 glue (residual stream, norms, AdaLN,
RoPE, softmax statistics, x0 recovery). Grounds, measured on this box
(`scratchpad/gemm_bench.py`, 2026-10-01):

| M tokens | fp16 | bf16 (emulated) | MLX int8 g64 | MLX int4 g64 |
| ---: | ---: | ---: | ---: | ---: |
| 1,536 | 18.5 TF/s | 13.5 | 15.4 | 15.4 |
| 6,144 | 19.3 | 13.9 | 15.8 | 15.8 |
| 14,080 | 18.6 | 14.0 | 15.9 | 15.9 |

M1 has only fp16 simdgroup MMA; every format is converted to fp16 in-register
at tile load, so a quant can match fp16 but never beat it, and bf16 compute
costs 25-30%. Weight traffic is 38 GB/forward = 48 ms at 800 GB/s against
seconds of compute. Memory does not force a quant: dev DiT 38 + runtime
low-rank distilled LoRA 9 + Gemma 24 + VAE/upscalers 3 + activations <2 GB
fits 128 GiB with room for untiled VAE decode. Gate: section 4.

**Precision on M5: official int8 convrot (W8A8, Hadamard g256, 5 bf16
islands) as the candidate**, because the neural accelerators do int8 MMA
(Draw Things: 1.6-1.9x int8 GEMM over fp16 on M5). A custom calibrated quant
only if the official one loses quality in the A/B. Nothing to build now.

**Baseline: dgrauet/ltx-2-mlx** (v0.15.12, cloned at
`~/.local/scratch/ltx25/ltx-2-mlx`, installed with uv). Most complete 2.5
port, fastest measured Mac numbers, engine under ltx-video-mac (422 stars)
and Rapid-MLX (3.9k). Baseline config = its recommended q8 pack; bf16 pack
also timed. Secondary bar: ComfyUI on MPS (popular GGUF path; 2-3.5x slower,
currently broken on M1 Ultra). Draw Things becomes a second bar if it ships
2.5 (it has real Metal kernels; stops at 2.3 today).

**Weights** (`~/models/ltx-2.5/`): `official/` (Lightricks/LTX-2.5 bf16 set
+ detailing IC-LoRA), `dgrauet-q8/`, `dgrauet-bf16/`. HF account auroter
holds the gate acceptances. int8-convrot and nvfp4 deferred to M5 work.

## 2. Physics on M1 Ultra (est., 1536x1024x121 output)

Peak ~20.8 TF/s fp16, 800 GB/s. Tokens: 6,144 at stage 1, 24,576 at stage 2.
Forward ~0.21 PFLOP at 6k (attn ~15%), ~1.2 PFLOP at 24k (attn ~40%).

| pipeline | DiT forwards | PFLOP | 100% ALU | ~70% ALU |
| --- | --- | ---: | ---: | ---: |
| distilled | 8 @ 6k + 3 @ 24k | ~5.3 | ~4.2 min | ~6 min |
| dev two-stage | 120 @ 6k + 3 @ 24k | ~29 | ~23 min | ~32 min |
| DFR | distilled + epilogue (+ temporal rounds) | > distilled, measure | | |

MLX `fast.scaled_dot_product_attention`, 32 heads x 128, self-attention
(`scratchpad/sdpa_bench.py`, 2026-10-01):

| B x N | fp16 | bf16 |
| --- | ---: | ---: |
| 1 x 6,144 | 15.1 TF/s (41 ms) | 11.5 (54 ms) |
| 4 x 6,144 (dev guidance batch) | 15.3 (161 ms) | 11.9 (209 ms) |
| 1 x 24,576 (stage 2) | 15.3 (645 ms) | 11.7 (846 ms) |

So MLX attention already runs at ~73% of peak in fp16; a flash kernel's own
headroom is ~1.2-1.3x (to the ~19 TF/s the GEMM reaches), and the bf16 ->
fp16 switch alone is worth ~1.3x on both GEMM and attention. The larger
unknowns are glue overhead (per-call RoPE, gating, masks, per-token AdaLN,
eval guards, allocation churn) and the VAE; the baseline profile decides
the ranking, not these microbenchmarks.

Plus conv VAE decode (M3 Max: 6 s untiled at 768x512x121; scales ~4x at
1536x1024), Gemma-4 12B encode (once per prompt, small), audio VAE + vocoder.
GEMM is already at ~90% of peak in MLX, so the headroom is in attention
(MLX SDPA is not IO-aware), conv3d (MLX decomposes to per-frame 2D Winograd;
Draw Things' implicit GEMM is 2.4x on M1-M4), fused epilogues (AdaLN
modulate, gated attention, RMSNorm+split-RoPE, GELU), and MPS/MLX glue
(transient allocations, the retrospective's lesson). Caching (TeaCache in
the baseline; FBCache/MagCache unported) and sparse attention multiply on
top but change outputs; they are a separate, opt-in tier.

## 3. Campaign phases

1. **Baseline** (blocked on downloads): time all three pipelines in
   ltx-2-mlx at the canonical clip (1536x1024x121, default DFR size) and a
   short dev clip (768x512x121 output), q8 and bf16 packs, record peak
   memory, per-op profile of one stage-1 forward and one VAE decode.
2. **Precision gate** (section 4). Output: the dtype policy per op class.
3. **Engine bring-up**: our own loader for the official safetensors,
   DiT forward on QuixiCore-Metal kernels (attn_fwd/cross_attn/rotary/
   gemm_* exist; conv3d does not), conv VAE decoder, Gemma-4 encode,
   distilled pipeline end-to-end, bit-compared against the runner.
4. **Kernel waves**, ranked by measured per-op share: flash attention
   (D128, fp16, fp32 stats, gated epilogue), conv3d implicit GEMM, fused
   AdaLN/GELU/RMSNorm-RoPE, runtime low-rank LoRA GEMM, tiled VAE.
5. **Dev + DFR pipelines** on the same engine; dev-vs-DFR quality A/B.
6. **Profiles**: three ids on platform `metal-m1ultra`; M5 variants later.

## 4. Precision gate (the "quick experiment")

Question: does fp16 operand compute change the output versus bf16?
Runs, same seeds/prompts/sampler, distilled pipeline at 768x512x49 (cheap):

1. bf16 reference (runner as-is; MLX bf16 is emulated but exact) with
   per-block max|.| of residual stream, attention out, FFN intermediate.
2. fp16 everywhere (weights cast at load, inputs cast fp16).
3. fp16 operands + fp32 residual/glue (if run 2 shows range trouble).

Metrics: latent cosine similarity and PSNR (stage 1 and final), per-block
max activation, NaN/inf count, decoded video side-by-side. Pass: cos-sim
~0.9999 (community int8 pack = 0.9982), no overflow. Also a one-time scan
of all bf16 weights for |w| > 65504 or subnormal-range mass. Any block that
overflows keeps fp32 operands rather than abandoning fp16 globally.

## 5. Open items

- Verify native bf16 on M3+ GPUs on hardware (affects whether fp16 cast
  stays the policy there).
- DFR forward count and memory at default size: measure in baseline.
- Diffusion VAE decoder (NATTEN neighborhood attention) vs conv decoder
  quality: conv is the profile default; revisit after kernels.

## 6. Baseline measurements (dgrauet/ltx-2-mlx v0.15.12, q8 pack, M1 Ultra, seed 42, 2026-10-01)

Prompt: snowy pine forest / red fox. Wall = `/usr/bin/time` real, includes
Gemma load+encode (~7-11 s), DiT load (~2-3 s), decode+mux.

| clip | pipeline | DiT forwards | per-forward | wall | peak RSS |
| --- | --- | --- | --- | ---: | ---: |
| 768x512x121 | distilled | 8 @ 1,536 tok + 3 @ 6,144 | 4.3 s / 17.8 s | **116.9 s** | 22.5 GB |
| 768x512x121 | dev two-stage (30 steps, CFG+STG+modality = 4 passes) | 120 @ 1,536 (batched 4 = 17.2 s/step) + 3 @ 6,144 | 4.3 s / 18.7 s | **606.9 s** | 22.5 GB |
| 768x512x121 | DFR (5 keyframe slots, detailing LoRA 0.5) | 8 @ 2,016 + 3 @ 9,600 | 5.5 s / 31.2 s | **170.9 s** | 22.8 GB |
| 768x512x121 | distilled, **bf16 pack** | 8 @ 1,536 + 3 @ 6,144 | 8.4 s / 32.3 s | **202.4 s** | 39.8 GB |
| 1536x1024x121 | distilled, untiled | 8 @ 6,144 + 3 @ 24,576 | 17.8 s / 111.1 s | DiT 475 s; untiled decode (est. 56 GB) killed the process | |
| 1536x1024x121 | distilled, `--tile-spatial 2` (the runner's recommended HD config; tiles attention, output differs) | 8 @ 6,144 + 3 @ 24,576 | 20.7 s / 80.8 s | **504.9 s** (decode 83.8 s, 53.2 GB peak Metal) | 22.4 GB |

Conv VAE decode 768x512x121: 18.1 s, 33.9 GB peak Metal memory (untiled).
Dev full-size projected: 30 x (4 x 17.8) + 3 x 111 = ~41 min DiT + decode.

Effective DiT throughput: 0.046 PFLOP / 4.3 s = 10.7 TF/s (1.5k tok);
0.21 / 17.8 = 11.8 TF/s (6k); 1.2 / 111 = 10.8 TF/s (24k). That is ~52-57%
of the 20.8 TF/s peak and ~70% of what MLX's own kernels reach in isolation
(int8 GEMM 15.8, bf16 SDPA 11.5). The q8 pack's compute is a bf16-SDPA /
int8-GEMM mix; an fp16-operand engine at the measured kernel rates (GEMM
~19, SDPA ~15.3) is ~1.5x on the DiT before any custom kernel, and the
remaining gap to peak is another ~1.3x. Baseline mp4s and logs:
`~/.local/scratch/ltx25/baseline/`.

## 7. Precision gate results (2026-10-01, bf16 pack, `gate/forward_gate.py`, `gate/precision_gate.py`)

Per-forward, identical captured stage-2 inputs (6,144 video tokens, real
pipeline state), DiT weights + operands cast per run, fp32 as truth:

| DiT dtype | video cos vs fp32 | rel-L2 vs fp32 | audio cos vs fp32 |
| --- | ---: | ---: | ---: |
| bf16 (as shipped / as the runner computes) | 0.999049 | 0.0437 | 0.999715 |
| **fp16** | **0.999989** | **0.0047** | 0.999999 |

fp16 operands are ~10x closer to fp32 than bf16 operands (3 more mantissa
bits; range was never the issue: residual-stream max over all forwards was
16.4k against the 65,504 fp16 limit, no NaN/inf, output absmax identical).
End-to-end (768x512x49 distilled, same seed): bf16 vs fp16 final latents
cos 0.950 / 29.8 dB, audio 0.9997 -- that spread is trajectory divergence
through 8 ancestral steps, not a per-forward error (frames are the same
scene, same fox, sub-pixel differences; `gate/out/sidebyside24.png`).
**Policy confirmed: fp16 MMA operands, fp32 accumulate/glue, on M1-M4.**
Per-forward speed by dtype: first attempt invalid (held an fp32 copy of the
DiT next to the original, 5 GB swap, bf16 "340 s/forward"); clean rerun in
section 8.

## 8. Where the baseline's forward goes (2026-10-01, bf16 pack, 6,144-token stage-2 forward)

Clean per-forward time is 32.1 s and is **identical for bf16, fp16 and fp32
weights** (`gate/forward_speed.py`, one process each, no swap). Precise
per-op profile (`gate/profile_hook2.py`, inputs evaluated before timing):

| op | sec | share |
| --- | ---: | ---: |
| FFN proj_out linear (K=16384) x48 | 13.8 | 40% |
| attention projections (4096x4096) x288 | 6.2 | 18% |
| FFN proj_in linear x48 | 5.3 | 15% |
| self-attention sdpa (32 x 6144 x 128) x48 | 3.4 | 10% |
| all other linears, text/AV cross sdpa, norms, rope, gelu | 3.7 | 11% |
| untracked glue (AdaLN modulate, residual, gating, reshapes) | 1.0 | 3% |

Linears take 2.5-3.4x longer than the same GEMMs in isolation. Root cause
(`gate/dtype_probe.py`): **every DiT linear and sdpa receives fp32
activations.** The AdaLN `scale_shift_table` and the per-token AdaLN
embedding are fp32; `rms_norm(x_bf16) * (1 + scale_fp32) + shift_fp32`
promotes the stream to fp32, and it never comes back (nn.Linear with
fp32 x and bf16 w runs an fp32 GEMM). Measured: proj_out GEMM with fp32 x
= 286 ms vs 83 ms with bf16/fp16 x (3.4x). So the "best Mac path" runs an
fp32 DiT by accident on a GPU whose fp16 MMA is 2x the fp32 rate, and the
q8 pack is faster only because quantized_matmul has its own kernel.

Consequence for section 7: the "bf16 vs fp16" per-forward delta measured
there came from the runner's **entry casts** (latent, timestep/sigma, text
embeds cast to the DiT dtype at the model boundary; bf16 ulp at timestep
~422 is 2.0, fp16 ulp is 0.25), not from MMA operand precision -- the
operands were fp32 in all three runs. Corrected gate in section 9.

## 9. Corrected precision + speed gate (2026-10-01, identical captured stage-2 inputs, 6,144 tokens, `gate/forward_gate3.py`)

Truth = fp32 weights, fp32 activations, fp32 entry. Repeat of truth = bit-identical (deterministic).

| config | s/forward | video rel-L2 vs truth | video cos | audio rel-L2 |
| --- | ---: | ---: | ---: | ---: |
| fp32 everything (truth) | 32.3 | 0 | 1 | 0 |
| fp16 weights, fp32 operands | 32.3 | 0.00000 (bf16->fp16 weight cast is exact) | 1.000000 | 0 |
| **fp16 operands (linear + sdpa), fp32 glue = our policy** | **13.7** | **0.0071** | 0.999975 | 0.0011 |
| bf16 operands, fp32 glue | 17.4 | 0.0350 | 0.999392 | 0.0054 |
| runner as-is (fp32 operands, bf16 entry casts) | 32.1 | 0.0422 | 0.999113 | 0.0254 |

Reading: on M1-M4 the fp16-operand path is 2.36x the baseline's forward
with zero custom kernels, and its deviation from fp32 is 6x smaller than
the baseline's own (the baseline's error is its bf16 entry cast of
timestep/latents, not its GEMMs). bf16 operands are 5x less accurate than
fp16 and 27% slower. **Decision confirmed: fp16 MMA operands, fp32
accumulate and glue, fp32 model boundary (timestep, sigma, latents, text
embeds). No quantization on M1-M4.** Baseline packs run bf16 weights that
convert to fp16 exactly, so the official bf16 checkpoint is the weight
source with no conversion loss.

## 10. Ranked optimization opportunities on M1 Ultra (2026-10-01 close of investigation)

fp16-operand forward, 6,144 tokens, precise profile (`LTX_OPCAST=fp16`,
16.6 s sync-inflated / 13.7 s clean; physics at measured kernel ceilings
~11.4 s, at 100% ALU ~10 s):

| op | sec | share | ceiling | note |
| --- | ---: | ---: | --- | --- |
| FFN proj_out GEMM (K=16384) x48 | 4.07 | 24.5% | ~2.0 s | MLX GEMM hits 10 TF/s at K=16384 vs 19 at K=4096: tile/split-K problem, custom kernel target #1 |
| attention projections 4096^2 x288 | 3.59 | 21.6% | ~3.3 s | at 18.7 TF/s already |
| FFN proj_in GEMM x48 | 2.25 | 13.5% | ~2.1 s | at 19 TF/s already |
| self-attn sdpa 32x6144x128 x48 | 1.96 | 11.8% | ~1.5 s | 15.1 -> ~19 TF/s with a flash kernel; grows to ~40% of FLOPs at 24k tokens |
| AV/text cross projections + sdpa | ~1.9 | 11% | ~1.7 s | small |
| norms, rope, gelu, gating, AdaLN, casts (tracked small ops + untracked) | ~2.8 | 17% | ~1.0 s | fusion target: rmsnorm+modulate+cast, gated-attention epilogue, rope-in-projection |

End-to-end, 768x512x121 distilled, same seed, this box:

| config | wall | vs bf16 baseline | vs q8 baseline |
| --- | ---: | ---: | ---: |
| baseline bf16 pack (as shipped) | 202.4 s | 1.00 | |
| baseline q8 pack (its recommended config) | 116.9 s | 1.73x | 1.00 |
| **baseline + our precision policy shimmed in (no kernels)** | **100.4 s** | **2.02x** | **1.16x** |
| projected: our engine, same policy + kernel waves + 2x VAE | ~73 s (est.) | ~2.8x | ~1.6x |

The 100.4 s splits: Gemma load+encode 8 s, DiT load 3 s, stage 1 8 x 3.5 s
= 28 s, stage 2 3 x 13.4 s = 40 s, conv VAE decode 17.6 s, misc ~4 s.

Ranked levers (M1 Ultra, output-preserving):
1. **Precision path** (fp16 operands, fp32 glue, fp32 boundary): 2.36x on
   the DiT, measured; strictly more accurate than the baseline. Table stakes
   for the engine; no kernel work.
2. **Conv3d VAE decode**: 17.6 s / 100 s at 768x512, 84 s / 505 s at
   1536x1024, 53 GB peak. Draw Things' implicit-GEMM conv3d is 2.4x on
   M1-M4 and MLX's conv3d is a per-frame Winograd decomposition. Largest
   absolute lever at HD; also the memory lever (untiled decode killed the
   baseline at HD).
3. **Large-K GEMM** (FFN proj_out): ~2 s/forward, 15% of the DiT.
4. **Flash attention** D128 fp16 with fp32 statistics + gated epilogue:
   ~0.5 s/forward at 6k, ~12% of the forward at 24k tokens.
5. **Glue fusion**: ~1-1.5 s/forward (8-10%).
6. **Fixed costs**: Gemma-4 encode 8 s/prompt (resident fp16 encoder,
   ~3 s), model load 3 s (mmap, residency set), audio path.
7. **Dev pipeline**: 4-way guidance batch already gives M=24k GEMMs; the
   same forward-level gains apply (606.9 s baseline -> ~375 s projected).

Output-changing tier (opt-in profile flags, measured separately):
TeaCache/FBCache (1.5-2x on step count), spatial tiling (baseline's
`--tile-spatial 2`: 24k-token forward 111 -> 81 s), sparse attention at
HD. M5 adds the int8 W8A8 GEMM path (1.6-1.9x on ~70% of the forward).

Honest ceiling: the M1 Ultra DiT at fp16 is within ~1.2x of its measured
kernel ceiling once the precision path is in. The remaining big absolute
wins are the VAE, HD attention, and the opt-in tiers.

## 11. Reference: QuixiAI/h3.c (antirez MiniMax-H3 engine, Eric's fork; clone at `~/.local/scratch/ltx25/ref/h3.c`)

Metal side is mostly MPSGraph: DiT attention = MPSGraph SDPA (no flash
kernel), conv3d/VAEs = MPSGraph fp32, M1-M4 GEMMs = MPSGraph matmul. Own
kernels: fused gate+residual+RMSNorm+AdaLN (per-token modulation table via
row_map), rotate-half RoPE in the QKV epilogue, GGUF span decoders +
simdgroup-bf16 GEMM (~5x slower than bf16 on M5), M5-only TensorOps bf16 and
int8 GEMMs (per-row activation scales, per-channel weight scales, grouped-K
FC2). bf16 everywhere with fp32 reductions/accumulators; no fp16 path;
validated on M3 Max / M5 Max only. perf/ docs in the fork are the CUDA
campaign. Transfers: AdaLN fusion pattern + modulation tables, M5 int8 GEMM
structure, command-buffer overlap, arena aliasing, mmap no-copy weights,
two-slot SSD streamer, bench/test harness discipline. Does not transfer:
attention, VAE (transformer decoder there, causal conv3d here), SwiGLU
epilogue. Eric's fork commits: GGUF in-place decoding, studio UI, CUDA
backend, perf docs (2026-08-12..25).

## 12. Architecture decision (2026-10-02)

Subsystem of SlimServe (Sean's call), brought up in SlimServe first, then
kernels upstreamed to QuixiCore-Metal as a separate PR (the standing
flow). Host recommended: Python orchestration on MLX with custom kernels in
QuixiCore-Metal via its MLX binding (MLX GEMM at ~90% / SDPA at ~73% of
peak measured here; the 2.36x precision lever needs no kernels; vllm-mlx
env already exists). Alternatives considered: torch-MPS host (slow linear
path, residency/alloc-churn pain from the LLM retrospective), C/Metal
engine a la h3.c (months of loader/encoder/VAE/mux work before first frame;
h3 still ended on MPSGraph for attention and conv). Awaiting Sean's go.

## 13. Engine bring-up, N0-N2 (2026-10-02, `slimserve/video/ltx25/`)

All runs one process at a time through `perf/ltx25_harness/gpu_run.py`
(section 14). Raw logs: `perf/results/2026-10-02-ltx25-n2/`.

**N0 loader.** Official safetensors read directly (`checkpoints.py`), bf16 ->
fp16 in 4 GiB chunks with the source dropped as it goes. Distilled DiT: 4,091
tensors + 258 connector tensors, 35.4 GiB fp16 + 0.02 GiB fp32 tables, 5.8 s,
39.1 GiB peak. Key map: upstream names with `model.diffusion_model.` stripped;
the runner's pack renames `to_out.0 -> to_out`, `ff.net.0.proj -> proj_in`,
`ff.net.2 -> proj_out`, `linear_1/2 -> linear1/2` and transposes convs to
channels-last, which the engine does at load instead.

**N1 forward** (`n1_forward_gate.py`, captured stage-2 inputs, 6,144 tokens):

| engine | s/forward | video rel-L2 vs fp32 | audio rel-L2 | vs the fp16 shim |
| --- | ---: | ---: | ---: | ---: |
| runner + fp16-operand shim (section 9) | 13.7 | 0.0071 | 0.0011 | 0 |
| ours, policy as shimmed | 13.71 | 0.00709 | 0.00107 | 0.0025 |
| ours + split-K FFN GEMM | **11.92** | 0.00692 | 0.00106 | 0.0025 |

Deterministic across repeats. The 0.0025 distance to the shim is two fp16
rounding realizations of the same math (keyframe embedding and AdaLN adds in
fp32 here); both sit at the same distance from the truth, so the planned
"<= 0.001 vs shim" gate was the wrong instrument and is dropped.

**Split-K** (`dit.py`, first kernel-wave item, no custom kernel needed): MLX's
fp16 GEMM is 18.4-19.3 TF/s for every DiT shape except K=16384, where it is
10.3-10.4 TF/s (6.3 at 24,576 rows). Summing eight K=2048 chunks: 79.4 -> 44.6
ms at 6,144 rows, 526 -> 172 ms at 24,576 (19.2 TF/s). Same error vs fp32
(2.07e-4, output rounding). Worth 1.8 s per 6k forward, ~17 s per HD forward.

**N2 components** (each vs the runner on identical inputs):

| component | dtype | parity | time, memory |
| --- | --- | --- | --- |
| Gemma-4 12B + projection + connectors (`text.py`) | fp16 operands, fp32 stream | rel-L2 0.0003-0.0007 vs fp32; runner as shipped 0.005-0.010; no overflow (max GEMM output 8,232) | load 4.8 s, encode 0.85 s warm, 28.1 GiB |
| video VAE decode (`vae.py`) | fp32 | rel-L2 2.0e-6 vs runner fp32, PSNR 89.7 dB; runner as shipped (bf16) is 47.3 dB from fp32; fp16 is 59.0 dB | 768x512x121: 9.6 s, 14.4 GiB |
| VAE encode | fp32 (fp16 overflows) | 4.0e-6 | 49 f: 5.7 s |
| latent upscalers (`upscaler.py`) | fp32 (fp16 overflows) | 2.9e-6 / 5.5e-6 | spatial 2.9 s cold |
| audio VAE + vocoder + BWE (`audio.py`) | fp32 | mel 129.7 dB; waveform 64.5 dB vs the runner with its three deviations from upstream patched in, 28.0 dB as-is (the engine follows upstream: no tanh on the 16 kHz stage, no +1e-9 under the STFT sqrt, upstream edge handling) | 5 s clip: 2.6 s |
| mux (`mux.py`) | | H.264 yuv420p CRF 18 + AAC 48 kHz stereo, raw rgb24 piped | 0.7 s |

Text tower runs on the real tokens only (padding is masked and replaced by
registers either way), which is why encode is 0.85 s.

**VAE memory** (`vae.py: _conv_slabs`): MLX conv3d scratch scales with the
whole input; the 121-frame decode peaked near 60 GiB for activations of 1.5
GiB each. Each conv now runs in temporal slabs (exact: output frame t reads
padded frames t..t+2). Peak 60 -> 14.4 GiB, 12.0 -> 9.6 s, output unchanged.
Conv profile (`conv3d_layers_121_fp32.md`): 117.5 TFLOP nominal; MLX's
Winograd path runs steady-state at an effective 18-25 TF/s in fp32, so a
direct implicit-GEMM fp16 kernel (19 TF/s nominal) is not a speed lever
here; the levers are fp16 operands with fp32 accumulate (2x, needs a kernel
to keep fp32 accuracy), first-touch allocation stalls (0.8-1.2 s on the
first conv of each new shape), and the remaining 9 full-size activations.

**N2 end to end** (`n2_e2e.py`, 768x512x121 distilled, seed 42, fox prompt,
cold process, wall includes imports and model loads):

| engine | wall | text | load | stage 1 (8) | upscale | stage 2 (3) | decode | audio | mux | peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline q8 | 116.9 s | | | 4.3 s/step | | 17.8 s/step | 18.1 | | | 33.9 GiB Metal |
| baseline bf16 | 202.4 s | | | 8.4 | | 32.3 | | | | 39.8 GiB RSS |
| baseline + precision shim | 100.4 s | 8 | 3 | 3.5 | | 13.4 | 17.6 | | | |
| **ours** | **90.4 s** | 5.7 | 4.8 | 24.7 (3.07/step) | 2.6 | 36.2 (12.06/step) | 12.7 | 2.65 | 0.7 | 52.8 GiB |

1.29x the q8 baseline, 2.24x the bf16 baseline, before any custom kernel.
Same scene as the baseline at frames 0/60/120 (`side.png`); stage-1 audio
latent cos 0.99976 vs the runner after 8 ancestral steps. mp4: 121 frames,
24 fps, H.264 + AAC 5.01 s. Gemma is unloaded after encoding in the one-shot
path (resident serving keeps it: 28 + 40 GiB).

## 14. Incident 2026-10-02: kernel panic from concurrent GPU jobs

Four model-loading processes at once (84.7 + 64.2 + 18.1 GiB resident plus
a 39 GiB DiT loading, 128 GiB box, `iogpu.wired_limit_mb=122880`) exhausted
the compressor; watchdogd starved for 90 s; forced power-off. Rules now in
HANDOFF.md: one model-loading process at a time through `gpu_run.py`
(exclusive lock, headroom check, 16 GiB reserve, refuses a wired limit above
110000), agents never touch the GPU, the engine sets an MLX memory limit of
RAM - 24 GiB and plans VAE decode inside what is left.

## 15. Kernel wave, dev, DFR, HD and serving (2026-10-02, after the restart)

**What the kernels can still give on M1 Ultra.** Per-op profile of the
engine forward at 6,144 tokens (`n3_forward_profile.py`,
`forward_profile_6k.md`): GEMMs run at 16.5-18.3 TF/s, self-attention at
15.1, glue is 10% sync-inflated. Clean forward 11.5 s against 10.1 s at the
20.8 TF/s peak: 84-88% of the hardware. Decisions, each measured:

| item | result | decision |
| --- | --- | --- |
| split-K FFN GEMM | 13.71 -> 11.92 s, error unchanged | kept |
| compiled glue (modulate+cast, gated residual, GELU, head gate) | 11.92 -> 11.52 s, bit-identical | kept |
| compiled RoPE; eval cadence 1/2/4/16/48 | 11.46-11.60 s, noise | not adopted |
| query-chunked attention | MLX's SDPA is already flash-style: 1.1 GiB peak and 15.4 TF/s at 24,576 tokens, chunking only costs | not adopted |
| row- or N-chunked GEMM at 24k rows | 18.4-19.2 TF/s either way; only K=2048 inputs at 24k rows gain (14.5 -> 18.0) | not adopted (audio-side, <1%) |
| fp16 VAE decode (operands, or operands + stream) | 8.7 -> 5.5-6.2 s but 55.1 dB from fp32, max error 14/255: MLX's fp16 Winograd conv loses accuracy in the conv itself, an fp32 stream does not recover it | opt-in only (`VideoVAE(dtype=mx.float16)`), fp32 default |
| custom flash attention / GEMM kernels | ceiling is 15.4 -> ~19 TF/s on 14% (6k) to 40% (24k) of the forward, and ~17.5 -> ~19 on GEMMs: at most ~10% of an HD forward. QuixiCore-Metal's own `gemm_v3` reached 94-99% of MPS without beating it and `attn_fwd` D128 measured 1.1x on small shapes | not started; ranked below the product milestones, see "Next" |
| conv3d implicit GEMM | MLX's fp32 conv3d is a Winograd path at an effective 18-25 TF/s steady state, above what a direct fp32 GEMM conv can reach (6 TF/s) and equal to a direct fp16 one; M1 has no fp16-operand/fp32-accumulate MMA | not a speed lever on M1-M4; the VAE win was memory and stalls (slabs) |

Honest reading: on M1-M4 the engine is at the MLX kernel ceiling and MLX's
kernels are within 10-25% of the chip. The measured 1.3-1.5x over the q8
baseline (2.2x over bf16) came from the precision path, split-K, glue
fusion, batching and the VAE restructure, not from hand-written Metal. The
remaining multipliers are the opt-in tier and M5's int8 MMA.

**HD forward** (`n3_forward_hd.py`, 24,576 tokens, synthetic latents, real
text): 69.7 s before glue fusion, 67.3 s in the end-to-end run, flat
39.5 GiB. Baseline: 111.1 s untiled, 80.8 s with tiled attention.

**Dev** (`pipeline.dev`): one guided step on identical inputs
(`n5_guided_step.py`) vs the baseline: cond 0.0050, negative 0.0064, STG
0.0050, modality 0.0050 video rel-L2; guided 0.0072 batched, 0.0074
sequential (the baseline's STG pass differs from its cond by 0.51, so the
skip path is exercised). Runtime distilled LoRA (`n5_lora_check.py`): dev +
LoRA vs the distilled checkpoint 0.0093 (dev alone 0.29); runtime vs fused
fp32 2.9e-4 per linear; 11.51 -> 14.06 s per forward; detach restores the
base bit for bit. 768x512x121, 30 steps: 406 s cold (stage 1 330 s at
11.0 s per 4-pass step, stage 2 42 s). The baseline's 606.9 s figure is
its 30-step run; its saved mp4 was a 2-step run and is not a visual
reference.

**DFR** (`pipeline.dfr`, default configuration): canvas and slot layout
identical to the baseline for 9..241 frames; 768x512x121: 125 s cold
(stage 1 31.3 s at 2,016 tokens, stage 2 61.1 s at 9,600), same scene as
the baseline's DFR clip (`dfr_side.png`); baseline 170.9 s.

**VAE** after the per-slab rewrite: decode 768x512x121 8.0-8.7 s at
10.1 GiB; 1536x1024x121 untiled 33.7 s at 27.3 GiB (baseline: 83.8 s tiled,
53 GiB; untiled killed it); encode 49 frames 3.4 s at 8.1 GiB. Planner:
5 GiB + 135 B per output pixel-frame.

**HD end to end**, 1536x1024x121 distilled, fully untiled: 358 s cold,
65.7 GiB peak (stage 1 92.4 s, stage 2 202.1 s, decode 40.4 s). Baseline
504.9 s with tiled attention and decode.

**Serving** (`n6_serve_check.py`, HTTP, weights resident, 768x512x49 clips):

| pipeline | ready | concurrent requests | per clip | resident | peak | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| distilled (MLX 0.31.2) | 17.2 s | 3 + 1 | 40.7 s | 68.6 GiB flat | 71.4 GiB | PASS |
| dfr | 16.6 s | 2 + 1 | 51.9 s | 68.9 GiB flat | 74.1 GiB | PASS |
| dev | 19.4 s | 2 + 1 | 194-203 s | 76.9 GiB flat | 82.1 GiB | PASS |

Requests run strictly one at a time (queued count visible in /health), an
oversize request is refused with a 400, every mp4 probes as H.264 + AAC with
the right frame count. `slimserve ltx25-distilled -p ... --size 768x512
--seconds 5`: 86.1 s through the real CLI.

**Hand-written GEMM experiment** (`n4_gemm_kernel_experiment.py`,
`mx.fast.metal_kernel`, fp16 `simdgroup_half8x8`, one simdgroup per output
tile, operands read straight from the arrays; M=6144, K=4096, N=4096):

| kernel | TF/s | note |
| --- | ---: | --- |
| MLX matmul | 18.7 | rel error vs fp32 2.1e-4 |
| tile 32x32, accumulators in arrays | 0.3 | the compiler keeps them in memory |
| tile 8x8 / 16x16 / 32x32 / 32x64 / 64x64, unrolled into registers | 2.6 / 5.0 / **9.2** / 8.7 / 7.6 | rel error 9.4e-3: plain fp16 accumulation over K=4096 |

Hypothesis: a simple tiled simdgroup kernel can match MLX. Result: half its
rate and 45x its error; MLX accumulates more carefully than fp16 tiles.
Closing the gap needs operand staging, split-K accumulation and tile tuning
that MLX already has. Decision: rejected under the pre-registered bar (drop
below 16 TF/s at the first prototype); no custom GEMM or attention kernel on
M1-M4. This is the measured basis for "at the MLX kernel ceiling".

**Next, in order.** (1) I2V conditioning (encoder already ported). (2) DFR
temporal rounds and spatial epilogue. (3) Opt-in tier with its own A/B:
step caching, fp16 VAE, HD sparse attention. (4) Kernel work only where MLX leaves room: revisit on M5 (int8 MMA) and if
a profile shows an MLX op far below 18 TF/s, as K=16384 was. (5) QuixiCore-Metal
PR: there is no new Metal kernel to upstream yet; the candidates are the
split-K GEMM dispatch and the slab conv3d driver. (6) M5 int8 path on M5
hardware.

## 16. Second optimization round and the remaining-headroom accounting (2026-10-02)

Asked for: every potential win estimated, and the campaign to run until the
rest is within ~1%. Measured on the warm resident short clip (77.3 s before
this round), which is what a server runs.

| lever | measured | kept |
| --- | --- | --- |
| dispatch bubbles: `async_eval` between block groups and sampler steps | forward 11.53 -> 11.44 s, bit-identical | yes |
| fused QKV(+gate) projection GEMMs at 1,536 rows | 8.53 vs 8.51 ms: MLX already pipelines back-to-back GEMMs at 18 TF/s; the low per-op rates in the sync profiler were inflation | no |
| spatial upscaler: 1024-channel conv3d at a 16x8x12 grid ran at 1.4-1.7 TF/s in MLX | per-tap split-K GEMM (K = 3 x 1024): 51 -> 11 ms per conv, rel 3e-6; upscaler 2.5 -> 0.45 s (HD 6.6 -> 1.4 s) | yes |
| VAE decoder convs of the same shape class (`conv3d_core`, choice measured per shape and persisted in `~/.cache/slimserve/ltx25_conv3d_choice.json`) | decode 8.0 -> 7.1 s, HD 33.7 -> 29.9 s; same parity (2.7e-6 vs the runner in fp32) | yes |
| vocoder anti-aliasing filters: 12-tap single-channel convs, 2.5-6 ms per call x 200, dispatch-bound | compiled shifted multiply-add FIRs: 0.9 ms per call, 1e-6 agreement; audio decode 2.5 -> 0.36 s | yes |
| MLX cache limit vs decode first-touch | 12 GiB: 8.5 s, 16: 7.95, 20: 7.7, 24: 7.15; 16 GiB chosen so the dev profile still decodes without evicting the text encoder | yes |
| stage-2 GEMMs "at 17 TF/s inside the forward" | sum of standalone kernel times at measured rates (GEMM 0.175 PF at 18.8 + attention 0.034 PF at 15.1) = 11.55 s vs 11.44 s measured: nothing left between kernels | closed, 0% |

After the round: warm short clip **71.2 s** (text 1.4, stage 1 23.1, upscale
0.45, stage 2 34.5, decode 9.9, audio 1.15, mux 0.7); cold HD distilled
**343 s** (was 358). Stage-1 profile: `stage1_profile.md`.

**What is left, output-preserving, M1 Ultra** (estimates against the 71.2 s
warm short clip and the 343 s HD clip):

| lever | short | HD | status |
| --- | ---: | ---: | --- |
| attention kernel 15.3 -> ~19 TF/s (D128 fp16) | ~1.5% | ~5% | days of kernel work, parity with MLX's steel kernel not assured; the one real kernel target left |
| GEMM kernel above MLX's 18.5-19.3 | <=5% | <=5% | section 15: our prototype reached half of MLX; rejected |
| decode: remaining first-touch under the 16 GiB cache | 1% | <1% | closed by the cache choice |
| cold loads overlapped with prompt encoding (CLI one-shot only) | 2% cold | <1% | not done; serving is warm |
| dispatch/glue | <0.5% | <0.5% | closed |

Everything beyond this changes the output and is opt-in: fp16 VAE decode
(55 dB, 3-4%), step caching on dev (1.5-2x), sparse attention at HD, lower
step counts. The campaign's output-preserving work on M1 Ultra is at its
epsilon except for the attention kernel, which is recorded as the open
item with its expected value.

## 17. The measured MMA ceiling, and the exact dev-guidance restructure (2026-10-02)

**Ceiling** (`n4_mma_peak.py`: fp16 `simdgroup_multiply_accumulate` with all
operands in registers, no memory traffic, 32,768 simdgroups):

| | TF/s |
| --- | ---: |
| fp16 MMA, register-resident | **19.4** |
| fp32 MMA, register-resident | 17.5 |
| MLX fp16 GEMM (section 13) | 18.5-19.3 |
| MLX SDPA D128 (section 13) | 15.1-15.4 |

So the spec's 20.8 is not reachable; MLX's GEMM is at 95-99% of the real
ceiling and attention at 78%. That is the quantitative basis of "custom
kernels buy a few percent": GEMMs are 77% of the transformer's FLOPs with
1-5% headroom, attention 14% (6k tokens) to 40% (24k) with 21% headroom,
giving at most ~3% (short) to ~8% (HD) of a forward for a perfect attention
kernel. New fact: fp32 MMA is nearly as fast as fp16 on M1 Ultra, so a
custom kernel can accumulate in fp32 at no MMA cost; the fp32 GEMM slowness
the baseline suffered is operand bandwidth, not the MMA.

**Exact dev restructure** (`dit.py: text_rows, share_from`;
`pipeline.GuidedDenoiser`): the STG pass is the conditional pass until the
first STG block (28 of 48), so it is forked from the conditional hidden state
there instead of recomputed; and the three conditional-text passes project
the text K/V once per distinct text row. Per guided step at 1,536 tokens:
10.86 -> 10.41 (shared text) -> **9.13 s** (fork), rel-L2 2.8e-3 to the
unshared result (GEMM row-count kernel selection; fp16 level), and the
guided step still matches the baseline at 0.0072 (`n5_guided_step.py`).
Dev 768x512x121 end to end: 406 -> **343 s** cold; frame 60 identical to
the previous clip by eye (`dev_after/side.png`).

## 18. Recommended workflow audit, the diffusion VAE decoder, and two MLX defects (2026-10-02)

**Audit against Lightricks' recommendations** (model card, LTX-2 repo
`utils/constants.py` and pipelines, the prompting guides; URLs in the
research file). Matching already: stage-1 768x512 -> 1536x1024, 121 frames,
24 fps, the sigma tables, CFG 3/7, STG 1, modality 3, rescale 0.7, the DFR
canvas and detailing strength 0.5, prompt enhancement off by default.
Deviations found and fixed: (1) STG block: upstream LTX-2.5 uses 29 (0-based);
the Mac baseline and therefore we used 2.3's 28; now 29. (2) Decoder:
upstream's default is the diffusion VAE decoder ("sharper faces, textures and
on-screen text"); the baseline and we used the conv decoder. Ported, now the
default. (3) Prompting: one present-tense paragraph under ~200 words, sounds
tied to visible sources, named cuts with re-established shots, no signage;
the demo prompt was rewritten to it. Note that "matches the baseline" was never
"matches Lightricks": the baseline carries at least the STG deviation.

**Diffusion decoder** (`diffvae.py`): four deterministic neighborhood-attention
stages (3x7x7, 3x7x7, 3x5x5, 3x5x5; 2048/1024/512/512 channels) with
pixel-shuffle upsamples, then 8 diffusion blocks at pixel/4 (256 channels,
11x11x11 windows, AdaLN from t, stage-4 feature as context, upstream's 4 haloed
W slabs) running one x0 step from pure noise. Parity vs the baseline's port in
fp32 (`n7_diffvae_parity.py`): every tapped stage rel-L2 4e-7 to 3e-6, pixels
7e-7 / 93.0 dB. Precision: fp16 operands + fp32 stream 65.4 dB, fp16 operands
+ fp16 stream **64.5 dB, max 1 level** (default; upstream runs it all in bf16,
which is coarser).

Neighborhood attention is a Metal kernel (`na3d`, `mx.fast.metal_kernel`):
one (query, head) per 8 lanes, each lane 8 dims, fp32 online softmax, exact
NATTEN clamping. vs the reference formulation: fp32 1e-7, fp16 6e-5.

| kernel variant (stage-5 slab 121x128x58, 4 heads, 11^3) | TF/s |
| --- | ---: |
| one thread per (query, head) | 0.6 |
| **8 lanes per query, shuffle reduce** | **3.0** |
| 4 / 2 / 1 lanes | 2.5 / 1.4 / 0.5 |
| threadgroup-staged key rows (32 queries) | 2.7 |
| row-batched scores before the softmax update | 1.6 (score array spills) |

Latency-bound rather than memory-bound; the simple 8-lane kernel stays. The
baseline's block-gather SDPA formulation is ~8x slower end to end.

| decode | baseline conv | ours conv | baseline diffusion | ours diffusion |
| --- | ---: | ---: | ---: | ---: |
| 768x512x49 | 6.3 s | 3.9 s | 69.3 s (bf16) / 40.1 s (fp32) | **9.0 s**, 7.5 GiB |
| 768x512x121 | 18.1 s | 7.1 s | | **30 s**, 16 GiB |
| 1536x1024x121 | 83.8 s tiled | 29.9 s | | **157 s**, 35 GiB |
| 1216x640x241 (10 s) | | 47 s | | 142 s |

Memory discipline that got there: the MLP's 4x hidden state in the operand
dtype and in temporal chunks (block peak 29.9 -> 11.1 GiB at 49 frames);
stage 5 keeps its residual stream as resident temporal chunks (HD peak 76 ->
52 GiB), fp16 stream (-> 35 GiB); stage-4/5 attention in exact token-budgeted
temporal chunks (full halo, slabs never shorter than the window).

**Two MLX 0.32.2 defects, found on the 10 s clip** (both reproduced and
worked around; `n7_metal_kernel_race.py`, `n7_mlx_2e31_split.py`):

1. A `metal_kernel` launched behind in-flight MLX ops read incomplete inputs
   (4/5 runs NaN on the stage-4 volume); `mx.eval` on the inputs did not
   prevent it, `mx.synchronize()` before the launch did (0/5); a second
   synchronize after the launch was needed for the full decode. Cost ~1 ms
   per call. With Metal API validation enabled the race never appears.
2. `mx.split` (and slicing) of an array past 2^31 elements returns wrong data
   for the tail; the GEMM and rms_norm producing it are correct. Hit by the
   fused QKV projection of stage 4 (2.4 G elements) and the stage-5 context
   (3.0 G). Fix: three projections, context per chunk, and `_lin` refuses
   outputs past 2^31 so the next case fails loudly.

Validated: the 10 s 1216x640 clip's decode is deterministic across runs and
within 36-38 dB of the conv decoder on every frame (no frame under 20 dB;
before the fixes the last 12 frames were garbage).

## 19. Demo renders at the recommended settings (2026-10-02, for the PR, pending review)

Prompt `~/.local/scratch/ltx25/demo/prompt_v3.txt` (cathedral, blue service
door, two cat-headed alchemists, pinball arcade; three named cuts), 1216x640,
241 frames (10 s; the largest 10 s size inside the 24,576-token envelope),
seed 7, diffusion decoder, cold process through `slimserve <id> -p`:

| mode | wall | stage 1 | stage 2 | decode | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| distilled (prompt v2) | 475 s | 92.6 | 195.1 | 167.2 | sharp; alchemists came out as men with cat ears |
| DFR | 730 s | 125.7 | 426.0 (3 x 142 s at ~37k tokens) | 158.0 | best adherence: cat faces, floor mural; identity bleed (her hair on the alchemists, a robe on her in the arcade) |
| dev, 30 steps | 1495 s | 1116.2 | 219.3 | 137.6 | cat faces; rendered the last cut as 0.75 s of black and framed the arcade shot like a screen |

Decoder output verified per frame against the conv decoder (36-38 dB
everywhere; the pre-fix tail was 15 dB). The remaining flaws are generation
and prompt behaviour, not engine faults; the prompt's identity bleed is the
next thing to iterate (contrasting traits for the alchemists, re-describe
her at each cut).

## 20. Audit against Lightricks' source: the missing BOS token and six more deviations (2026-10-05)

The three demo renders of section 19 looked unlike the model's published
output (people not lifelike), which is the signature of a conditioning or
sampler mismatch rather than a kernel fault. Section 18's audit was against
the model card and constants only; every numeric parity gate in this ledger
was against the dgrauet Mac port, so anything the port gets wrong we matched
exactly. This pass compared `text.py`, `sampling.py`, `pipeline.py`, `dit.py`,
`vae.py` and `checkpoints.py` line by line against Lightricks/LTX-2 HEAD
9ec55f9 (`~/.local/scratch/ltx25/upstream`, `ltx-core` + `ltx-pipelines`)
with the port as the third column. Three read-only passes (text, sampling and
guidance, transformer and VAE); no renders until the comparison was complete.

| # | item | upstream | ours (and the port) | reach |
| --- | --- | --- | --- | --- |
| 1 | `<bos>` | `LTXGemmaTokenizer` prepends id 2 explicitly ("Gemma 4 does not [emit BOS], so we prepend"); head truncation to 1024 | raw `tokenizers` encode, the bundled post-processor adds nothing: **no BOS, ever**; tail truncation. Same in the port | every mode, every prompt |
| 2 | stage-2 sampler | ancestral Euler on 2.5 checkpoints (`ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)`, noise seed + 20000) | plain Euler; the port quotes an upstream docstring that no longer exists | distilled, DFR |
| 3 | STG block | `_PARAMS_SINCE_VERSION` gives a 2.5 checkpoint `LTX_2_4_PARAMS`: `stg_blocks=[28]`; `[29]` is the 2.0 dataclass default | 29 (section 18's change; the port had 28) | dev |
| 4 | dev sigma shift | `scheduler.execute(steps)` with no latent: shift fixed at the 4096-token anchor, 2.05 | token-count dependent (2.78 at 1024x1536). Same in the port | dev |
| 5 | dev stage-2 audio | `freeze_audio=True`: clean stage-1 audio, sigma 0 for its tokens, prompt AdaLN and cross gates | re-noised at 0.909 and co-denoised, result discarded | dev |
| 6 | negative prompt | begins `has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, …` | those five tags missing. Same in the port | dev |
| 7 | connector q/k RMSNorm eps | 1e-6 | 1e-5 (the port's `nn.RMSNorm` default) | all, negligible |

Empirical check of (1), no model load: encoding "A young woman walks through
a cathedral." with the checkpoint's tokenizer gives `[236776, 3184, …]`;
`<bos>` is id 2 and the post-processor is `TemplateProcessing` with
`special_tokens: {}`. Gemma's hidden states depend on BOS at every position
(it is the attention sink that sets the scale of everything after it) and the
188160 -> 4096 aggregate projection plus the 8-layer connectors were trained on
BOS-prefixed states, so the DiT received off-distribution text conditioning
for every clip in this ledger. The text-path parity of section 13 (rel-L2
0.0003 against the port) could not see it.

Verified as matching upstream, by code reading with quoted lines: both sigma
tables; Euler and ancestral step formulas; x0 convention; the CFG/STG/modality
combination and the global-std rescale; the STG value-passthrough mechanism
and the all-blocks modality skip; guidance values; the batched four-row pass;
stage-2 re-noise at sigma 0.909; the T2V masks; timestep embedding and the
AdaLN tables and chunk indexing; per-token timesteps and frozen-modality
sigma; RoPE (fp64 grid, split layout, front padding, midpoint positions,
fps scaling, temporal-only cross RoPE); qk RMSNorm placement; gated
attention; FFN; norms and the output block; keyframe abs-pos embedding;
patchify and the per-channel latent statistics; the conv decoder (no timestep
conditioning on 2.5), its block order, padding, depth-to-space, output
mapping; the diffusion decoder's one x0 step from noise at t=1; weight keys;
LoRA scaling; which Gemma hidden states are used (embeddings + 48, last one
final-normed), the per-token RMS over layers, the rescale and the two
aggregate projections, left padding and positions, the registers, the
connector architecture; DFR segment layout and LoRA strengths. There are no
sigma-schedule or LoRA-blend differences. Upstream's official default output
is 1536x1024x121 (`default_2_stage_distilled_arg_parser` sets
height/width to `stage_2_*`; stage 1 at 768x512), the same as our profile
default. An earlier draft of this section said 768x512: wrong.

Fixes (this commit): `text.py` prepends BOS and truncates from the head,
connector eps 1e-6; `sampling.py` STG block 28, the full negative prompt,
`ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET`; `pipeline.py` ancestral stage 2 for
distilled and DFR, dev schedule at the 4096-token anchor, dev stage 2 with
frozen audio. Section 18's STG claim is withdrawn. Parity against the port is
now expected to differ on the text path (by design) and on stage 2; the
reference for parity from here on is upstream, not the port.

**A/B after the fixes** (`~/.local/scratch/ltx25/demo/v4/`, distilled,
diffusion decoder, seed 7). A single-shot portrait prompt at 768x512x121
(`prompt_portrait.txt`): all fixes 107.7 s; BOS removed only 105.1 s; plain
Euler stage 2 only 105.0 s. All three are lifelike; BOS changes prompt
adherence and tone (dark-blonde hair and warm directional light with it,
dark brown hair and a flatter image without), the stage-2 sampler is a small
texture difference on this clip. The section-19 prompt v3 (three cuts, two
cat-headed alchemists) at 1216x640x241 with all fixes: 465.2 s,
`distilled_v3prompt_fixed.mp4`. Against the section-19 DFR/dev renders of the
same prompt: the identity bleed is gone (no hair on the alchemists, she
keeps her own clothes through the greeting), the alchemists are furred,
whiskered cats, hands and fabric are correct. So the fixes matter most where
the conditioning is complex; a simple prompt hid them. The user's own
observation stands as well: upstream's guidance is one continuous shot per
prompt, and multi-cut prompts remain the weakest case.

**Stage-1 size is not negotiable** (2026-10-05, later). The 768x512 and
1216x640 renders above ran stage 1 at 384x256 and 608x320; upstream's
distilled and DFR defaults are output 1536x1024 with stage 1 at 768x512
(24x16 latent cells). At 12x8 cells a walking figure is a few cells wide and
the anatomy and gait come out wrong (`beat1_768.mp4`); a static close-up
face survives it, which is why the portrait A/B looked fine. The same prompt
at 1536x1024x121 (`beat1_1536.mp4`, 478.3 s: stage 1 91.9, stage 2 202.0,
decode 168.9) has correct proportions, a natural stride and the written gaze
change. Quality judgements are only valid at the default size; smaller sizes
are for timing, not for looking at. Lightricks' own example prompt uses
"a Caucasian man", so ethnicity words are in-distribution; the earlier note
that they are weak was wrong.

## 21. Closing the gaps: keyframe-aware decode, I2V, dev re-validated (2026-10-05)

**Dev at the default size with section 20's fixes** (`demo/v4/beat1d_dev_1536.mp4`,
1536x1024x121, seed 7): 1540.5 s cold (stage 1 1152.1 s = 30 guided steps at
6,144 tokens x 4 passes, stage 2 227.7 s, decode 139.2 s). Clean, the most
photographic of the three modes; STG 28, the anchored shift, the frozen stage-2
audio and the full negative prompt all in.

**Keyframe-aware diffusion decode** (upstream `keyframes.py`,
`fallback_na/joint_eager.py`, `joint_triton.py`; DFR always decodes this way).
The decoder carries a second stream of P keyframe planes (the stage-2 slot
latents, one pixel frame each; `decoder.type_emb` added before the shared
`conv_in`) through every stage with shared weights; the streams meet only
inside one softmax. A video query sees its own window plus the Kh x Kw window
at its (h, w) on the 2 nearest planes by |t_s(plane) - t| (ties to the lower
index); a plane query sees its own plane's window plus the same window on its
2 nearest video frames. Plane times `t_s(f) = (f + (r-1)/2) / r`, `t_s(0) = 0`,
with r the remaining temporal upsampling (8, 8, 4, 2, 1), so both streams share
one RoPE origin; planes upsample spatially only (phase-1 shuffle), draw their
own stage-5 noise, and their pixels are discarded.

The edge rule is different from the plain decode and it matters: upstream's
joint backends use **centred windows with out-of-volume taps masked**
("clamp-and-mask, not NATTEN's inward shift", matching their Pallas kernel),
where the plain decode shifts the window inside. The first joint kernel reused
the plain rule and missed by rel-L2 0.5; with the centred rule `na3d_joint`
matches upstream's pure-torch `joint_na3d` (staged standalone at
`~/.local/scratch/ltx25/n8/ltxkf`, run on CPU) at **rel-L2 4.0e-7 (video) /
3.5e-7 (planes)**, `n8_joint_na_parity.py`. The kernel is the 8-lane online
softmax with a second slot loop sharing m/l/acc.

DFR 1536x1024x121 with the keyframe decode (`beat1d_dfr_kf_1536.mp4`):
806.7 s (stage 1 125.8, stage 2 415.2, decode 246.7 s vs ~170 s plain: the
plane stream and the joint windows). 5 planes for 121 frames.

**I2V, first-frame conditioning** (`image.py`, `sampling.condition_latent_frame`,
CLI `--image`, API `image` base64 / data URL, `image_strength`): the still is
re-compressed as one H.264 frame at CRF 18 (what a 2.5 checkpoint resolves to
through `LTX_2_4_PARAMS`, not 33), fill-resized (bilinear, align_corners=False,
no antialias) and centre-cropped to each stage's own pixel size, mapped to
[-1, 1], VAE-encoded, and written into latent frame 0 with denoise mask
1 - strength in both stages (`VideoConditionByLatentIndex`,
`image_conditionings_for_chunk`). 13 CPU tests cover the geometry, the
request field and the mask arithmetic.

I2V trial (`i2v_beat1_1536.mp4`, distilled, start frame = frame 60 of
`beat1c_1536.mp4`, seed 11): 483.0 s (image 2.4 s). The clip continues the
still: same character, hall and light, the walk and pull-out carry on. Identity
across clips is therefore available by chaining the last frame of one clip into
the next.

**Whole-decoder parity against upstream, on the CPU** (`n8_keyframe_decode_parity.py`;
the oracle is Lightricks' `DiffusionVideoDecoder` built from the checkpoint by
`ltx_core`'s own builder in torch fp32 on the CPU, `CHUNKED_EAGER` mode, with
the noise it draws captured and fed to ours; 3x8x8 latent, 17 frames of
256x256, planes at pixel frames 4 and 12; ours fp32 operands and stream):

| decode | PSNR vs upstream | note |
| --- | ---: | --- |
| keyframe-aware, first attempt | 67.7 dB | centre of every frame exact; the error sat in the edge columns of the plane frames: the plane pass ran full-width with masked edges while upstream cuts both streams into the same W slabs (replicated halos) |
| keyframe-aware, slab-wise plane pass | **120.0 dB, max-abs 0** | bit-exact to fp32 rounding on all 17 frames |
| plain (no keyframes) | **120.0 dB, max-abs 0** | the first direct check of the plain decoder against upstream rather than the port |

The parity reference for the decoder is now upstream itself, on this machine.

**Transformer and guider parity against upstream, on the CPU**
(`n8_dit_parity.py`; oracle = `ltx_core`'s `LTXModel` built from the official
dev transformer in torch fp32 on the CPU, 84 GiB, under the memory guard;
3x8x12 latent = 288 video tokens + 18 audio tokens, random text context shared
by both sides, first-frame keyframe mask, sigma 0.7):

| check | ours fp16 operands (production) | ours fp32 operands |
| --- | ---: | ---: |
| one velocity forward (dev) | rel-L2 1.6e-3, cos 0.999999 | **5.4e-6** |
| one guided step: cond / negative / STG block 28 / modality passes batched, CFG 3 (video) 7 (audio), STG 1, modality 3, rescale 0.7, through upstream's `BatchedPerturbationConfig` + `MultiModalGuider` vs our `GuidedDenoiser` (STG fork at the first STG block, shared text K/V) | rel-L2 5.5e-3, cos 0.99998 | **1.9e-5** |

The fp32 numbers are fp32 rounding: transformer, perturbation masks, x0
conversion and the guidance arithmetic are semantically identical to upstream.
The production residual is fp16 operand rounding (section 13's 0.0069 at
stage-2 shapes), amplified by the guidance scales. With this, every stage of
the engine has been compared against Lightricks' code rather than the port:
text path by reading (plus the BOS fix), DiT and guider numerically, both
decoders numerically, the samplers by reading with quoted lines.

## 22. Decoder optimization at the production shape (2026-10-05)

With quality judged only at 1536x1024x121 (section 20), the diffusion decoder
was 35% of a distilled run (169.6 s of 480). Profile at that shape
(`n9_diffvae_profile.py`, decoder alone, default MLX limits): 121.8 s, 35 GiB;
stage 5 93%; the `na3d` kernel 60.6 s = **2.1 TF/s against the 19 TF/s MMA
ceiling** (11%); ~28 s of projections, norms and RoPE around it; MLP 19 s;
stages 1-4 8 s.

**Kernel** (`n9_na_kernel_experiment.py`, stage-5 slab 33x256x106, 4 heads x 64,
11^3): one key/value row load now feeds QW adjacent-in-W queries whose windows
overlap in all but QW-1 columns.

| variant | ms | TF/s | vs shipped |
| --- | ---: | ---: | ---: |
| shipped (8 lanes, 1 query) | 408 | 2.99 | 1.00 |
| 8 lanes, QW=2 | 317 | 3.85 | **1.29** (bit-identical) |
| 8 lanes, QW=3 | 898 | 1.36 | 0.45 (register spill) |
| 4 lanes (16 dims per lane), QW=1 / 2 | 597 / 1938 | 2.04 / 0.63 | 0.68 / 0.21 |

**Halo queries.** Every stage-5 slab carries a 5-frame temporal halo on each
side and 5 columns of W halo, and the kernel (and the Q projection and RoPE)
computed those halo positions as queries only to crop them: 43 frames computed
for 33 kept, 106 columns for 96. Both kernels now take a query block with an
offset into the key/value slab, and `_qkv` projects Q for the core only.

Decoder alone, 1536x1024x121: **121.8 -> 91.3 s**, peak 35 -> 33 GiB; kernel
60.6 -> 35.4 s; keyframe decode 106.4 s (joint kernel 41.4 s). All three
upstream parity gates unchanged (joint kernel 4e-7; plain and keyframe decodes
120 dB), the rendered clip bit-identical.

**Buffer cache.** In the pipeline the same decode took 124 s, not 91: the
engine caps MLX's buffer cache at 16 GiB (wired memory next to a 44 GiB DiT),
and the decoder's working set of *distinct buffer sizes* (three or four
slab-shape families, each with q/k/v, RoPE temporaries and outputs) is ~40 GiB;
under the cap nearly every call allocates from the OS. Measured, decoder alone
with ballast for the resident DiT: cache 16 GiB 113 s, 32 GiB 104 s, 48-96 GiB
104 s; without ballast and no cap 91 s (the cache reaches 87 GiB, 40 of it
dead stage-4 giants; clearing the cache between stage 4 and 5 does not help,
stage 5's own set is the problem). The synchronize brackets around the kernel
cost nothing (121.5 vs 121.3 s). `render()` now lends the decoder up to 48 GiB
of cache, parking the text encoder (4.8 s to reload) and then the DiT (6 s)
when that is what makes room, and restores the 16 GiB cap afterwards.

Distilled 1536x1024x121 end to end: **480.2 -> 406.6 s (-15%)**; decode span
169.6 -> 96.7 s; frames identical (`beat1d_1536_v3.mp4`).

Remaining in the decoder (of 91 s): kernel 35 s, projections/RoPE 25 s (per
slab-chunk: 3 projections 22 ms, 2 RMS norms 10 ms, 2 fp32 RoPEs 40 ms, pre()
with fp32 scale/shift 8 ms; a fused RoPE kernel and fp16 modulation would take
~5 s off), MLP 20 s (GEMMs with K=256 at ~7.6 TF/s plus the 4x hidden-state
traffic; ~7 s possible). Each is 1-2% of the distilled wall: at the epsilon.
A tiled simdgroup-MMA kernel was estimated (section 22 notes in the session):
the 11^3 window's union over an 8x8 query brick is 2.7x the useful keys, so
MMA throughput buys ~1.6x on the kernel at best, ~4% of the wall; not pursued.

Resident engine, second clip (`n2_e2e.py --repeat` at 1536x1024x121): 403.8 s
(text reload 5.2 s, DiT reload 3.8 s from the page cache, decode 97.1 s).

**Measured and not shipped** (2026-10-05, later): MLP variants on one stage-5
chunk (fused gate+up GEMM, 1/4/8 temporal chunks): all 285-289 ms = 17.8 TF/s,
already at the GEMM ceiling; the 19.6 s the profiler attributes to the MLP
includes the lazily evaluated fp32 modulation pass. A fused Metal RoPE kernel
(one thread per token-head, fp32 math, fp16 out): 16.7 vs 25.9 ms per call on
the 43x256x106 slab, i.e. ~2.3 s per decode (0.6% of the distilled wall) at
the cost of 1 fp16 ulp against the compiled path (Metal's cos/sin); below the
epsilon and output-changing, not shipped. The simdgroup-MMA NA kernel idea:
the best hand-written simdgroup GEMM of this campaign reached 9.2 TF/s
(section 17) and the 11^3 window's union over an 8x8 query brick is 2.7x the
useful keys, so the useful rate would be ~3.4 TF/s against the shipped
kernel's 3.85; withdrawn.

**Campaign position after section 22.** Output-preserving avenues on the
M1 Ultra, as a share of the 1536x1024x121 distilled wall (407 s): decoder
RoPE 0.6%, decoder modulation in fp16 ~0.5%, decoder MLP ~0 (at ceiling),
NA kernel rewrite ~0, DiT GEMM 0 (95-99% of the MMA ceiling), DiT attention
(MLX SDPA at 78% of the ceiling) ~1.5% here and up to ~5% at DFR's 37k-token
stage 2 - the one item still nominally above the epsilon, and the one with the
least credible path: a hand-written flash-attention kernel would have to beat
MLX's steel SDPA, when our hand-written GEMM reached 49% of MLX's GEMM. Every
remaining lever that is larger changes the output (step count, guidance skip,
fp16 conv VAE, step caching, sparse attention) and belongs to the opt-in tier.

## 23. Remaining-headroom assessment at the production size (2026-10-05)

`n9_forward_profile.py F H W [variant] [batch]` profiles one DiT forward at any
shape with synthetic tokens (every op evaluated synchronously; the clean
forward time and the per-op rates against the 19.4 TF/s ceiling are the
output). 1536x1024x121:

| shape | forward | useful | attention (SDPA) | video GEMMs | other GEMMs | audio GEMMs |
| --- | ---: | ---: | --- | --- | --- | --- |
| stage 2, 24,576 tokens | 67.3 s | 16.9 TF/s, 87% | 32.6 s at 79% (6.9 s to the ceiling) | 30.7 s at 94% (1.9 s) | 4.7 s at 87% (0.6 s) | 0.6 s at 13% (0.5 s; 126 rows, launch-bound) |
| stage 1, 6,144 tokens | 11.4 s | 17.4 TF/s, 90% | 2.4 s at 75% (0.6 s) | 8.4 s at 87% (1.1 s) | 1.5 s at 68% (0.5 s) | 0.6 s at 13% (0.5 s) |
| dev stage 1, batch 4 (plain batch; the guided step's fork saves ~15% on top) | 44.9 s | 17.7 TF/s, 91% | 9.3 s at 78% (2.1 s) | 30.1 s at 97% (0.8 s) | 4.6 s at 91% (0.4 s) | 0.9 s at 37% |

Glue per block at the stage-2 shape (measured one op at a time, eval'd): norm
+ modulate 2.3 ms x3, gated residual 1.7 x3, q/k norms 0.9 x2, head
transposes ~1, RoPE 7.1 x2 (86 GB/s: slow), head gate 0.9, GELU 5.3: ~37 ms
x 48 = 1.8 s per forward (2.7%); bandwidth-optimal would be ~1 s.

As a share of the 407 s distilled wall:

| avenue | gain | feasibility |
| --- | ---: | --- |
| attention at the MMA ceiling | 5% distilled, 4% dev, ~7% DFR (38k-token stage 2 is attention-dominated) | MLX's steel SDPA at 78-79%; MFA, the best published M1 flash attention, 62-86%; our hand GEMM reached 49% of MLX's. Not credible by hand. |
| audio-stream GEMM fusion (q/k/v, cross projections) | ~0.8% | easy |
| decoder RoPE kernel | 0.6% | easy, 1 fp16 ulp |
| DiT RoPE pass | 0.4% | easy |
| decoder modulation in fp16 | ~0.5% | easy, fp16 rounding |
| glue fusion (norm+modulate, residual+norm) | <= 0.6% | moderate |
| uniform decoder slab shapes (cache ~15 GiB, DiT stays resident) | 0.9% resident only | moderate |
| video GEMMs | MLX at 94-97% | nothing |

No single output-preserving avenue is both above the 1% epsilon and
achievable; the sub-epsilon items sum to ~3%. Everything larger is in the
opt-in, output-changing tier (step count, guidance skip, fused LoRA for stage
2 ~1.3% of dev, fp16 conv VAE, step caching, sparse attention).

## 24. The fast tier: output-changing profiles on the same engine (2026-10-06)

Decision: a second set of profiles (`ltx25-distilled-fast`, `ltx25-dev-fast`,
`ltx25-dfr-fast`) that trade exactness for speed, judged by eye and by PSNR
against the exact pipeline at the production size, never by parity. The exact
profiles stay the reference (the thing the parity harnesses prove, and what
the fast ones are measured against). Precedent: section 1 and 10 planned this
tier; Lightricks ships `distilled` as the lossy variant of `dev`; the baseline
runner's HD "recommended config" (`--tile-spatial 2`) is already lossy. The
baseline's TeaCache is disabled for 2.5 packs (its polynomial was fitted on
2.0), so there is no prior Mac step cache for this checkpoint.

Why not int8 (the question that started this): Apple GPUs before M5 have no
int8 MMA; MLX's quantized matmul dequantizes to fp16 inside the kernel and wins
only when bandwidth-bound (M of a few rows). The DiT runs 1.5k-24k rows per
GEMM; measured int8 GEMM 15.8 TF/s vs fp16 19 (section 6), and the baseline's
own code with our fp16 policy and no quantization beat its q8 config, 100.4 vs
116.9 s (section 10). q8 buys memory (22 vs 44 GiB of weights), not speed, on
this machine.

**Mechanisms** (all pipeline-level flags on the exact engine, so every kernel,
precision and memory result of the campaign carries over; `pipeline.Fast`,
profile key `engine.fast`):

- `stage1_sigmas` / `stage2_sigmas`: schedule overrides for the distilled and
  DFR flows and the dev stage 2 (`steps` for the dev stage 1).
- `guidance`: dev Guidance overrides applied to both modalities. A neutral
  scale now drops that pass from the batch (`GuidedDenoiser` builds its row set
  from the non-neutral terms; `Guidance.combine` takes None for a pass not
  run), so `modality: 1.0` runs 3 passes instead of 4. Exact for the full
  configuration (n8 guided step unchanged bit for bit after the restructure).
- `step_cache`: the first-block cache (FBCache formulation, `dit.StepCache`):
  every step runs block 0; if the block-0 residual (hidden after block 0 minus
  the embedded input) has moved less than the threshold in relative L1,
  accumulated since the last computed step, blocks 1..47 are skipped and that
  step's residual over block 0 is added to the current block-0 hidden state;
  the head still runs on the current timestep. The last step of every sampler
  loop is always computed. One cache per loop; works through the dev guided
  batch (the STG fork row is inserted before the residual is applied).
- `attention_tiles` + `attention_halo`: stage-2 video self-attention cut into
  (rows x cols) spatial tiles over the latent grid, every frame; a tile's
  queries attend to the keys of its own cells widened by `halo` latent cells
  (DFR slot and reference tokens are keys of every tile and attend globally).
  One SDPA call per tile, gathered with `mx.take`. Attention is 32.6 of the
  67.3 s stage-2 forward at 24,576 tokens (section 23); 2x2 tiles with a
  2-cell halo keep 30% of the keys (18 of 32 rows x 26 of 48 columns).
- `decoder: conv` (existing): the conv VAE instead of the diffusion decoder,
  97 -> ~30 s at 1536x1024x121, 36-38 dB from the diffusion output.

**Method.** `n10_fast_ab.py MODE OUTDIR NAME:JSON ...`: one resident engine,
the beat1d prompt at 1536x1024x121 seed 7, one lever at a time against the
exact render of the same run (`exact.npy`, uint8 frames before the mux), then
the stack. PSNR against exact is the number; the contact sheets and the clips
(`~/.local/scratch/ltx25/n10/<mode>/`) are the judgement, reviewed by the
user. A lever stays in a profile only if the clip is not visibly worse.

**Distilled, one lever at a time** (1536x1024x121, seed 7, beat1d, resident
engine; `n10/distilled/results.tsv`). The exact run of this session: 413.5 s
(stage 1 95.1, stage 2 202.1, diffusion decode 101.0).

| lever | wall | saved | PSNR vs exact | reading |
| --- | ---: | ---: | ---: | --- |
| conv decoder | 336.0 | 77 s (19%) | 35.3 dB (min 30.5) | the known decoder difference; softer fine texture |
| stage 2: 3 -> 2 steps (0.909, 0.42, 0) | 330.7 | 67 s (16%) | 30.8 dB (min 28.5) | same shot, slightly less mosaic detail at sheet scale |
| stage 1: 8 -> 5 steps (drop the four sigma 0.975-0.99 steps) | 367.7 | 35 s (9%) | 15.4 dB | a different sample (the near-1 steps fix the composition), clean; not a degradation but not the same clip |
| step cache 0.10 | 399.3 | 0 | identical | **never fires**: the block-0 residual moves 22-70% between consecutive steps (ancestral re-noising, large sigma jumps); no threshold short of "skip everything" would. Dead on the distilled schedules. |
| attention tiles 2x2, halo 2 | 338.3 | 63 s (15%, stage 2 202 -> 139) | 23.2 dB (min 21.2) | **visible seam across the face** in the opening close-up (the boundary runs through the frame centre). Rejected at this geometry. |

**Dev, one lever at a time** (same clip; `n10/dev/results.tsv`). Exact this
session: 1513.4 s (stage 1 1156.1 = 30 guided steps x 4 passes, stage 2 228.3,
decode 102.5).

| lever | wall | saved | PSNR vs exact | reading |
| --- | ---: | ---: | ---: | --- |
| modality guidance off (3 passes) | 1180.2 | 333 s (22%) | 22.4 dB | same shot, clean; what it costs is audio-video alignment, which only listening judges |
| 30 -> 20 steps | 1129.3 | 384 s (25%) | 22.5 dB | same shot, same quality at sheet scale |
| step cache 0.10 | 1099.8 | 414 s (27%) | 21.8 dB | 11 of 30 stage-1 steps skipped (rel-L1 0.06-0.12 per step from step 4 on, alternating skip/compute under the accumulator); stage 2 never skips (0.5-0.65 per step). Clean. |
| attention tiles 2x2, halo 2 | 1454.0 | 59 s (4%; stage 2 only) | 23.9 dB | the same seam through the face. Rejected. |

The three dev levers act on different things (passes, steps, repeated steps)
and should compose; the stack is measured below.

**DFR, one lever at a time** (`n10/dfr/results.tsv`). Exact this session:
680.7 s (stage 1 127.6, stage 2 418.2 at ~38k tokens, keyframe decode 119.4).

| lever | wall | saved | PSNR vs exact | reading |
| --- | ---: | ---: | ---: | --- |
| stage 2: 3 -> 2 steps | 527.1 | 154 s (23%) | 32.8 dB | same shot |
| step cache 0.10 | 664.0 | 0 | identical | never fires (0.22-0.67 per step), as distilled |
| attention tiles 2x2, halo 2 | 607.9 | 59 s (9%; stage 2 418 -> 359) | 28.5 dB | **no seam** here, checked at full resolution on the opening close-up (`tiles_face_cmp.png`): the clean half-resolution reference tokens and the slot planes are keys of every tile and anchor it. Tiling is usable on DFR only. |

**Distilled stacks.** A = conv decoder + 2-step stage 2: **277.9 s** (exact
413.5; 1.49x), 29.6 dB, the same shot. B = A + 5-step stage 1: **229.3 s**
(1.80x), clean, a different composition (15.4 dB, as the lever alone). The
profile ships A; B is the documented opt-in (`fast.stage1_sigmas`), because
one prompt is not evidence that the four near-sigma-1 steps the distillation
was trained with can go in general.
Tiling retried gently on distilled, 1x2 tiles with a 6-cell halo (the key
set is 66% of the frame): 365.3 s, 31 s saved (stage 2 202 -> 171), 24.8 dB,
no seam at full resolution on the close-up (`tiles12h6_face_cmp.png`). Kept
as an opt-in, not in the profile: 8% for a seam risk that one prompt cannot
rule out.

**Dev stacks.** A = modality off + 20 steps + step cache 0.10 + conv decoder:
**730.4 s** (exact 1513.4; 2.07x), 21.2 dB, the same shot. With 20 steps the
per-step moves are larger and the cache skips 4 of 20 (the levers overlap:
fewer steps leaves less for the cache). B = A + 2-step stage 2: **644.2 s**
(2.35x), 21.4 dB, the same shot. The profile ships B.

**DFR stacks.** A = conv decoder + 2-step stage 2: **455.2 s** (exact 680.7;
1.50x), 28.4 dB min. B = A + 2x2 tiled stage-2 attention (halo 2): **421.0 s**
(1.62x), 27.1 dB, the same shot, no seam. The profile ships B.

**The fast profiles** (`ltx25-*-fast`, `engine.fast` + `decoder: conv`),
1536x1024x121, M1 Ultra, resident engine:

| profile | exact | fast | vs exact | vs the baseline runner |
| --- | ---: | ---: | ---: | --- |
| `ltx25-distilled-fast` | 413.5 s | **277.9 s** (229 s with the opt-in 5-step stage 1) | 1.49x (1.80x) | 505 s tiled (its lossy HD config): 1.8x (2.2x) |
| `ltx25-dev-fast` | 1513.4 s | **644.2 s** | 2.35x | cannot run HD |
| `ltx25-dfr-fast` | 680.7 s | **421.0 s** | 1.62x | not measured at HD |

What did not make it, with the reason: the step cache on the ancestral
schedules (never fires), tiled attention on distilled and dev (seams through
faces; the 1x2/halo-6 geometry passed one prompt and is an opt-in), the
5-step distilled stage 1 (a different sample, not a worse one; opt-in). All
levers stay available through `engine.fast` for anyone who wants them.
Artifacts: `~/.local/scratch/ltx25/n10/{distilled,dev,dfr}/` (mp4, uint8
frames, contact sheets, `results.tsv`); the clips are for the user's review.

**Review (the user, 2026-10-06).** No meaningful difference between the exact
and the fast clips in any mode; dev is a lot better than distilled and DFR in
quality and, most importantly, prompt adherence; in the dev folder stack A
(3-step stage 2) looks slightly better than stack B. Decision: `ltx25-dev-fast`
ships stack A (730 s, 2.07x); the 2-step stage 2 is an opt-in. Analysis notes
behind it (`review_<mode>.png`, full-resolution crops; audio envelopes): every
visible difference is the conv decoder (wirier hair strands, slightly softer
mosaic); the step / pass / cache levers are invisible at full resolution;
frame-to-frame change 0.95-0.99x of exact (no flicker); dev with the modality
pass off keeps the footstep timing exactly (onsets 1.3 / 2.0 / 2.6 / 3.3 s in
both) but mixes ~9 dB quieter; DFR audio is bit-identical (it comes from stage
1). Prompt note for future renders: no close-up that pulls out from a face
while the subject walks; one steady continuous shot.

**Dev audio level (2026-10-06, after the review).** The fast stage-1 levers
lower the audio level: relative to exact, modality pass off -1.6 dB, 20 steps
-3.9 dB, step cache -3.5 dB, stack A (all three) -6.6 dB, stack C (20 steps +
cache + conv, all four passes) -3.4 dB at 911.4 s (1.66x). Footstep onsets are
at the same instants in every variant (1.3 / 2.0 / 2.6 / 3.3 s): the sync is
intact, the mix is quieter. It is systematic (every lever that takes
denoising work away from stage 1 moves it the same way), and in the dev
pipeline the audio is frozen in stage 2, so it only ever gets stage 1's
steps. Tried: re-noising the stage-1 audio to the stage-2 start sigma and
refining it with the video for the 3 stage-2 steps (what distilled does):
-3.2 dB, onsets perturbed (1.1 / 1.9 / 2.5 / 3.2). Measured negative, code
removed. Dropping the modality pass is out of the profile. The remaining
-3.4 dB of stack C is a decision for the user: ship it as a documented level
difference, or keep dev's stage 1 whole (then dev-fast is the conv decoder
alone, 1513 -> ~1444 s, not worth a profile).
Split by content (`dev/*.wav`, 10 ms RMS envelope): the background floor is
-62 dB in every variant and the footstep timing is identical; what moves is
the footstep peaks: exact -42 dB, 20 steps -46, cache -46, modality off -44,
stack A -50, stack C -45. Less denoising work gives softer generated
transients, the audio counterpart of the softer hair strands and mosaic in
the picture; not a decode fault (the decoder path is bit-identical on DFR
and on the tiles variant, 0.0 dB). Decision (user, 2026-10-06): that is the
model and the diffusion working as designed; `ltx25-dev-fast` ships stack C
(911.4 s, 1.66x) with the tradeoff stated in the profile note.

## 25. The component checklist, resumed (2026-10-06)

The HANDOFF's "nothing skipped" checklist still had five unbuilt items
(temporal rounds, second spatial epilogue, duration head, prompt enhancer,
res_2s); section 23 had mislabelled them as out of scope. Resumed in order of
user value.

**Duration head** (`duration.py`, `n11_duration_parity.py`): upstream's
DurationHead (modality projections + modality embeddings, one learnable query
cross-attending the connector tokens with 4 heads, GELU MLP, exp) ported in
fp32 from `model_patches/ltx-2.5-duration-head-bf16.safetensors`. Parity
against the torch head on random connector tokens: rel 9e-8 (video-only,
audio-only, both). On real prompts (24 fps): beat1d 10.74 s -> 257 frames,
"a red fox trotting ..." 4.55 s -> 105, a one-line greeting 2.43 s -> 57, a
cloud time-lapse 5.92 s -> 137. Wired as upstream's auto duration: a request
with neither `seconds` nor `num_frames` is predicted, clamped to 1-20 s,
snapped to 8k + 1, then capped at the profile's token envelope for the clip's
size (121 frames at 1536x1024; 505 at 768x512); frames larger than the
validated frame are refused. End to end through the CLI at 512x320: the
greeting rendered 57 frames (2.4 s). The text-encoder tower was refactored
into per-layer pieces on the way (bit-identical on the beat1d encode).

**Prompt enhancer: blocked on a second model.** The resident tower, run as a
language model (tied embeddings, final norm, logit soft-cap, Gemma-4 chat
template, greedy with no 5-gram repeats as upstream's generate kwargs),
emits garbage: the LTX fine-tune is not a generative model, and upstream's
own code refuses `enhance_prompt` with a Gemma-4 encode root unless
`--prompt-enhancer-gemma-root` names a separate generative instruct checkpoint
("gemma3 or gemma4 E2B-it"). That is a new download (gated, ~5 GB) and a
second small model to port; not started without the user's go-ahead. The
cached-generation code is parked in `~/.local/scratch/ltx25/n11/
text_generate_wip.patch`.

## 26. DFR temporal rounds and the second spatial epilogue (2026-10-06/07)

Both DFR options that the HANDOFF listed and section 23 had written off are
in: `temporal_upscalings` (0-2) and `spatial_upscalings` (1-2), API and CLI,
`pipeline.dfr` -> `_temporal_round` / `_spatial_epilogue`, ported from
`ltx_pipelines/dfr_stages.py`, `dfr_helpers/{layout,ops}.py` and
`ltx_core/tiling.py`.

**Temporal round.** The stage-2 canvas is x2 temporally upsampled (the planes
move to 2x their pixel frames), the canvas is cut on its keyframe seams into
2**round windows (largest-first, remainder to the leading ones), and each
window is re-denoised from sigma 0.975 (DISTILLED_SIGMAS[4:], ancestral, eta
0.5, noise seed seed + 1000 round + tile) as its own clip: its seams pinned as
0.95-strength anchor keyframes (appended unmarked tokens), one generated slot
per segment midpoint seeded from the nearest cell, a non-first window starting
on the last plane before its seam with the previous window's cells to the
seam held fixed (the lead-in, strength 1 at index 0), the stage-1 audio window
resampled to the window's token count and frozen. Owned cells (after the
pinned prefix) are stitched; new planes join the carry bag. Layout
(`sampling.temporal_tile_plan`, `tile_prefix`, `split_at_seams`) matches
upstream's for 10 canvases x round counts and 3 prefixes
(`tests/slimserve/data/ltx25_temporal_layout.json`, captured from
ltx_pipelines).

Parity (`n11_temporal_round_parity.py`; upstream's run_one_temporal_round on
the CPU through ltx_pipelines with OpenImageIO stubbed, the transformer built
in fp32 by ltx_core's builder and the sampler's latent updates pinned to fp32;
ours with fp32 operands and upstream's noise replayed draw by draw): on a real
stage-2 canvas (our DFR at 256x256x49, two planes, real audio and text) one
round (97 frames, two windows, two anchors, two new slots) ends at rel-L2
**1.2e-3** on the latent and 1.1e-3 on the carry planes. On random tensors
the per-step capture showed every window's first step exact (inputs 6e-7, x0
6e-5 / 2e-4) and the forward amplifying input error ~10x per step at low
sigma - the model, not the port. Two defects found and fixed on the way:
`condition_latent_frame` re-blended every token with its own mask (a no-op at
masks 0/1, so nothing shipped changes; wrong for the 0.95 anchors once the
lead-in pinned them), and the reference harness's own dtype plumbing (the
stage builds bf16 and disposes per window; the loop defaults to bf16
updates).

Timings: 512x320x49 round 1 25.7 s (97 frames at 48 fps), round 2 +54 s (193
at 96); no motion spikes at the seams (frame-to-frame change flat).

**Spatial epilogue** (`spatial_upscalings` 2; sizes multiples of 128): stage 1
at a quarter, stage 2 at half, then at the output size: the carry planes and
(without a still) the opening cell decoded one at a time with the diffusion
decoder, Lanczos x2 in RGB, re-encoded as one-frame planes; the stage-2 latent
x2 spatially upsampled; keyframe-seam windows (one window without temporal
rounds - upstream's plan fails on an empty seam list there) each denoised
under the detailing IC-LoRA with the stage-2 cells as reference tokens, the
planes pinned at strength 1, the lead-in pinned, frozen resampled audio: one
plain Euler step from sigma 0.909 on 2x2 spatial tiles (10-cell overlap,
trapezoid blend, a conditioning token in every tile its extent overlaps,
averaged), then the remaining steps on 4x4 tiles (`TiledDenoiser`; tile
layout and masks match upstream's `split_by_count` / trapezoids,
`ltx25_spatial_tiles.json`; 4x4 triple-covers a few cells to 1.08 as upstream
does, nothing renormalizes). The rebuilt planes drive the keyframe decode.

Result: 2048x1024x49 from the same base as a 1024x512 DFR clip (identical
stages 1-2, same seed): the same shot, visibly sharper fur, snow and branches,
no tile seams (`n11/epilogue_ab2_crop.png`); 516 s (epilogue 375, decode 68).
At 1024x512 the x2 path puts stage 1 at 256x128 and is softer than plain DFR:
the option is for output sizes above the two-stage envelope, as upstream
uses it. Not parity-checked against upstream's run_spatial_epilogue (its
VideoDecoder/ImageConditioner/TiledDiffusionModel chain on the CPU is a day
of its own); its pieces are: the layout (fixtures), the windows (the temporal
round's code), the IC-LoRA stage (DFR stage 2), the decoders and encoder.

## 27. The HQ pipeline: res_2s (2026-10-07)

`ltx25-hq` = upstream's TI2VidTwoStagesHQPipeline (`pipeline.hq`, dev
variant): LTX_2_3_HQ_PARAMS is a plain constant (no 2.5 override): 15 steps,
CFG 3 / 7, STG off, modality 3, rescale 0.45 video / 1.0 audio; the dev
transformer with the distilled LoRA at 0.25 in stage 1 and 0.5 in stage 2;
the stage-1 schedule shifted by the stage-1 latent's token count (upstream
hands the scheduler the latent here, unlike the dev pipeline); stage 2 the
3 distilled sigmas with the audio re-noised to 0.909 and refined (it ships
from stage 2). The guided step is the 3-pass batch (no STG term).

**res_2s** (`sampling.res2s_loop`, from utils/res2s.py, samplers.py and
Res2sDiffusionStep): per step h = -log(sigma_next / sigma); x0 at sigma; the
midpoint x + h a21 (x0 - x) at sqrt(sigma sigma_next), SDE-noised from the
substep stream at eta 0.5 (sigma_up = eta sigma_next, alpha_ratio = (1 -
sigma_next) + sqrt(sigma_next^2 - sigma_up^2), the result blended on the
mask); when h < 0.5 and sigma > 0.03 the anchor is refined 100 times
(x_anchor = x_mid - h a21 eps_1, eps_1 = x0 - x_anchor); x0 at the midpoint;
x_next = anchor + h (b1 eps_1 + b2 eps_2) with a21 = c2 phi_1(-h c2), b2 =
phi_2(-h) / c2, b1 = phi_1(-h) - b2, c2 = 0.5; SDE noise from the step
stream at eta; a terminal 0 becomes 0.0011 and the loop ends on that x0. The
noise is standard normal standardized globally then per batch row. Parity
(`n11_res2s_parity.py`: both loops around the same synthetic denoiser 0.5
tanh(xW) + 0.3 x, upstream's draws replayed, upstream in float64): stage-1
schedule video 3.1e-5 / audio 2.4e-4, stage 2 2.4e-5 / 1.7e-4 (fp32 against
float64 through 15 x 2 evaluations and the 100-iteration refinement).
Upstream's HQ pipeline never passes a noise seed to the loop (it runs at the
loop's default); ours seeds both streams from the request seed.

Smoke: 512x320x33 in 131 s (stage 1 88 s = 15 steps x 2 evaluations x 3
passes, stage 2 18 s), a coherent clip. Profile `ltx25-hq` (the dev file set).
1536x1024x121 (beat1d, seed 7, `n11/hq_beat1d_1536.mp4`): **1901.7 s**:
stage 1 1249.3 s (15 steps x 2 evaluations x 3 passes, with the LoRA), stage 2
534.6 s (3 steps x 2 evaluations + the terminal x0 = 7 forwards at 24.5k
tokens, with the LoRA), decode 98 s. Slower than dev's 1513 s: res_2s halves
the steps but doubles the evaluations, the LoRA adds its +22% to both stages,
and stage 2 runs 7 forwards instead of 3. A clean clip with the prompt's
elements in it.
