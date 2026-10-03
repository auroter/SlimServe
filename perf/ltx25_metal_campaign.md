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
