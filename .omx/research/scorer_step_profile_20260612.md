# Scorer step profile — where the per-step time goes, and the realistic speedup ceiling (2026-06-12)

**Authority:** `[macOS-CPU advisory]` (a TIMING benchmark, not a score). Measured on the M5 Max torch-CPU
substrate. Frontier UNMOVED 0.19109982 `[contest-CPU]`. $0, local, no MPS, no MLX-GPU run.

**Source data:** `.omx/tmp/throughput_mlxplan_n8_confirm.json` (produced by
`experiments/measure_local_scorer_throughput.py` — the EXACT upstream `modules.py` EfficientNet-B2 SegNet +
FastViT-T12 PoseNet with the REAL safetensors weights at the canonical 512×384 input). n=8 pairs,
base_channels=20, torch threads=4. This memo READS that JSON; it does not re-run the GPU.

## The headline number

The scorer forward+backward is **98.36%** of the per-step wall-clock. Everything else — render, eval-roundtrip
resize, preprocess, optimizer — is the remaining **1.64%**. This is the single most important fact for the
whole speedup program: **any speedup that does not attack the scorer forward+backward is rounding error.**

## Per-stage breakdown (torch-CPU, n8, base_ch=20)

| Stage | ms / step | % of full step | Notes |
|---|---:|---:|---|
| MLX render + sync | 25.62 | 0.12% | renderer forward (the part we are NOT trying to speed up) |
| np copy render→leaf | 0.00 | 0.00% | negligible |
| eval-roundtrip bicubic (384↔874) | 277.66 | 1.30% | the uint8 roundtrip simulation; cheap |
| permute→contiguous | 0.01 | 0.00% | negligible |
| preprocess input (resize + rgb_to_yuv6) | 46.00 | 0.22% | negligible |
| **SegNet forward** | **5830.55** | **27.37%** | EfficientNet-B2 U-Net — the dominant FORWARD term |
| **PoseNet forward** | **601.84** | **2.83%** | FastViT-T12 — ~10× cheaper forward than SegNet |
| forward subtotal | 6756.05 | 31.72% | |
| **backward (estimate)** | **14519.29** | **68.16%** | the dominant term overall — ~2.15× the forward |
| **full step (fwd+bwd)** | **21300.96** | **100%** | ~21.3 s/step at n8 |
| → scorer fwd+bwd fraction | — | **98.36%** | the lever lives here |

The eval-pass cost block confirms a separate fact relevant to the gate's eval cadence: a **fused d_seg+d_pose
eval** is only **1.012×** faster than computing them separately (9065.6 ms vs 9175.3 ms) — fusing the two
eval terms is NOT a meaningful lever; the cost is in the shared scorer forward, not in running it twice.

## Within the scorer: which term, forward vs backward

- **Backward dominates: 68.16% of the step** (14.5 s) vs forward 31.72% (6.8 s) — a ~2.15× fwd→bwd ratio,
  which is the textbook depthwise-conv backward cost (the backward computes both the input-gradient and,
  here, propagates `dL/d(pixels)` all the way back through the frozen scorer to the rendered pixels).
- **SegNet is the heavy module: 27.37% (forward) vs PoseNet 2.83% (forward)** — SegNet's forward is ~9.7×
  PoseNet's. The backward is not separated by module in this run, but it scales with the forward graph, so
  SegNet backward is also the dominant backward term. **SegNet (EfficientNet-B2) is the module to attack.**
- PoseNet is cheap in throughput but EXPENSIVE in risk: it is the axis that diverged at n600. So PoseNet is
  a *small* throughput lever but the *load-bearing* correctness gate. Throughput optimization should target
  SegNet; the acceptance gate must always watch PoseNet.

## Why the convs are the bottleneck (bandwidth-bound, the mechanism)

On Apple-Silicon arm64, torch has **no mkldnn / no MKL** (`torch_mkldnn_available: false`,
`torch_mkl_available: false` in the JSON). EfficientNet-B2's depthwise (grouped) convolutions therefore
dispatch to the naive `aten::_slow_conv2d_forward` reference kernel. Depthwise conv is **memory-bandwidth-
bound, not compute-bound**: it does ~1 MAC per weight load with almost no data reuse, so it is gated by how
fast the hardware can stream activations and weights, not by FLOPs. This is why (a) the convs dominate and
(b) the speedup levers that matter are the ones that cut **bytes moved** (lower precision = fewer bytes;
fusion = fewer round-trips to memory), not the ones that cut FLOPs.

## The biggest lever and the HONEST ceiling (Amdahl)

The single biggest lever is the **scorer backward (68.16%)**, and second the **SegNet forward (27.37%)**.
Together the scorer fwd+bwd is **98.36%** of the step. Amdahl's law on a 98.36%-dominant section:

| If you make the scorer fwd+bwd … | the WHOLE step speeds up by at most | total step time (from 21.3 s) |
|---|---:|---:|
| 2× faster | **1.97×** | 10.8 s |
| 3× faster | **2.89×** | 7.4 s |
| 5× faster | **4.61×** | 4.6 s |
| 10× faster | **8.46×** | 2.5 s |
| ∞ (free scorer) | **61.0×** (hard cap) | 0.35 s |

**The realistic ceiling is set by precision, not by "engineer harder".** A bf16/fp16 scorer fwd+bwd is a
~2× bandwidth win (halve the bytes streamed for the bandwidth-bound convs) → **≈1.97× total step** as the
honest first-order cap from precision alone. `mx.compile` fusion adds a multiplicative factor on top
(fewer kernel launches + fewer intermediate round-trips), plausibly pushing the *combined* scorer speedup
toward 3–5× → **≈2.9–4.6× total**. Beyond ~5× scorer speedup you are chasing the long tail of the
remaining 1.64% and the wins flatten hard. **Do not promise more than ~3–5× total step throughput from the
precision+fusion program; the 98.36% section caps it there.** The only way past that cap is to remove the
scorer from the per-step gradient loop entirely (a distilled surrogate), which is a *different* program
with its own (large) fidelity risk and is out of scope for this profile.

### One caveat on the n: this profile is n8

The breakdown FRACTIONS are n-independent (the scorer dominates at any n — it is per-pair work), so the
Amdahl ceiling holds at n600. The ABSOLUTE per-step time grows ~linearly with n (n600 ≈ 75× the per-epoch
steps), which is exactly why the throughput program matters (n600 torch-CPU ≈ 18–19 min/epoch). The ceiling
is a *fraction*-level statement and transfers; the absolute times do not.

## What this means for the plan

1. Attack the scorer fwd+bwd or attack nothing. The render/preprocess/optimizer are already free.
2. SegNet (EfficientNet-B2 depthwise convs) is the throughput target; PoseNet is the correctness tripwire.
3. The convs are bandwidth-bound ⇒ precision (bytes-moved) and fusion (round-trips) are the right levers,
   FLOP-cutting is not.
4. The honest total-step ceiling from the precision+fusion program is ~3–5×. Anything claiming more is
   either removing the scorer from the loop (different program) or mis-attributing the n.
5. Every lever's correctness is adjudicated by the BOTH-TERMS acceptance gate
   (`tac.mlx_pr95_port.speedup_acceptance_gate`) at the REAL n — never on d_seg alone, never at n8 only.

## Provenance / NO-FAKE

- All numbers above are READ from `.omx/tmp/throughput_mlxplan_n8_confirm.json` (a measured torch-CPU
  timing run). The Amdahl projections are explicitly labeled as **projections from the measured 98.36%
  fraction**, not measured total-step speedups. No MLX-GPU benchmark was run for this memo; the MLX-GPU
  validation happens later through the acceptance gate. The `[advisory]` axis is preserved per CLAUDE.md.
