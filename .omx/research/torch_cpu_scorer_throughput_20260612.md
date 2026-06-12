# torch-CPU frozen-scorer throughput — correct-by-construction speedup audit (2026-06-12)

`[macOS-CPU advisory / engineering-throughput; NOT a score claim]`
Lane: `lane_torchcpu_scorer_throughput_20260612`
Subagent: `torchcpu-throughput-20260612`. $0, local, no MPS, no paid dispatch.

## TL;DR — HONEST NEGATIVE (lead with the wall-clock verdict)

The capstone training step is >97% frozen-scorer (EfficientNet-B2 SegNet +
FastViT-T12 PoseNet) fwd+bwd. I profiled it and tested every **correctness-
preserving** CPU lever (memory layout / fusion / freezing / threads / FPU mode)
with a BOTH-terms (d_seg AND d_pose) descent-equivalence gate.

**Verdict: there is NO net correct-by-construction CPU speedup available on this
hardware. Every lever is REJECTED on the real full SegNet+PoseNet bridge path.**
This is a clean negative, banked so a future agent does not re-spend on these
levers. The frontier pointer is UNMOVED (0.19109982 [contest-CPU]); this was a
throughput investigation, not goal progress.

| Lever | full-bridge s/step ratio | gradient parity | Verdict |
|---|---|---|---|
| `channels_last` (NHWC) | **0.65x (35% SLOWER)** | cosine 1.0, d_seg/loss delta 0 (CORRECT) | REJECTED — routes fwd to `aten::_nnpack_spatial_convolution` + generic-slow backward; net loss |
| `torch.compile` (inductor) | **0.85x (15% SLOWER)** + 120s compile/resume | cosine 1.0, d_seg delta 0 (CORRECT) | REJECTED — inductor CPU kernels don't beat eager `_slow_conv2d` here; recompiles every daemon resume |
| `flush_denormal` (FPU FTZ) | **0.917x median / 0.969x mean (NEUTRAL-to-SLOWER)** | cosine 1.0, d_seg/d_pose/loss delta 0 (CORRECT) | REJECTED — a SegNet-ONLY microbench showed +9% but it did NOT replicate on the full bridge (FastViT pose path isn't denormal-bound); net neutral-to-negative |
| `to_mkldnn` / oneDNN | **N/A — absent** | — | REJECTED — `MKL-DNN build is disabled` (`mkldnn.is_available()==False`, `mkl.is_available()==False`); the entire textbook CPU-conv playbook has no backend to target |
| thread count | already at the measured sweet spot (6) | bit-stable d_seg | NO CHANGE — daemon pins `min(6,...)`; flat 6–10, degrades ≥14 |

**Why the hoped 2–4x channels_last win is unavailable:** that win is an *oneDNN*
win. This arm64 torch 2.11.0 build has NO oneDNN/MKL, so convs dispatch to the
naive reference kernel (`aten::_slow_conv2d`), and the only alternative backend
(NNPACK, selected by channels_last forward) is slower here for both the forward
and the (un-accelerated) backward.

**The real throughput lever is NOT the CPU scorer — it is the MLX-GPU gradient
path** (owned by the sibling subagent fixing its n600 pose-gradient divergence).
torch-CPU is currently both the authority gradient and ~19s/step; no layout/
fusion/FTZ change moves it on this box.

## Profile (the bottleneck, the evidence)

Baseline full bridge `loss_and_pixel_grad`, B=8 pairs, threads=6: **12.2 s/step
uncontended** (~18 s/step under the live daemon's contention). Self-CPU one step:

```
aten::_slow_conv2d_backward    38.48%   7.029s   (25798 calls)
aten::_slow_conv2d_forward     30.17%   5.510s   (25798 calls)
aten::copy_                    13.47%   2.459s   (77327 calls)   <- im2col churn
aten::cat                       5.68%   1.037s   (Unet skip concats)
aten::native_batch_norm_*       ~2%
```

~69% is the naive reference conv; the rest is im2col copy churn + Unet concats.
The MLX render / numpy copy / eval_roundtrip are <3% combined — there is no host-
overhead win worth chasing.

Frozen-weight backward is ALREADY minimal: all scorer params are
`requires_grad=False` (0 trainable), so autograd already passes
`output_mask=[True,False,False]` (grad_input only, no wasted grad_weight). No win
there either.

channels_last profile confirms the kernel routing: forward moved to
`aten::_nnpack_spatial_convolution` (15.997s) and the backward stayed on a
generic-slow `aten::convolution_backward` (16.483s) — a net 35% regression.

## The descent-equivalence gate (the n600 anti-anchor)

Per `.omx/research/mlx_custom_backward_DIVERGES_at_n600_pose_gradient_20260612.md`:
an MLX custom-backward kernel was validated on d_seg ONLY and DIVERGED at n600
because its POSE gradient was wrong (d_pose 0.8→7→36). So I gated every lever on
BOTH terms. **All three software levers measured cosine 1.0 / d_seg delta 0 /
d_pose delta 0 — they are numerically PERFECT (correct-by-construction).** They
were rejected purely on SPEED, never on signal. (flush clean A/B: grad_cosine
1.0000000000, d_seg delta 0, d_pose delta 0, loss delta 0; speed 0.917x median.)

## What was adopted

**Nothing in code.** Wiring a flag named `--fast-cpu-scorer` that does not make
the real bridge faster would be a fake speedup (CLAUDE.md NO-FAKE supreme rule:
"do LESS, but make it REAL"). The candidate opt-in (`flush_denormal` on
`configure_torch_cpu_threads` + a daemon `--fast-cpu-scorer` flag) was implemented,
tested (4 NO-FAKE descent-equivalence tests passed), then REVERTED when the
robust interleaved A/B proved it is neutral-to-negative on the full bridge. The
durable output is this negative-finding memo + the rejected-lever table.

## Relaunch the basin daemon (UNCHANGED — no speedup to add)

The live daemon (pid 42035, out-dir
`experiments/results/capstone_n600_correct_faithful_20260612T010134`) needs NO
change: there is no measured CPU speedup to adopt. Its existing launch/resume is
already optimal on the thread axis (`min(6,...)`). Do NOT add `channels_last`,
`torch.compile`, or `flush_denormal` — all three are measured slower-or-neutral
on this box. The throughput unlock to pursue instead is the MLX-GPU gradient
path once its pose gradient is fixed (sibling subagent).

## Measurement methodology (so the negative is trustworthy)

All ratios are baseline-vs-optimized measured BACK-TO-BACK within one process
under the same contention, so the live daemon's CPU load cancels in the ratio.
The flush verdict used an 8-round INTERLEAVED A/B (OFF/ON alternating) and
reports the median (0.917x) and mean (0.969x) to cancel contention drift — the
isolated SegNet-only +9% did not survive this. Bench scripts:
`.omx/tmp/bench_levers.py`, `.omx/tmp/test_channels_last.py`,
`.omx/tmp/test_compile.py`, `.omx/tmp/validate_flush_denormal.py`,
`.omx/tmp/ab_flush_interleaved.py`, `.omx/tmp/probe_backward.py`.

## 6-hook wire-in (per CLAUDE.md "Subagent coherence-by-default")

1. sensitivity-map — N/A (throughput investigation; no per-byte sensitivity).
2. Pareto — N/A (does not change rate/seg/pose; only wall-clock, and it didn't).
3. bit-allocator — N/A.
4. cathedral autopilot dispatch — N/A (local CPU knob, not archive-deployable).
5. continual-learning posterior — THIS memo + the rejected-lever table ARE the
   durable negative signal: channels_last (0.65x), torch.compile (0.85x),
   flush_denormal (0.917x), mkldnn (absent) are negative anchors so a future
   agent does not re-test them on no-mkldnn Apple Silicon. The throughput lever
   is the MLX-GPU gradient path, not the CPU scorer.
6. probe-disambiguator — N/A (the lever table IS the empirical arbitration; each
   row is a measured ratio, not an interpretation).

## Honest scope

Throughput investigation, not goal progress. The exact frontier pointer is
UNMOVED (0.19109982 [contest-CPU]). The deliverable is a measured, BOTH-terms-
gated NEGATIVE: no correct-by-construction CPU scorer speedup exists on this
hardware, and the rejected-lever evidence redirects future effort to the
MLX-GPU path.
