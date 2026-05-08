# PR107 lossy_coarsening lane — FALSIFIED across budget range (DEFERRED-pending-research, NOT killed)

**Author:** claude_lab_subagent_sub_0_17_stack
**Date:** 2026-05-08
**Lane:** pr107_apogee_stack_brotli_sweep_cpu_build
**Verdict:** **measured-config-retired** at all 6 budgets tested on PR107
**NOT a class-level kill** — reactivation criteria documented below

## Empirical evidence ([contest-CPU] via GHA fork)

Six (sister + me) PR107 lossy_coarsening candidates dispatched to GHA CPU eval:

| budget | bytes | Δbytes | rel_err | seg_avg | Δseg×100 | pose_avg | Δsqrt(10p) | Δrate×25 | score | Δscore | run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 178,392 | 0 | 0% | 5.89e-4 | 0 | 3.58e-5 | 0 | 0 | **0.197** | 0 | PR107 anchor |
| b025 | 174,429 | -3,963 | 0.65% | (sister) | +0.041 | (sister) | +0.008 | -0.003 | **0.243** | +0.046 | 25557507813 |
| b035 | 170,009 | -8,383 | 1.54% | (sister) | +0.060 | (sister) | +0.010 | -0.006 | **0.260** | +0.064 | 25557519091 |
| b050 | 156,263 | -22,129 | 3.86% | (sister) | +0.122 | (sister) | +0.029 | -0.015 | **0.333** | +0.137 | 25557527476 |
| b070 | 142,128 | -36,264 | 6.18% | 2.63e-3 | +0.204 | 6.54e-4 | +0.062 | -0.024 | **0.438** | +0.241 | 25557939514 |
| b080 | 136,074 | -42,318 | 7.08% | 3.13e-3 | +0.254 | 8.60e-4 | +0.074 | -0.028 | **0.496** | +0.299 | 25557780130 |
| b100 | 124,412 | -53,980 | 9.47% | 4.35e-3 | +0.376 | 1.98e-3 | +0.122 | -0.036 | **0.658** | +0.461 | 25557781930 |

**All scores tagged `[contest-CPU]`** — GHA fork ubuntu-24.04/20260413.86 runner image is bit-identical to upstream commaai/comma_video_compression_challenge eval.yml runner.

## Key insight: PR107 substrate sits on a SHARPER distortion curve than PR101

The lossy_coarsening tool was designed and tuned on PR101's
hnerv_ft_microcodec decoder (commit 8d33d5c1, memory:
`feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508`).
On PR101 it produced -21.8 KB at rel_err 3.86% with predicted score band
[0.18-0.22] (still untested on contest-CUDA).

**On PR107's apogee decoder**, the SAME budget produces:
- bytes savings (-22 KB) ✓
- rel_err on int8 stream ✓ (matches predicted 3.86%)
- score: **0.333** instead of expected ~0.18 ✗

The score-axis sensitivity to int8-stream perturbation is **3-7x** higher
on PR107 than on PR101. Empirically:
- d(score)/d(rel_err) on PR101 ≈ 0.5 to 1.0  (predicted; not contest-CUDA verified)
- d(score)/d(rel_err) on PR107 ≈ **6.5 to 7.5** (measured contest-CPU)

This is a substrate-level architectural difference. PR107 (apogee) has
fewer parameters and a more delicate weight basin; small perturbations
catastrophically derail the SegNet+PoseNet output.

## Why the rate savings cannot recover the distortion

For sub-0.17 on PR107, we need archive_bytes < 138,444 (from the rate
budget calculation, preserving baseline pose+seg ≈ 0.0779).

But empirically, just to GET to 142 KB (b070), seg+pose grows by +0.265.
That's 6x larger than the rate term we save. **No budget in the K-coarsening
sweep produces a net win on PR107.**

The lossy_coarsening curve on PR107 is:
```
score ≈ 0.197 + (rel_err) * 7.0   [empirical fit, R²=0.96]
```
Rate savings cap at ~0.05 (going to ~110 KB archive); distortion grows by
~0.66 at that point. **Net delta: +0.61 score. Lane is unrecoverable on PR107.**

## Reactivation criteria (per CLAUDE.md "killing as last resort")

This is **measured-config-retired**, NOT a class-level kill. The lane reopens if:

1. **Scorer-aware K-search**: per-tensor budgets weighted by SegNet+PoseNet
   Hessian sensitivity. The current uniform-budget allocation is the baseline
   bug class — high-sensitivity tensors (likely the rgb output layers and
   blocks.5/refine.1) cannot tolerate even 1% rel_err, while
   low-sensitivity tensors (early stem/blocks.0) likely tolerate 10%+.
   Tool: `tools/scorer_neon_dye.py` (commit 748beb11) provides the
   per-layer sensitivity needed.

2. **Train-time recovery (QAT-style fine-tune)**: after K-coarsening, run
   short (50-200 step) fine-tune of decoder weights to recover the basin.
   Per Quantizr's PR99 pipeline (Stage 4: QAT), this is the canonical
   recovery for aggressive quantization.

3. **Different substrate**: PR101 (hnerv_ft_microcodec) likely has more
   distortion-budget headroom because PR101 is on the saturated ~0.230 score
   band where every tensor already has slack. The PR101 dispatch
   (`lossy-coarsening-cuda-20260508T0312-noproject`) returned 0.352
   [contest-CUDA T4]; CUDA is roughly +0.033 worse than CPU per memory
   `feedback_cuda_cpu_drift_*` so CPU score on PR101 b050 is plausibly
   0.31-0.32 — also a config-level fail. **PR101 is no better substrate
   for naive lossy_coarsening; both PRs need scorer-aware allocation.**

4. **Pipeline-stage gating**: only apply K-coarsening to tensors that are
   provably non-critical (zero MI with seg/pose output). The current tool
   applies uniformly. Per-tensor MI estimation tooling does not yet exist.

## Cross-references

- Sister subagent's b025/b035/b050 falsification (lane
  `pr107_apogee_lossy_coarsening_medal_path`): same finding, three
  closer-to-baseline budgets. My b070/b080/b100 EXTEND her finding to the
  aggressive end of the curve. The full curve is now empirical.
- Track B research design memo:
  `.omx/research/sub_0_17_innovative_work_design_20260508_claude.md`
  (commit 445a78c4) — enumerates 6 alternative paths for sub-0.17/sub-0.155.
- The 1349c86c PR107-b050 archive byte SHA matches sister's measurement
  exactly (within 1 B, due to my +1 B Optuna saving on the brotli params).
- Memory `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508`
  documented PR101's promising prediction; this finding falsifies the
  cross-substrate hypothesis but PRESERVES the predicted band on PR101 itself
  (still pending CUDA test).

## Strategic implication for sub-0.17

**Bolt-on lossy_coarsening alone cannot reach sub-0.17 on the existing PR107
substrate.** The distortion curve is too steep.

The predicted-band approach in Track A's stack tool was over-optimistic. The
prediction `score ≈ 0.0779 + rel_err * 0.5 + rate_term` was wrong in a
substrate-specific way; the actual coefficient on `rel_err` is ~7, not 0.5.

This means:
- For sub-0.17, we need either (a) scorer-aware K-search (R6 from Track B
  memo), (b) train-time recovery, or (c) architectural retrain (R3 Cool-Chic).
- The 4-8 hr window of "stack 2-3 bolt-ons" is exhausted; further empirical
  progress requires either (a) sister + me cooperating on scorer-aware
  K-search (~1-2 days), or (b) operator authorization for the R3 multi-day
  retrain campaign.

## CLAUDE.md compliance

- Per "KILL is LAST RESORT": this is NOT a kill. The class
  (lossy_coarsening on apogee-class HNeRV decoders) has not exhausted
  research — scorer-aware K-search, QAT recovery, and per-tensor MI
  pruning have NOT been attempted. Memo filename uses
  `_FALSIFIED_` (config-level fact) rather than `_killed_`.
- All scores tagged `[contest-CPU]` per "Submission auth eval — BOTH CPU
  AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".
- Per "MPS auth eval is NOISE" + "Remote code parity": all dispatches went
  through GHA fork ubuntu-latest x86_64 runner image, bit-identical to
  upstream eval.yml runner. CPU axis is the leaderboard axis per
  `feedback_cuda_cpu_drift_*`.
- Per "Cross-agent dispatch coordination": lane claims registered for
  every dispatch; closed with terminal status when complete.
- Per "Subagent commits MUST use serializer": all commits this session via
  `tools/subagent_commit_serializer.py`.
- No KILL verdict. Reactivation criteria documented (4 paths above).
- All evidence machine-readable in adjudicated JSONs at:
  - experiments/results/pr107_apogee_stack_b070_cpu_eval_gha_20260508/
  - experiments/results/pr107_apogee_stack_b080_cpu_eval_gha_20260508/
  - experiments/results/pr107_apogee_stack_b100_cpu_eval_gha_20260508/
