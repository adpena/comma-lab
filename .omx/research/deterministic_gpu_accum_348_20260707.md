# Task #348 — DETERMINISTIC GPU ACCUMULATION: the L70 wall is ONE op class, and the cure was already in-tree (2026-07-07)

**Verdict: GO.** The full witness trainer is now **cross-process BIT-IDENTICAL on MLX-GPU**
(0/28 tensors diverged, N=10 separate processes; Muon-finisher arm 0/28, N=5) by enabling the
in-tree `--fused-r-kernel` (#252) — whose Metal transpose-VJP is exactly the task's
"fixed-order reduction" attack (no atomics), already built and parity-tested. Overhead is
**NEGATIVE**: ~8% FASTER (25.35s → 23.44s, 200-epoch n=1 timing smoke). No new fixed-point
int64 atomic kernel was needed (phase-2a not built — 2b already existed; building 2a would
have duplicated a solved mechanism).

**Authority:** every verdict here is a `[macOS-MLX research-signal]` bit-identity FACT
(sha256 hash equality of fp32 bytes), never a score. CPU (numpy-fp32 / MLX-CPU) remains the
score/verdict authority. What this changes: bit-exact PROOFS (crash-resume, byte-close
round-trip, A/B trunk-identity) may now run on GPU **iff the graph is atomic-scatter-free**
(operationally: fused-R ON) — a large wall-clock lever for parity work.

## STORES CONSULTED (proactive recall)

- memory `mlx_gpu_not_bit_identical_crossprocess_bitexact_proof_cpu_locked_20260702` (the L70
  source: witness forward, 2 processes, 28/28 diverged; no per-op localization existed).
- `.omx/research/n205_full_run_risk_register_watchlist_20260702.md` D6 (named the missing
  equation `mlx_gpu_crossprocess_nondeterminism_v1` — registered by this task).
- `.omx/research/arch_override_fp32_exact_gpu_training_scorer_20260611.md` +
  `mlx_scorer_port_drift_audit_20260611.md` (NAX kernel-selection class — ruled OUT here:
  divergence persists under `MLX_METAL_GPU_ARCH=applegpu_g15`).
- `src/tac/local_acceleration/metal_fused_r_operator.py` (the P2b fused VJP docstring already
  said "Deterministic (no atomics) … the prior mx.vjp of the pure-MLX oracle … carried a
  ~1-ULP scatter non-determinism floor" — this task measured that floor as the WHOLE 28/28
  wall and verified the cure end-to-end).
- `experiments/tests/test_levelset_crash_resume_smoke.py` (the CPU-locked resume-proof
  runner + the cell recipe this task's harness replicates).

## Localization table (M5 Max, Metal 4, MLX 0.31.2; N=10 separate processes per cell; sha256 of fp32 output bytes)

| op class | in-process repeat | cross-process | verdict |
|---|---|---|---|
| seeded random / elementwise | identical | identical | deterministic |
| matmul 1024² / huge-K (32×2^18)·(2^18×32) / GEMV 2^20 | identical | identical | deterministic (no split-K nondet) |
| sum 2^24 / mean-axis 4096² / softmax 4096² | identical | identical | deterministic |
| conv2d s2 / grouped s2 | identical | identical | deterministic |
| custom grouped-backward Metal kernel (17× lever) | identical | identical | deterministic |
| MLP fwd+bwd (GEMM VJPs) / FiLM+code-select grads | identical | identical | deterministic |
| take-grad (dup idx, trivial cotangent) | identical | identical | deterministic (special case) |
| **`arr.at[idx].add` dup-index scatter** | **DIVERGES** | **10 unique/10** | **NONDET (atomics)** |
| **take-VJP with strided cotangent (T2/T3 differential)** | — | **unique/proc** | **NONDET** |
| **reference-R bicubic-UP backward (`_resize_axis_nhwc`)** | — | **unique/proc** | **NONDET — THE witness poison** |
| reference-R bilinear-DOWN backward (874→384) | identical | identical | deterministic (low fan-in) |
| fused-R Metal forward + fixed-order transpose VJP | identical | identical | **deterministic (the cure)** |

Differential detail: take→sum-sq grad is deterministic (fused special case), but the SAME
gather with the S5 pattern (reshape→×weights→sum-taps) diverges even with CONSTANT
host-materialized indices → the take-VJP lowers to atomic scatter-add whenever the cotangent
is non-trivially strided. Consistent with `.at[].add` being nondeterministic outright.

## Witness-trainer composite cells (real launch-path trainer, n=1 pair, 96×128, 5ep CE/tau/l7, seed 0)

| cell | N | diverged tensors (liveP+emaP) |
|---|---|---|
| GPU, reference R (kernel env on OR off) | 3 | **28/28** (reproduces L70; onset at ep1 — ep0 verdict identical) |
| GPU, reference R, `MLX_METAL_GPU_ARCH=applegpu_g15` | 3 | 28/28 (NOT the NAX class) |
| CPU | 3 | 0/28 (CPU-locked discipline confirmed) |
| **GPU, `--fused-r-kernel`** | **10** | **0/28** |
| **GPU, `--fused-r-kernel` + `--muon-start-epoch 3`** | **5** | **0/28** |

Bisect chain that got there: op probes all-clean → full trainer diverges → per-stage ckpts
show ep1 onset → piecewise step probe (fwd_rgb/R_out/loss identical, grads diverge) →
A/B/C/D/E variants (no-R deterministic; R-backward alone nondet) → R stage bisect (UP
backward nondet, DOWN deterministic) → T1–T4 differential (atomic scatter-add mechanism).

## Overhead + fidelity

- 200-ep n=1 96×128 timing smoke: reference 25.35s vs fused-R 23.44s → **fused is ~1.08×
  faster** (R backward is matmul-form instead of scatter). Determinism costs LESS than zero.
- Fidelity: `src/tac/tests/test_metal_fused_r_operator.py` — 25/25 pass (forward bit-identity
  at real 874×1164→384×512 vs the numpy authority + fused-vs-non-fused VJP parity). No new
  fixed-point quantization introduced → no overflow/saturation surface (the int64 plan's
  saturation counter is N/A).

## What is now deterministic / what remains

- **Deterministic (measured):** the ENTIRE 5-ep witness smoke config through checkpoints —
  MLP forward, fused-R, CE/tau/l7 losses, AdamW, EMA, Muon finisher.
- **Remaining unverified at composite scale:** n600 + self-orient ON (self-orient recompute
  is numpy-side → expected neutral), annulus/nucleus telemetry paths (torch-CPU → neutral),
  any future lever that introduces `.at[].add`/gather-VJP into the GRAPH (the probe + tests
  are the guard). n600 composite re-verification owed before relying on GPU bit-identity
  there (cheap: 2× short resumed segments, hash compare).
- **Discipline update:** "bit-exact proofs CPU-locked" relaxes to "bit-exact proofs CPU-locked
  OR GPU-with-fused-R (atomic-scatter-free graph), re-proven per config by the probe."

## Landed artifacts

- `tools/mlx_gpu_determinism_probe.py` — the reusable localization instrument (19 op cells,
  `--child` self-spawning, N configurable).
- `src/tac/tests/test_mlx_gpu_determinism.py` — 5 tests: positive guarantees (fused-R VJP +
  fwd + GEMM cross-process identity) + non-flaky mechanism documentation (warn-on-news).
- `src/tac/canonical_equations/mlx_gpu_crossprocess_determinism_20260707.py` — registers
  `mlx_gpu_crossprocess_nondeterminism_v1` (closes risk-register D6's named equations-leg gap;
  2 VERIFIED anchors).
- **P0 trainer crash fix (discovered en route):** the 2026-07-07 `per_class` verdict telemetry
  (dict-valued key) crashed EVERY fresh launch at the baseline_v0 verdict print
  (`round(dict)` TypeError) and every sync in-loop verdict — 2 sites in
  `experiments/train_levelset_witness_realized_through_R_mlx.py` fixed (pop mirroring
  `_emit_verdict_row` + additive row-only `d_seg_by_class`/`flip_share_by_class` fields).
  Verified by ~30 real trainer executions in this session (CPU + GPU, incl. Muon arm).

## Triality note

- **DSL leg:** no NEW lever — the determinism cure IS the existing `--fused-r-kernel` #252
  lever (already DSL-held). Its activation-ledger value gains a second dimension: it is now
  the GPU-bit-identity enabler, not just a throughput lever (duty-to-measure note for the
  costate queue: fused-R ON is the recommended default for any run whose resume/byte-close
  proof wants GPU speed).
- **equations leg:** `mlx_gpu_crossprocess_nondeterminism_v1` registered (D6 gap closed).
- **DAG leg:** this ledger is the FEED row source; memory L70 updated in the same landing.

## Reactivation criteria

- Any MLX version bump → re-run `tools/mlx_gpu_determinism_probe.py` (the mechanism tests
  warn if scatter becomes deterministic — then the CPU-lock can relax further).
- Before relying on GPU bit-identity at n600/self-orient-ON: run the 2-process composite
  check at that config.
- If a future witness lever adds scatter/gather-VJP ops to the training graph: probe first.

Pointer contest-CPU 0.19110 UNMOVED — this is apparatus/means (a wall-clock lever for the
proof paths), not a score row.
