# Pose — the OPTIMAL allocation, derived (not laddered)

**Date:** 2026-08-01 · **Axis:** `[macOS-CPU advisory]` · `score_claim=false` · `promotable=false`
**Cost:** $0, scorer-free (statistics of a banked GT array; no scorer slot)
**Task:** #850 · **Pointer:** v4d 0.9639878 UNMOVED — this is MEANS.
**Source:** `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` key `gt_poses`, shape (600, 6).

---

## §0 WHY THIS FILE EXISTS — a self-correction

An earlier pass in this session laddered **generic** storage formats (fp16 / int8 / int16 at an
arbitrary 8σ range) and reported the winner. That is the generic-basis reflex
(`[[generic_basis_metric_never_optimal_cosine_fourier_euclid_20260729]]`): a control, not an answer.
Operator binding 2026-08-01: *"Make sure all implementations of these new techniques are fully
optimal and informed. No naive or toy."*

For **MSE distortion over parallel sources the optimal allocation is not a format choice at all.**
It is **reverse water-filling** (Shannon), and it is exact. This file derives it.

---

## §1 MODEL VALIDITY — checked BEFORE applying the formula

| dim | var | skew | excess kurtosis | note |
|---|---:|---:|---:|---|
| 0 | 1.57839 | +0.073 | **+2.421** | heavy-tailed |
| 1 | 1.27793e-3 | −0.578 | −0.084 | ~Gaussian |
| 2 | 8.93247e-4 | −0.059 | −0.092 | ~Gaussian |
| 3 | 9.15146e-5 | +0.322 | **+1.910** | heavy-tailed |
| 4 | 5.46788e-5 | +0.482 | **+2.061** | heavy-tailed |
| 5 | 8.19250e-4 | −0.632 | −0.231 | ~Gaussian |

**Consequence: Gaussian R(D) is an UPPER BOUND here, not an estimate.** Three of six dims are
leptokurtic, and the Gaussian maximises differential entropy at fixed variance — so a real
entropy coder **beats every byte figure below**. Reporting them as achievable-or-better is honest;
reporting them as "the cost" would overstate.

**Correlation — the dims are NOT independent.** max |off-diagonal ρ| = **0.5291** (dims 1↔5).
Covariance eigenvalues `[1.581038, 1.7196e-3, 9.1503e-4, 3.7846e-4, 7.3770e-5, 4.2498e-5]`.

> **KLT gain = (Π var / Π eig)^(1/6) = 1.165176.** Decorrelating first (rotate to the eigenbasis)
> buys **16.5%** before a single bit is allocated. Water-filling on RAW dims is therefore itself
> sub-optimal. Sister: task #140 (low-rank pose codec, 2.7× byte cut) is the same family and must
> be recalled before any build — do not re-derive it.

---

## §2 THE DERIVED OPTIMUM

Reverse water-filling: `D_i = min(θ, σ_i²)` , `R_i = ½·log₂(σ_i²/θ)` when `σ_i² > θ`, else 0.
θ solved so that `(1/6)·Σ D_i = target d_pose`. All six dims exceed θ at every target below, so all
receive bits (no dim is discarded).

| target d_pose | contribution √(10·d) | θ | bits/dim (b0…b5) | bits/pair | **bytes / 600 pairs** | rate S |
|---:|---:|---:|---|---:|---:|---:|
| **2.353e-5** *(PR130 demonstrated)* | 0.015341 | 2.3530e-5 | 8.02 2.88 2.62 0.98 0.61 2.56 | 17.67 | **1,325** | 0.00088 |
| 1.0e-5 | 0.010000 | 1.0000e-5 | 8.63 3.50 3.24 1.60 1.23 3.18 | 21.37 | 1,603 | 0.00107 |
| 1.0e-6 | 0.003162 | 1.0000e-6 | 10.30 5.16 4.90 3.26 2.89 4.84 | 31.34 | 2,350 | 0.00157 |
| 1.0e-7 | 0.001000 | 1.0000e-7 | 11.96 6.82 6.56 4.92 4.55 6.50 | 41.31 | 3,098 | 0.00206 |

The allocation is sharply anisotropic — **8.02 bits to dim0, 0.61 to dim4** — which is the
k-ladder done exactly instead of by threshold. Uniform-width storage is the same error as the
solver's uniform 6-equation weighting, in a different clothing.

---

## §3 THE HEADLINE — pose rate is SETTLED and nearly free

At PR130's demonstrated d_pose the entire counted pose stream is **≈1,325 B = 0.00088 S** (and less,
per §1's heavy tails). PR130 ships ~23 KB for the same distortion — **~17× more than the bound.**

| | S |
|---|---:|
| distortion available on pose (live 0.292941 → 0.015341) | **0.277600** |
| optimal rate cost to buy it | **0.000880** |
| **ratio** | **≈ 315×** |

> **The pose bottleneck is 100% REALIZATION and ~0% RATE.** Bytes are not, and were never, the
> binding constraint on this axis. Every remaining pose question is whether the render can be made
> to *hit* the values — which is exactly the photometric wall CLAUDE.md's 2026-07-10 clarification
> names, and which the reformulation makes qualitatively easier (one dominant coordinate to realize
> accurately, not six equal targets).

---

## §4 WHAT AN OPTIMAL-FORM IMPLEMENTATION MUST DO (binding on any build)

1. **DO NOT KLT-rotate at this scale — MEASURED, and it was my own error to prescribe it.**
   The first draft of this section said "rotate, 16.5%, free." **Measured 2026-08-01: the rotation
   LOSES.** Water-filling on the eigen-spectrum saves **50 B of payload** (17.67 → 17.00 bits/pair
   at the PR130 target) but the 6×6 fp32 basis is video-derived ⇒ **COUNTED at 144 B**. Net
   **−94 B** (1,419 B rotated vs 1,325 B axis-aligned). Same −50 B/+144 B at every target tested,
   so the rotation loses by MORE in relative terms as the target loosens.
   *Why:* 600 pairs × ~17 bits is far too small a denominator to amortise a fixed basis.
   **Axis-aligned here is therefore a DERIVED result with a receipt, not a generic default** — which
   is what `[[generic_basis_metric_never_optimal_cosine_fourier_euclid_20260729]]` demands. The rule
   forbids *assuming* a basis; it does not forbid *reporting* the axis-aligned one after measuring
   its rival. Recall #140 (low-rank pose codec, 2.7×) before revisiting — and note it claimed its
   gain on a DIFFERENT stream geometry, so its ratio does not transfer here unexamined.
   *Reactivation:* if the pose stream ever grows past ~10× its current size, or the basis becomes
   derivable-at-decode (⇒ 0 counted bytes, rule-118 free), re-measure — the payload gain is real.
2. **Allocate by reverse water-filling at the target θ** — never a uniform width, never a
   hand-picked σ-range.
3. **Entropy-code the residuals.** §1 says the marginals are leptokurtic, so a real coder beats the
   Gaussian bound; a fixed-width field discards that for nothing.
4. **Quantization-aware solve.** The solve must target the *representable* lattice point under the
   §2 allocation, not a continuous optimum that is rounded afterward. Post-hoc rounding adds error
   the solver never saw; solving on the lattice makes bit-allocation and effort-allocation one
   problem. (Note: the existing `amplitude_q8` is a **correction-amplitude scale in 1/256 units**
   — a lattice on the STEP, not on the stored pose values. Do not conflate them.)
5. **Per-dim effort in the solve** mirrors per-dim bits: iterations/trust-region are the same
   scarce resource, allocated against the same 29,000× variance spread.

---

## §5 HONEST SCOPE

- **DERIVED, not measured-through-R.** These are floors from GT target statistics. They bound what
  storage can cost; they say nothing about what the render achieves. Live d_pose is 0.00858144 —
  three orders above every floor here.
- Gaussian R(D) with heavy tails = **upper bound** (§1). Real-coder races (r7 SMEVR / Brotli /
  LZMA1) are owed before any byte figure is quoted as achieved.
- **KLT gain 1.165 is measured on THIS 600×6 array**; a rotation fit on the targets is itself
  video-derived and must be counted, which §4.1 does.
- Single source, single axis, advisory. No noise floor (population statistics of a fixed array, so
  sampling noise is nil — but the array is one clip).
- **Rate being settled does NOT make pose solved.** It relocates the entire question to realization.

Sisters: `ddm_gc16_upstream_score_lowering_convocation_20260731.md` (already measured ≥93.06% of
d_pose is dim0, and self-refuted it as an unconditional bound; fp16-ULP floor at dim0's magnitude;
QA65 `pose_dim0_offset` = 31.515625) · task #140 · task #850 · CLAUDE.md §"Pose is SOLVED"
2026-07-10 CLARIFICATION (post-hoc storage is measured dead on the witness — the wall is photometric).
