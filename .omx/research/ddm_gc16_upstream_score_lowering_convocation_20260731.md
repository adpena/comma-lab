---
title: "gc16 — what the upstream decomposition is WORTH: pursue-rows from the frozen weights"
utc: 2026-07-31
lane_id: lane_ddm_gc16_upstream_score_lowering_convocation_20260731
convocation: 18th (operator-convened) · Schmidhuber LEAD + pantheon + pantheon-of-pantheons
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
evidence_axis: "[macOS-CPU advisory]"
pointer: 0.1910828242 [contest-CPU] UNMOVED
pointer_moved: false
upstream_mutated: false
machine_readable: .omx/research/ddm_gc16_headchain_spectrum_20260731.json
---

# gc16 — "I didn't have you do that just for fun"

Input: `ddm_ua1` (weights as files), `ddm_ua2` (defenses + budget), `ddm_ua3` (submissions +
runtime closure), MAIN's read of `modules.py` / `evaluate.py` / `frame_utils.py`. This memo does
**not** re-derive them. It adds four exact weight-level computations they did not do, controls
them, reconciles them against what the campaign already banked, and converts the survivors into
**pursue-rows with ΔS arithmetic**.

**$0.** No forward pass, no scorer slot, no training, no dispatch. `upstream/` read-only; nothing
inside it was created, edited, moved, or deleted.

---

## 0. DENOMINATOR, and the correction that reframes the whole charter

**Search scope for every "did not find" below**, stated once: two async recall sweeps over
`.omx/research/` (full-text `rg` over `*.md`/`*.json`/`*.jsonl`, ~9.6k entries), `.omx/state/`
(`current_focus.md`, `canonical_equations_registry.jsonl`, `operator_p0_ledger.jsonl`,
`canonical_task_status.jsonl`), the memory dir, `src/tac/`, and the byte-closed receipts on
`/Volumes/VertigoDataTier/pact/ddm_{v4c,v4d,gr1,ep2,tt1}_*`. One sweep (pose axis) was still
running at write time; its result is **owed** and named in §8.

**Computation denominator:** 4 scripts, all pure linear algebra on `upstream/models/*.safetensors`
+ the cached `gt_n600.npz` / `gt_n96.npz`. 200-draw random-matrix control on the head chain.
12 of 96 frames (12,208,032 px × 3) for the GT histogram. No result here uses a sampled subset
where an n600 quantity was available.

### The charter's axis weights are from a superseded coordinate — CORRECTED

The charter says "pose ~1.24 S · seg 0.431 · rate 0.239". **Re-derived from the primary
artifacts**, the live own-vehicle coordinate is **v4d, 2026-07-31**:

| axis | contribution | quantity | receipt |
|---|---:|---|---|
| seg `100·d_seg` | **0.431179** | d_seg 0.00431179 | `ddm_v4d_adaptive_hybrid_20260731.md:45` |
| pose `√(10·d̄)` | **0.292939** | d̄_pose **0.008581** | `:46` |
| rate `25·B/37.5M` | **0.239868** | **360,238 B** (not 359,750 — that is v4c) | `:47` |
| **S** | **0.9639878** MEASURED | predicted 0.9639858, residual 1.82e-6 | QA78, real `evaluate.py` n600 |

The 1.24 is a base-coordinate number. Carrying it across is the exact failure the campaign's own
memory names (`Base≠composed — never carry a term across`). **At v4d, pose is the SECOND-largest
axis, not the first.** Ranked by **gap to a demonstrated floor** (which is the only ranking that
routes work):

| axis | ours | demonstrated floor | gap in S | floor's provenance |
|---|---:|---:|---:|---|
| **seg** | 0.431179 | 0.0297 (d_seg 2.966e-4) | **0.4015** | PR130, external, lessons-only |
| **pose** | 0.292939 | 0.0153 (d_pose 2.33e-5) | **0.2776** | PR130, external, lessons-only |
| **rate** | 0.239868 | 0.1272 (191,052 B) | **0.1126** | PR130, external, lessons-only |
| | 0.9639878 | 0.172141 | 0.7917 | sums exactly to the PR130 row ✓ |

**Where the frozen-weight substrate pays is pose (#2), and its main payoff is that it UNLOCKS
seg (#1).** That is the routing this memo argues for, and §7/P1 is the mechanism.

### The rate correction, absorbed

Rate = `archive.zip` bytes, full stop (`evaluate.py:63`). `inflate.py`, `inflate.sh`, the runtime
tree, declared dependencies, the installed closure and decode time cost **ZERO rate**; there is no
time term (`:92`). The `uv sync --group cu128` 3,190,398,780 B vs `--group cpu` ~78 MB asymmetry
is re-filed as a **WALL-CLOCK** coordinate only (§6), and per the standing directive it is not a
discount criterion there either. The absence of brotli/zstandard/constriction from the contest
runtime (ua3, direct import probe) is a **bootstrap-reliability** question, never a rate one.
**Consequence: the free-interpreter doctrine is undamaged and has measured headroom.**

---

## 1. SURFACE — *pose observability as a function of depth*

**Coordinates.** Depth in the PoseNet scored path; measure = spectral effective rank
`exp(H(σ²))` of the 6×n Jacobian of the **scored** outputs, and the share of an isotropic
input perturbation's d_pose landing in each output dim.

**MEASURED, exact under all-ReLU-active.** Two source facts make this exact rather than
approximate: (a) `AllNorm` is `BatchNorm1d(1)` over `x.view(-1,1)` — in eval mode it is a
**scalar** affine map `x ↦ ax+b` (`modules.py:29-33`); (b) `final_layer.pose` is `nn.Linear`,
so `Δp = W_scored·Δh` is **exact**, not first-order. With every ReLU active the whole chain
2048→512→32→6 is affine, and any inactive ReLU can only *lower* the rank — so these are
**upper bounds on rank / concentration**.

| level | eff-rank /6 | dim0 share | cond | σ₁²/σ₆² |
|---|---:|---:|---:|---:|
| `h32` (final layer only — **ua1's coordinate, replicated exactly**) | 1.1665 | 97.363% | 24.82 | 615.9 |
| `z32` (through `res_layer`) | 1.0842 | 98.687% | 43.24 | 1,870 |
| `x512` (hydra input) | 1.0717 | 98.891% | 55.68 | 3,100 |
| `s512` (pre-`hydra.resblock`) | 1.0656 | 98.983% | 57.79 | 3,339 |
| **`v2048` (vision output, full summarizer+hydra)** | **1.0130** | **99.706%** | **547.0** | **299,224** |

**DEGENERATE-BASELINE CONTROL (required, and it is the point of this section).** A product of
5 matrices concentrates *by itself* (multiplicative ergodic theorem) — so the trained numbers
mean nothing without this. 200 draws, **same shapes, same per-matrix Frobenius norm**, iid
gaussian, identical chain:

| statistic | trained | control median | control p95 | control max |
|---|---:|---:|---:|---:|
| eff-rank /6 | **1.0130** | 5.2345 | 5.5145 | 5.7121 |
| dim0 share | **99.706%** | 23.054% | 29.717% | 37.085% |
| cond | **547.0** | 2.2752 | 3.0872 | 3.8535 |
| σ₁²/σ₆² | **299,224** | 5.1764 | 9.531 | 14.85 |
| argmax dim | **0** | uniform: 44/23/45/24/30/34 across dims 0–5 | | |

Trained cond is **142× the control's maximum**; trained gain-ratio is **20,150× the control's
maximum**; and the control's dominant dim is uniformly distributed while the trained one is
dim 0. **The concentration is trained structure, not a random-product artifact.**

**Where the language flips.** eff-rank ≈ 6 ⇒ "pose is a 6-DOF solve." eff-rank ≈ 1 ⇒ "pose is a
scalar servo with five nuisance coordinates." At every measured depth we are on the second side,
and it gets *stronger* toward the input.

**What moves this level set.** Nothing we control (frozen file). What moves our *reading*: the
FastViT nonlinearity below the 2048-dim vision output, which is **not** in this linearization.
`ddm_pi2_posenet_inversion_20260730` measured the concentration **at the real input on 53 real
pairs** and reports "p0 dominates ~50×" (≈91% share). **pi2 is the authority; my 99.7% is a
head-chain-only upper bound.** The two are consistent in direction and the gap is itself the
finding: **FastViT's nonlinearity de-concentrates the response by roughly 4× relative to the
linear head chain.**

**Which way it falls.** Toward "pose is a 1-D observable." **Prior-work position:** pi2 (07-30)
already has the dominance and the 24.8 condition number; ua1 (today) re-derived 24.8
independently. **New here: the depth profile and the control.**

---

## 2. SURFACE — *where OUR d_pose lives* (a bound that needs no sensitivity claim)

**Coordinates.** The share of our measured `d̄_pose = 0.008581` attributable to output dim 0.

`compute_distortion` is a **raw** MSE over the six outputs (`modules.py:84`). It does not
standardize. The GT targets (`gt_n600.npz::gt_poses`, 600×6, MEASURED here):

| | dim0 | dim1 | dim2 | dim3 | dim4 | dim5 |
|---|---:|---:|---:|---:|---:|---:|
| mean | **31.2606** | −0.01884 | −0.00900 | 0.00164 | 0.00159 | 0.00083 |
| std | **1.25634** | 0.03575 | 0.02989 | 0.00957 | 0.00739 | 0.02862 |
| RMS | **31.2859** | 0.04041 | 0.03121 | 0.00971 | 0.00756 | 0.02863 |

**DERIVED, CONDITIONAL — and the condition is MEASURED-VIOLATED on a predecessor base, which is
the finding.** If our predicted dims 1–5 lie within the GT range, their worst contribution is
`(1/6)·Σ_{i≥1} RMS_i² = 5.96e-4`, giving

> **≥ 93.06% of our d_pose would be a dim-0 error of |Δp₀| ≥ 0.2189** — 0.700% of dim0's mean,
> 17.4% of its std — **CONDITIONAL on `|pred_i| ≲ max|GT_i|` for i ≥ 1.**

**I attacked this bound and it does not hold unconditionally.** Nothing forces the renderer's
pose output to stay inside the GT range, and `ddm_pfs1_d2_price_receipt_20260729.json` MEASURES a
violation on the warp base: `e_p_per_dim_std = [0.8195, 0.1170, 0.2379, 0.0096, 0.0075, 0.0286]`.
Dims 1 and 2 have residual std **3.3× and 8.0× their own GT std** (0.0358, 0.0299) and together
contribute `(1/6)(0.117² + 0.238²) = 0.0117` — **20× my assumed dims-1–5 ceiling**. On that base
the solve *overshot* the small dims. (Same receipt: `e_p_delta_svd_energy_frac[0] = 0.9059`, so
dim0 still dominates there — but not by the margin my bound assumed.)

**Which makes the routing measurement obvious, and it has not been done.** Recall's exhaustive
sweep: *"did not find in scope an artifact that decomposes the current residual
`d̄_pose = 0.00858145` into its six per-dim MSE components."* Every per-dim attribution in custody
is on a **predecessor base** (pfs1 warp, sc1 painted, pi2's 53 gt_n96 pairs) or is a
storage/Jacobian *sensitivity*, never the realized residual of the live carrier.

> **OWED, and it routes the whole axis: the six-way per-dim MSE breakout of v4d's 0.00858145.**
> It is ~free (six subtractions on arrays the composed co-measure harness already computes) and it
> decides between two completely different programs — "drive one scalar" vs "stop the solve from
> overshooting dims 1–2." See P0.

**And the residual is not spread over 600 pairs.** MEASURED (`ddm_v4d_adaptive_hybrid_20260731.md`
§4): **top-17 pairs = 74.3% of total d̄ mass**; top-50 = 85.9%; top-100 = 91.1%; **median d =
0.00088 vs mean 0.00858** (9.8× ratio). Tao's QA48 17-pair hard core survived every rung intact.
So the pose axis is **one scalar-ish observable × ~17 pairs**, not a 600-pair field problem.

**Level sets on this surface (the servo's setpoints):**

| |Δp₀| | as % of mean | d_pose | contribution |
|---:|---:|---:|---:|
| 0.2189 (ours today) | 0.700% | 0.008581 | **0.292939** |
| 0.0983 | 0.314% | 0.001610 | 0.126886 ← banked R1 `dxi`, **different lineage** (r6cal/witness, 7.2 KB, from a STOPPED run still descending at −1.26%/ep); NOT the v4c/v4d carrier |
| 0.0245 | 0.078% | 1.0e-4 | 0.031623 |
| 0.00775 | 0.0248% | 1.0e-5 | 0.010000 |
| — | — | 2.33e-5 | 0.015264 ← PR130 existence proof |
| — | — | 9.3e-10 | 0.000096 ← `pose_plane_proximity_corollary_v1`, ZERO pose bytes |

**The named floor of the cheapest strategy.** "Nail dim0 exactly, let dims 1–5 fall to zero" gives
`d_pose = 5.96e-4 → contribution 0.0772`. **ΔS available on that strategy alone = 0.292939 −
0.077201 = −0.2157.** Against PR130's demonstrated 0.0153 it is **−0.2777**.

**INFERRED, labelled** (upstream names no units): dim0 is forward speed in m/s — 31.26 m/s =
70 mph, a highway segment, consistent with the comma2k19 segment named in
`public_test_segments.txt`; dims 3–5 are ~0.01 rad/s rates. Supported by magnitude, sign, range,
and pi2's independent "p0 (fwd trans)" label. **Not MEASURED.**

---

## 3. SURFACE — *rank-1 as a CODING premise vs as an OBJECTIVE geometry* (a unification)

Two banked results look like they contradict each other. They do not, and reconciling them
produces the design rule.

- **The oldest anchor is `project_posenet_rank1_discovery.md`, ~2026-04-24** — *"PoseNet Jacobian
  rank 1.008. GT pose dim 0 captures 99.80% of total variance (mean 31.26, std 1.26)."* My
  centred-SVD dim0 share of **99.8025%** (mean 31.2606, std 1.25634) **reproduces a 98-day-old
  measurement to four digits.** Re-confirmed by **`da1`/`ar1` (07-28)**: `t_p` rank-1 at **0.998**
  SVD energy, ξ-smooth ⇒ pose field ~2 KB. **None of this is new here; it is a control on my
  pipeline.**
- **`QA50` rider §7 / `QA61` dz-carrier:** *standardized* per-dim SVD energy is
  `[0.221, 0.180, 0.171, 0.158, 0.139, 0.130]` = **FLAT/isotropic** ⇒ "the rank-1 premise is a
  scale artifact, **CLOSED**."

**Both are correct, on different surfaces.** In *standardized* coordinates the six dims carry
near-equal independent information — so you **cannot compress the field as rank-1**, and QA61's
closure stands untouched. In the *contest's raw* coordinates the objective weights all six
equally in **absolute** error, and dim0's scale is ~10³× the others — so you **do not need to
represent dims 1–5 accurately at all**.

> **The design rule that falls out: not rank-1 CODING — asymmetric PRECISION.** Bits should be
> allocated proportional to a dim's contribution to raw MSE, i.e. to its *absolute* quantum, not
> to its relative precision.

**Prior work already implements exactly this, and it is SOUND.** `f16` gives *relative* precision,
so on dim0 (~31.26) its ULP is 2⁴·2⁻¹⁰ = **0.015625** (quantization RMS 0.00451, d_pose floor
3.4e-6, S floor 0.0058) while on dim5 (~0.029) its ULP is ~1.4e-5 — a ~10³× misallocation against
a raw-MSE objective. **QA65's `pose_dim0_offset` (one manifest float, 31.515625; dim0 stores the
f16 *residual*) buys 19.3× finer effective quantum at zero counted bytes** and drives dim0's
quantization S floor to ~3e-4. That is the correct fix, already shipped in `pose_warp.stp`.

**Which way it falls.** The pose *stream* is not the problem: it is 8,621 B (2.4% of the archive)
and its dominant error term is already below 1e-3 S. **Our 0.008581 is a REALIZATION error, not a
storage error** — which agrees with the campaign's own five-times-re-proven crux.

---

## 4. SURFACE — *the cost of pose-auditing a correction* (the new, closed-form ladder)

**The problem this addresses is already binding.** `cb1` MEASURED **+22.7 d_pose from ONE Lane
repaint** (contribution +15.07), and the campaign's rule is that "every correction is pose-audited
through the composed co-measure harness"; `j11`'s pose-null / seg-null split is the built
machinery. **What was missing was the ladder: how many modes must you null, and what does each
buy?** It is now exact.

σ² of the scored Jacobian at the vision-2048 level, and the residual after nulling the top k
right-singular directions:

| k nulled | residual Σσ² | fraction | d_pose reduction | cost |
|---:|---:|---:|---:|---|
| 0 | 35,527.53 | 1.0 | 1× | — |
| **1** | 56.454 | 1.589e-3 | **629.3×** | 1 inner product |
| 2 | 12.227 | 3.442e-4 | 2,905.7× | 2 |
| 3 | 3.4331 | 9.663e-5 | 10,348.4× | 3 |
| 4 | 0.40007 | 1.126e-5 | 88,802.8× | 4 |
| 5 | 0.11854 | 3.337e-6 | 299,700.4× | 5 |
| **6** | **0** | 0 | **exact** | **6 inner products** |

**Where the language flips.** k=6 is *exact* because the scored operator has rank 6 — the pose-null
subspace is 2042 of 2048 dimensions at the vision level, and 26 of 32 at `h32`. The whole ladder
costs six inner products; there is no reason to stop at k<6 except compute, and the k=1 rung
already buys 629×.

**Applied to cb1's measured number** (stated as an arithmetic, not a promise): a rank-1 null turns
+22.7 d_pose into ~0.0361 (contribution 15.07 → **0.601**, a 25× contribution reduction); k=3 turns
it into 2.19e-3 (contribution **0.148**); k=6 is exact **to first order**.

**The honest bound, stated with the number.** The projection is exact in the linearization; a
finite repaint is not confined to it. `pi2` already reports the pose tangent is
"realization-limited (tangent over 3–10×)" — so **assume a 3–10× shortfall against the ladder and
the k=6 rung still lands ~10³–10⁴× better than no projection.** The ladder is a *ranking* of
where to spend, and it is exact as a ranking regardless of the second-order shortfall.

**Which way it falls.** Toward "seg repaints are not pose-vetoed; they are pose-*priced*, and the
price is six inner products." **This is the row that converts the #1-gap axis (seg, 0.4015) from
blocked to open.**

---

## 5. SURFACE — *the PoseNet input map, exactly*

### 5a. The exactly-invisible half

`rgb_to_yuv6` (`frame_utils.py:51-78`) at the scorer plane 384×512: luma is a **lossless**
2×2 polyphase de-interleave (`y00,y10,y01,y11` at 192×256 reconstruct Y exactly); chroma is an
**exact 2×2 box mean** (`·0.25`). Per 2×2 RGB block: 12 DOF in, 6 read (4 luma + 1 U-mean +
1 V-mean).

> **DERIVED, exact: exactly 50% of RGB degrees of freedom at the scorer-resize plane are
> EXACTLY PoseNet-invisible.**

**Prior-work position:** `frozen_scorer_exact_factorization_20260715` §5 already states
"PoseNet is chroma-blind below 2px" and "luma is lossless space-to-depth" — the qualitative fact
is banked and SOUND. What is added is the exact DOF count and the composition below. **Guard:**
that memo's §5/§6.4 claim "chroma plane = the 2-D orthogonal complement of ℓ" carries an
APPEND-ONLY CORRECTION (2026-07-19, #570/#564) — it is FALSE (principal angle 30.279°). My
statement counts the DOF of the yuv6 map and does **not** use that orthogonality, so it survives.

**Composition (all three legs banked, the product is new):** frame_0 is structurally seg-free
(d_seg obligation 8.5e-9); ker(A) for the 874→384 bilinear resize is ~52% (#580: range(A) =
19.33%); yuv6 kills a further exact 50%. ⇒ **frame_0's scored DOF ≤ 0.4833 × 0.50 = 24.2%; ≥75.8%
of frame_0's payload DOF are provably unscored**, and what survives feeds a functional of effective
rank 1.013 (§1) / ~91% dim0 (pi2). **The live vehicle already exploits this**: v4d stores *no*
frame_0 at all (`f0 := a·warp(f1) + b`, photometric rung-B). Credit, not a proposal.

### 5b. The stem is not a temporal differencer — NEW

`vision.stem.0` reads the 12 channels `[f0:y00,y10,y01,y11,U,V | f1:same]`. MEASURED per-input-
channel Frobenius energy share:

| | luma (8 ch) | chroma (4 ch) | frame_0 | frame_1 | `cos(W_f0,W_f1)` | antisym share |
|---|---:|---:|---:|---:|---:|---:|
| `conv_kxk` 3×3 | 73.19% | **26.81%** | 52.19% | 47.81% | **+0.3507** | **32.48%** |
| `conv_scale` 1×1 | 81.34% | **18.66%** | 50.86% | 49.14% | **+0.4971** | **25.15%** |
| trivial baseline | 66.67% | 33.33% | 50% | 50% | 0 | 50% |

Two readings, and only one of them is new:

1. **Chroma is under-weighted but NOT ignored** at the stem (0.80× / 0.56× of trivial). This is
   *consistent with but weaker than* pi2's end-to-end "99.3% luma / 0.7% chroma" — the stem admits
   chroma and the deep network discards it. Reported so nobody re-reads the stem weights as
   contradicting pi2. pi2 is the authority.
2. **NEW: the stem is SUM-dominated, not a differencer.** `cos(W_f0,W_f1)` is **positive** on both
   branches, and the antisymmetric share is **32.5% / 25.1% against a 50% trivial baseline** —
   i.e. the stem is *more symmetric than random*. Ego-motion is not extracted at the stem; it is
   inferred deep. **Hypothesis this names (not a claim):** carrier error that is **common-mode**
   (identical in f0 and f1) may be pose-cheap despite entering the stem at above-random gain,
   because it is motion-neutral. This matters specifically because the live carrier is
   `f0 := a·warp(f1)+b` — a construction whose error is *structurally* common-mode. Cheapest test
   in §7/P3.

---

## 6. SURFACE — *wall-clock*, re-filed (zero rate consequence)

`T_residual(CPU) ∈ [17.4, 22.2] min`, `T_residual(CUDA) ∈ [13.7, 21.7] min` (ua2). Shipped rungs:
numpy-fp64 4-worker **13.9 min bit-exact = 1.25× margin**; torch-fp32 CPU **6.59 min
score-preserving = 2.64× margin**. Required-speedup placement, never "too slow": `N_req = t_s /
T_residual`, closed by a shipped bit-exact rung at ≤3.5×, by a shipped score-preserving rung at
≤7.4×. Metal/MLX rungs are **fenced out** of both axes (absent from both runners).

**The score consequence is a permission, not a cost:** arbitrarily complex deterministic decode
compute is free of rate and has *measured* headroom. **ua3 identifies the under-exploited
coordinate**: every non-trivial shipped `inflate.py` applies a zero-byte unsharp at decode with
α spanning **0.27 → 2.0 (7.4×)** plus one variance-adaptive variant, and **none of them carries a
measurement against the exact scorer.**

---

## 7. SURFACE — *what to PURSUE*, with ΔS arithmetic

Rows are placed by **gap to a demonstrated floor** (§0), then by whether the frozen-weight
substrate actually supplies the mechanism. ORIGINAL-DERIVED = derived here from the frozen weights;
ADAPTED = a banked/external structure re-priced.

### P0 — Break out v4d's 0.00858145 per pose dim, and per pair · the routing measurement, ~FREE
- **Axis.** pose (gap 0.2776) — but its real value is that it **routes P1–P3** and it is the only
  row here that is nearly free.
- **Why it is missing.** Recall's exhaustive sweep found **no artifact decomposing the *current*
  residual per dim**; all custody is on predecessor bases. §2 shows the two hypotheses give
  opposite programs, and `ddm_pfs1_d2_price_receipt_20260729.json` MEASURES a dims-1–2 overshoot
  (residual std 3.3× / 8.0× their GT std) on the warp base — so this is not a formality.
- **What it decides.** (a) If ≥90% of d̄ is dim0 → the axis is a **1-D servo** and P2's
  preconditioning + the QA43 aliasing cure are the program. (b) If dims 1–2 carry ≥20% → the
  solve is **overshooting the small dims**, and the cure is a per-dim trust region in the raw
  (unstandardized) metric, which is a different build.
- **ΔS arithmetic.** None directly. It re-prices −0.2777 of gap between two programs.
- **Cheapest measurement.** Six subtractions on arrays the composed co-measure harness already
  materialises: `mean over pairs of (p_gen[:,i] − t_p[:,i])²` for i=0..5, plus the same table
  restricted to the measured **top-17 pairs (74.3% of mass)**. No new forward pass if the v4d gate
  cached `p_gen`; one n600 pose forward if not.
- **Named falsifier.** None needed — it is a decomposition, not a hypothesis. It cannot fail; it
  can only be uninformative if the six components are equal, which would itself be a finding.
- **Class.** $0 if cached, else one pose-only forward.

### P1 — Pose-audit every seg correction with the exact k-mode null, k up to 6 · ORIGINAL-DERIVED
- **Axis / weight.** Unblocks **seg (gap 0.4015, #1)**; the mechanism lives on pose.
- **Mechanism.** Project each candidate carrier perturbation orthogonal to the top-k right-singular
  directions of the exact scored Jacobian. The 6×2048 chain is closed-form from the frozen weights
  (§1); pulling it to carrier coordinates is one VJP per pair, which the joint-descent engine
  already computes.
- **ΔS arithmetic.** cb1's measured Lane repaint: +22.7 d_pose = +15.07 contribution. k=1 → ~0.601
  (25×); k=3 → ~0.148; k=6 → exact to first order. Apply pi2's measured 3–10× tangent shortfall and
  k=6 still lands 10³–10⁴× better than unprojected. **The ΔS is whatever seg repaint mass the pose
  veto is currently blocking — which I do NOT assert** (no exchange without demonstrated coupling).
- **Cheapest measurement.** Re-run **one** already-measured repaint (cb1's Lane repaint, whose
  unprojected +22.7 is banked) with the k=6 projection inserted, through the existing composed
  co-measure harness. One scorer slot. The banked number is the control.
- **Named falsifier.** If projected d_pose ≥ 22.7/10 (i.e. the realization shortfall exceeds 10×,
  outside pi2's measured 3–10× band), the linearized null does not survive finite repaints and the
  row dies at FORMULATION scope.
- **Class.** scorer-slot (one), $0 to build.

### P2 — Precondition the pose block of the fd1 / family-d GN·CG solve by the exact head metric · ORIGINAL-DERIVED
- **Axis.** Engine capacity — which the campaign's own adjudication names as the binding
  constraint ("sc1's far seed 0.0705 = 464× q1 is an ENGINE-CAPACITY failure, not physics"), and
  the named build explicitly calls for a "scorer-metric preconditioner."
- **Mechanism.** The pose half of that metric is now exact and closed-form: `M = JᵀJ`, rank 6,
  `cond(J) = 547` at vision-2048. Unpreconditioned CG on the normal equations converges at
  `(κ−1)/(κ+1)` with `κ = cond² = 299,224` → a per-iteration factor of `1 − 6.7e-6`. **Effectively
  never.** A 6×6 whitening by `M^{-1/2}` (free — the SVD is of a 6×2048 matrix) makes the pose
  block κ=1.
- **It is not only a speedup — it is the cure for a MEASURED pathology.** `ddm_qa43…` names
  **rotation/translation aliasing**: *"a single-plane homography makes the solver substitute speed
  for turn,"* and the tail's within-chart correction is **98.1% rank-1 along dim0 (+32.6
  near-constant)**. That substitution is exactly what an *unpreconditioned* GN does when one
  output direction is 10³–10⁵× cheaper to move in input space than the others — `pi2` MEASURED
  the input-Jacobian ratios p0 : p1 : p2 : p3 : p4 : p5 = 1 : 1/76 : 1/64 : 1/215 : 1/426 : 1/233.
  **Whitening by `M^{-1/2}` removes the incentive to substitute.**
- **ΔS arithmetic.** Not a direct ΔS; it is a **convergence + correctness** lever on the binding
  constraint, priced against §2's setpoints: |Δp₀| = 0.0245 buys −0.261 S; 0.00775 buys −0.283 S.
  Context for how far there is to go: `su2` MEASURED that the bar needs `d_pose < 3.124e-4` at
  `d_seg = 0`, and sub-0.15 needs `< 1.139e-4` — **27× to 75× below our 8.58e-3.**
- **Cheapest measurement.** $0: instrument the existing solve to log the pose residual's projection
  onto the six singular directions per iteration. If the residual is stalled in σ₄–σ₆ while σ₁ is
  converged — the signature of aliasing — the diagnosis is confirmed with no new run. Compose with
  P0's per-dim breakout, which is the same projection at the endpoint.
- **Named falsifier.** If the logged residual is already at machine precision in all six modes
  after k iterations, preconditioning is moot and the row is dead. Also dead if P0 shows dims 1–2
  dominate *and* the overshoot is not reduced by whitening in a 1-pair rehearsal.
- **Class.** $0 diagnostic; the fix is a code change inside an existing engine
  (`experiments/ddm_pfs1_ep_warp_pose_solve.py::solve_pair_gn`, `experiments/ddm_tt1_joint_tto.py`).

### P3 — Test whether common-mode carrier error is pose-cheap · ORIGINAL-DERIVED
- **Axis.** pose (gap 0.2776) — and it prices the live `f0 := a·warp(f1)+b` carrier directly.
- **Mechanism.** §5b: the stem is sum-dominated (antisym 32.5%/25.1% vs 50% trivial), so motion is
  inferred deep. If a perturbation identical in both frames is motion-neutral, the live carrier's
  structurally common-mode error is cheap and its **antisymmetric** component is the whole cost.
- **ΔS arithmetic.** None asserted — this is a *typing* measurement that would re-price where the
  0.008581 comes from. Its value is that it splits our residual into a part we can stop paying for
  and a part we must attack.
- **Cheapest measurement.** On ≤24 cached pairs: inject a fixed photometric perturbation (a) into
  f1 only, (b) into f0 only, (c) identically into both; measure Δd_pose for each at matched
  perturbation norm. Ratio (c)/((a)+(b)) is the answer. One short scorer slot.
- **Named falsifier.** If Δd_pose(common) ≥ 0.5·(Δd_pose(f0)+Δd_pose(f1)), common-mode is not
  cheap and the hypothesis dies.
- **Class.** short scorer slot. Pre-register that the *across-pairs* statistic is not the test.

### P4 — Sweep the zero-byte decode operator, CO-MEASURED on pose · ADAPTED (ua3) + HYBRIDIZED
- **Axis.** seg (0.4015) at **exactly zero rate**.
- **Mechanism.** ua3: shipped `inflate.py` unsharp α ∈ {0.27, 0.40, 0.85, 2.0, variance-adaptive},
  7.4× spread, zero archive bytes, **no measurement against the exact scorer anywhere in the tree**.
- **HYBRIDIZE.** α is 1-D and the pose response is ~1-D in dim0 (§1–§2), so the (α, ΔS) trade is a
  1-D × 1-D curve that can be solved rather than swept blindly — and there may be an α that is
  pose-*improving*, since unsharp partially inverts the low-pass that our record already names as
  the enemy (`R_surv`).
- **SCOPE GUARD, stated because it may kill the row.** The shipped unsharp is gated on
  `if H != target_h or W != target_w` — it exists to invert *their* store-at-522×392 upscale. **I
  did not resolve whether our description-carrier decode contains an equivalent low-pass to
  invert.** If v4d's receiver emits at 874×1164 natively with no upscale, this row is void. That
  check is $0 and is the first step.
- **ΔS arithmetic.** Unmeasured by construction. The row's value is that it is *pure* — any gain is
  rate-free. Its cost is one scorer slot per α.
- **Named falsifier.** If the $0 scope check finds no upscale in our decode, dead immediately.
- **Class.** $0 check, then ≤5 scorer slots.

### P5 — Re-price the pose stream by absolute quantum, and confirm nothing is left · ADAPTED, CONFIRMATORY
- **Axis.** rate (gap 0.1126) — but this row's honest verdict is **near-closed**, and saying so is
  the deliverable.
- **Arithmetic.** `tp_member` = 6,378 B of 360,238 (1.77%). §3 shows f16 misallocates ~10³× against
  a raw-MSE objective; **QA65's dim0-offset already fixed it** (19.3× finer, 0 bytes). Residual
  headroom: §2's setpoint says dim0 needs |Δp₀| ≤ 0.00775 for d_pose 1e-5, i.e. ~8.8 bits/pair over
  the observed 11.79 range ⇒ **~660 B for all 600 dim0 values**, against 6,378 B for the 600×6 f16
  plane. **Max ΔS from perfect pose-stream coding = 25·(6378−~1500)/37,545,489 ≈ −0.0032.**
- **Which way it falls.** −0.0032 is real but ~90× smaller than the pose *realization* gap
  (−0.2777). **Do not spend an arm here.** Consistent with `ja1`'s `saturated_do_not_spend` and
  with `xi1`'s measured 1,617 B (= 0.001077 S) unreachable token-stream gap.
- **Class.** $0; already answered. Recorded so it is not re-opened.

### P6 — The limited-vs-full range asymmetry: MEASURED, and it is EMPTY · ORIGINAL-DERIVED (honest negative)
- Recall reports this is the one colour-space surface with **no prior work in scope** — "the
  cleanest open surface." I measured it rather than leaving it open.
- **Coordinates.** GT decode is BT.601 **LIMITED** (`(y−16)·255/219`, `frame_utils.py:176-183`);
  the scorer's encode is BT.601 **FULL** (`(B−Y)/1.772+128`, `:62-63`). If the GT's uint8 values
  concentrated on the limited-range-expanded grey lattice (220 of 256 codes), a codebook over that
  lattice would save 2.7% of bits on any raw-uint8-coded field.
- **MEASURED** (12 of 96 frames, 12,208,032 px × 3 ch = 36,624,096 samples):

  | | R | G | B |
  |---|---:|---:|---:|
  | distinct codes used | **256** | **256** | 255 |
  | on-lattice fraction | **85.924%** | 86.557% | 85.944% |
  | trivial baseline (220/256) | 85.938% | 85.938% | 85.938% |
  | marginal entropy | 6.115 b | 5.929 b | 5.661 b |

  On-lattice fraction is **exactly the trivial baseline**. Only at *exactly grey* pixels (0.119% of
  the frame) is it 100%, and that is tautological. At `max−min RGB ≤ 2` (1.617% of pixels) the
  excess is 2.6pp. Median chroma spread is 13.
- **Mechanism of the negative.** The bilinear chroma upsample (`F.interpolate(..., 'bilinear')`,
  `:173-174`) makes `uf`/`vf` non-integer at essentially every pixel, dithering the value off the
  lattice before the round.
- **Which way it falls.** **CLOSED at FAMILY scope for this video.** No dead codes, no lattice
  structure, no codebook. Recorded so the "cleanest open surface" is not re-opened.

### P7 — Two structures reported as bounds, explicitly NOT levers · ORIGINAL-DERIVED (honest)
- **PoseNet's 681/20,744 dead BN channels (3.283%)**, concentrated in FastViT stage-2/3
  `token_mixer.conv_kxk` (18.8–25.8%), vs **SegNet's 0/33,368**. A dead BN channel is a *feature*
  channel, not an image region — **it is not input-addressable and therefore not an actuator.**
  What it is: a capacity bound on the pose representation, mechanistically upstream of §1's
  eff-rank 1.013. Reported as corroboration. The "prune the scorer" reading is a NEGATIVE.
- **SegNet head flip-leverage.** MEASURED pairwise `‖Δw‖`: Lane–Movable **4.0074** (easiest to
  flip) → Road–Undrivable **2.6018** (hardest), ratio **1.5402×**; 3×3 tap energy is centre-weighted
  at 23.89% (trivial 11.11%) but the centre tap alone captures only **24.18%** of argmax-effective
  Frobenius² — **a 1×1 approximation of the head loses 76%**. Composed with the banked copy-flip
  mass (Road 50 / Lane 25 / Undriv 13.5 / Movable 6 / MyCar 6): **our dominant flip class sits on
  the lowest-leverage pair**, so Road flips are margin-limited, not leverage-limited — which agrees
  with rp1's measured 166× margin gap (0.034 flipped vs 5.61 held). **I do NOT convert this into a
  per-class capacity price**, because `gc13` MEASURED that per-class rate floors are ILL-POSED (the
  tr1 partition stream is class-SHARED, universal lower bound 0 B). The leverage table is a
  *coordinate*, not an exchange rate. The pairwise `‖Δw‖` structure itself is PRIOR
  (`segnet_recursive_fractal_factorization_20260715`, carried in `ja1`); the tap-energy split and
  the 1×1-loses-76% number are new.

---

## 8. Round-1 adversarial review of myself

| I tried to refute | outcome |
|---|---|
| "the head-chain concentration is just a random-matrix product artifact" | **REFUTED with a 200-draw matched control.** Control eff-rank median 5.23, dim0 23.1%, argmax dim uniform; trained 1.013 / 99.71% / dim0. Trained cond is 142× the control max. |
| "the 99.7% dim0 share is my headline" | **DEMOTED BY MYSELF.** `pi2` measured ~50× (≈91%) at the *real input on real pairs*; mine is a head-chain upper bound that excludes FastViT's nonlinearity. pi2 is the authority. I kept mine only as a depth profile and stated the 4× gap as the finding. |
| "the p0 dominance is my discovery" | **REFUTED — it is PRIOR, and older than I first credited.** The original is `project_posenet_rank1_discovery.md`, **~2026-04-24** ("Jacobian rank 1.008; dim0 = 99.80% of variance; mean 31.26, std 1.26") — my 99.8025% / 31.2606 / 1.25634 reproduces it to four digits, **98 days later**. Then `da1`/`ar1` (07-28, t_p 0.998) and `pi2` (07-30, QA51: p0 ~50×, cond 24.8, the **26-dim head-null explicitly named**, 99.3% luma, 52% Jacobian on frame_0). ua1 re-derived 24.8. **Surviving as new: the depth profile, the 200-draw control, the σ-ladder, the stem symmetry, the §3 unification.** |
| "≥93.06% of our d_pose is dim0" — my own §2 headline | **REFUTED AS AN UNCONDITIONAL BOUND, by our own banked receipt.** It assumes predicted dims 1–5 stay inside the GT range; `ddm_pfs1_d2_price_receipt_20260729.json` measures `e_p_per_dim_std = [0.8195, 0.117, 0.238, …]` — dims 1–2 overshoot their GT std by 3.3× / 8.0× and contribute 0.0117, **20× my assumed ceiling**. Downgraded to CONDITIONAL, condition stated, counterexample cited, and the fix promoted to **P0** (the per-dim breakout of the *current* residual, which recall confirms **nobody has measured**). |
| "pose is a 600-pair field problem" | **REFUTED by banked measurement.** `ddm_v4d…` §4: top-17 pairs = **74.3%** of d̄ mass, median 0.00088 vs mean 0.00858. It is ~17 pairs. Added to §2. |
| "the σ-ladder is only a speedup" | **STRENGTHENED, not refuted.** `qa43` MEASURED rotation/translation aliasing ("the solver substitutes speed for turn"; tail correction 98.1% rank-1 on dim0) and `pi2` measured the input-Jacobian ratios 1 : 1/76 … 1/426. Unpreconditioned GN *should* alias given that spectrum. P2 is therefore the cure for a measured pathology, not a convenience. |
| "the banked R1 dxi 7.2 KB is our pose carrier" | **CORRECTED.** `dxi` is the **r6cal/witness** lineage and is SUPERSEDED (`iv1`/`iv2` list it superseded by `xi_pose_coder` delta_ar at 474–875 B); the live carrier is `pose_warp.stp` (~8.6 KB, grammar `PFS1WPD1`). Its 0.001610 also came from a **STOPPED** run. Table annotated. |
| "e_p rank-1 ~2 KB closes pose" | **REFUTED by our own adjudication.** `su2`: *"Representation/init race after the solve. Do not add 2,039 B or infer a d_pose endpoint."* Not cited as a floor anywhere here. |
| "so rank-1 means we can code the pose field rank-1" | **REFUTED by our own banked QA50/QA61**: standardized SVD energy is FLAT `[0.221…0.130]`, the dz-carrier rank-1 premise is a scale artifact and is CLOSED. §3 reconciles rather than re-opens: the consequence is asymmetric *precision*, not rank-1 *coding*, and **QA65 already shipped the right fix**. |
| "then the pose stream is the lever" | **REFUTED by arithmetic.** Max ΔS from perfect pose-stream coding ≈ −0.0032, ~90× below the realization gap −0.2777. Row P5 records it as near-closed. |
| "frame_0 should be generated, not stored" | **REFUTED as novel — ALREADY BUILT.** v4d stores no frame_0 (`f0 := a·warp(f1)+b`). This is exactly the re-anchor-as-discovery failure the memory names; caught by recall before writing. |
| "warp is closed, so the frame_0 warp carrier is dead" | **SCOPE-CORRECTED.** r2s's closure is on warp-as-SEG-PREDICTOR (flip support 0.9988× neutral, strat_full +7.1% worse). The same memo's §7(a) calls the store-nothing WARP carrier "the live pose path at ~0B." Different surfaces; I did not over-apply the closure. |
| "the limited-vs-full range mismatch is an open exploit" | **REFUTED BY MY OWN MEASUREMENT.** On-lattice fraction 85.92/86.56/85.94% vs 85.938% trivial baseline; 256/256/255 codes used. Empty. Closed rather than left open. |
| "PoseNet's dead channels are a lever" | **REFUTED.** Feature channels are not input-addressable. Demoted to a bound in P7 and labelled a negative. |
| "the stem symmetry is a claim" | **DOWNGRADED to a hypothesis with a named test** (P3). The measurement (cos +0.351/+0.497, antisym 32.5%/25.1% vs 50%) is MEASURED; the *consequence* for d_pose is not, because motion is read deep. |
| "the charter's 1.24 S pose weight" | **CORRECTED at the primary artifact.** At v4d, pose = 0.292939. Ranked by gap-to-demonstrated-floor: seg 0.4015 > pose 0.2776 > rate 0.1126, and those three sum to exactly the PR130 row 0.172141. |
| "my ker(W₆) unscored-mass number" | **DISAGREEMENT REPORTED, NOT SMOOTHED.** ua1 reports 93.5% of the unscored head's Frobenius mass lies in ker(W_scored); my projection onto the orthogonal complement of W₆'s row space gives **87.35%**. I could not reconcile the two definitions in the time available. Neither number is load-bearing for any row here. **OWED.** |
| numpy emitted `divide by zero / overflow / invalid in matmul` warnings | **Investigated, not ignored.** All 13 weight matrices verified `isfinite` = True; all 8 AllNorm scalars finite and O(10⁻²–10⁰); all outputs finite and the control behaved as expected. Attributed to the Apple Accelerate BLAS backend, not to the data. |

**Where my coverage is incomplete, specifically.** (1) The pose-axis recall sweep landed *after*
the first draft and forced four corrections above — the base rate of my un-recalled claims was
therefore **not zero**, and I have no reason to think it is zero now for surfaces neither sweep
covered (neither sweep read `/Volumes/VertigoDataTier/pact/` raw receipts). (2) Every Jacobian here is
**linearized under all-ReLU-active and stops at the vision-2048 boundary**; FastViT is excluded and
pi2's real-input measurement already shows it de-concentrates by ~4×. (3) P4's scope guard is
unresolved and could void that row outright. (4) I fired no forward pass, so **no row here is a
measurement of our vehicle's response** — only of the frozen operator and the GT targets.

---

## 8b. Where these rows sit against the LIVE blocker

`current_focus.md` names the standing decision: **ep854 dominates the gr1 seg base by −0.035996 S
(seg+rate), byte-closed — but v4d's pose payload was solved against gr1's RENDERS.** A base swap
ships corrections fitted to different pixels; it is *measurable in one n600 eval, never
assumable*. The ordered program (su2) is **rate parent FIRST, then re-solve pose.**

Every row here is compatible with that order and none of them requires jumping it:

| row | when it fires | cost |
|---|---|---|
| **P0** per-dim breakout | now, on the *existing* v4d gate — its value is diagnostic and does not depend on the parent | ~free |
| **P2** preconditioning diagnostic | now, on existing solve logs | $0 |
| **P2** preconditioning fix | **before** the post-swap pose re-solve — that re-solve is exactly when it pays | code change |
| **P1** k=6 pose-null | at the repaint stage, which the #383 staging law places *before* the terminal pose solve | 1 slot |
| **P3** common-mode typing | any time; it types the carrier, not the parent | short slot |
| **P4** decode-α | after the $0 scope check; independent of the parent | ≤5 slots |

`su2` already MEASURED the wall this order is fighting: **every non-ideal pose term in the banked
table exceeds the 0.172141 bar on its own**, and the bar needs `d_pose < 3.124e-4` at `d_seg = 0`
(sub-0.15 needs `< 1.139e-4`) — **27× to 75× below our 8.58e-3.** The pose axis is not a polish
item on this vehicle; it is load-bearing, and §1–§4 say its target is low-dimensional.

---

## 9. What this feeds

- **P0 (per-dim breakout)** → the composed co-measure harness; it is the cheapest open measurement
  on the #2 axis and recall confirms it has never been taken on the live carrier.
- **P1 (k-mode pose-null ladder)** → `j11`'s built pose-null/seg-null machinery + the composed
  co-measure harness; it is the price tag that was missing from "every correction is pose-audited."
- **P2 (exact head metric)** → the `fd1` / family-d GN·CG build's named-but-unspecified
  "scorer-metric preconditioner," pose half, closed-form.
- **P3 (common-mode)** → types the live `f0 := a·warp(f1)+b` carrier's residual.
- **P4 (zero-byte decode α)** → ua3's most actionable coordinate, now with a pose co-measure and a
  scope guard.
- **P5/P6/P7** → three surfaces recorded as **closed or near-closed** so no future arm re-opens
  them: pose-stream coding (−0.0032 max), the limited/full range asymmetry (empty), scorer
  dead-channel pruning (not an actuator).
- **§0** → the axis-weight correction; the charter's 1.24 is a superseded base coordinate.

**Pointer 0.1910828242 [contest-CPU] UNMOVED.** This memo moved no score and claims none. Every
row above is MEANS; the END is a lower exact score, and none of this is that until an arm fires.
