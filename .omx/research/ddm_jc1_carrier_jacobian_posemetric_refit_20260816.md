# jc1 — the carrier→PoseNet Jacobian, and what it measured: K = 0, and the pose-metric re-fit is WORSE than the Euclidean one

`date_utc: 2026-08-16` · `owner: ddm_jc1` · `axis: [macOS-CPU advisory, n600]`
`score_claim: false` · `promotable: false`
`verdict: REFUSED (re-fit rung, r=11 MEASURED; other rungs DERIVED monotone)`
payload: `/Volumes/APDataStore/pact/ddm_jc1/retained/`

## THE ANSWER, FIRST

1. **K = 0.** The 600 per-pair pose-null subspaces intersect trivially — at every tolerance
   from 1e-16 to 1e-1. There is no exactly-free carrier direction. Worse for the idea: the
   stacked Jacobian's condition number is only **12.02**, and the 12 column sensitivities
   span just **2.16×**. Nothing in the carrier is cheap.
2. **The pose-metric re-fit is MEASURED and it LOSES to the Euclidean incumbent.** Five
   candidates, exact n600 `d_pose` through the real chain, at the ladder's best rung (r=11,
   1,854 B back): Euclidean **235.3×**, pose-metric **238.9× / 258.6× / 272.2× / 256.7×**.
   Best of all five misses the advisory bar (1.0653) by **220.9×** and the T4 bar (1.3197)
   by **178.3×**.
3. **The first-order promise was an artifact, and the exact row is what killed it.** The
   linear model predicted the pose re-fit would land at ratio 0.24–1.93 (i.e. clear the
   bar). Measured: 256.7–258.6. The model was wrong by **134× to 1,065×** on exactly the
   candidates it designed. I nearly reported those predictions; the measurement is why I did not.
4. `rank_refit_status = NOT_MEASURED_BY_THIS_EXACT_RACE` (`ddm_mp2_carrier_exact_byte_race.py:132`)
   **is now closed.** It is measured. It is refused.

**ΔS for the best candidate, both axes** (`d_seg` invariant, measured identically on four
treatments now): advisory `ΔS = +0.5494`; T4 by ratio-transfer `ΔS = +0.1177`. Against a
remaining gap of **−0.0096**.

---

## 1. Step 0 — what I actually found before building

**The ms3/ms4 reuse claim is REFUTED by source inspection, not merely unverified.**
`src/tac/optimization/ddm_metric_producers.py:183-217` (`pose_quadratic_row`) computes no
Jacobian at all. Its "low_rank_factors" is `np.eye(6)/sqrt(6)` — a fixed identity scaling
that makes `‖Fᵀδ‖² = mean(δ²)` hold for the quadratic `mean((pose6 − center)²)`. That is a
quadratic in **pose-output space**, with a stored 6-vector center. Nothing in it is
differentiated with respect to carrier coefficients, and `tools/produce_ddm_ms4_metric_custody.py:5-9`
says in its own header that only the Pose producer is identifiable and the rest await a PF2
bucket assignment. MAIN's suspicion was right; the ERRATUM's proposed reuse was not available.

**What DOES exist, and what I reused.** `experiments/ddm_js4_pose_null_projected_conditioning.py:228`
computes a real per-pair exact PoseNet Jacobian — but it is **6 × 589,824 in PIXEL space**,
at the **CP135** custody base, over the js3 correction, i.e. a different vehicle and a
different variable. `src/tac/boundary_math/posenet_subspace_spectrum.py` is the general
pixel-space pose-null machinery (`_orthonormal_row_basis`, `participation_ratio`). I reused
js4's *pattern* (6 `autograd.grad` passes, STE through uint8, differentiable yuv6) and built
the carrier-coordinate producer, which is 49,152× smaller per pair and is the object the two
open items actually need.

**Reusing js4 wholesale would have been wrong anyway**: a 6×589,824 pixel Jacobian answers
"which pixels does pose read", not "which carrier coefficients does pose read". The chain
rule between them runs through the render, which is where the clamps and the selector live.

## 2. The producer

`experiments/ddm_jc1_carrier_pose_jacobian.py` — for every pair `i` in 0..599:

    J_i = ∂( PoseNet(pair_i)['pose'][:6] ) / ∂( carrier_coeff_i )   ∈ R^{6×12}

through the shipped chain: fx1 carrier render (`inflate.py:659-676`) → the sparse frame-0
selector → upstream PoseNet preprocess (bilinear→384×512, **then** `rgb_to_yuv6`, verified at
`upstream/modules.py:71-75`) → frozen CPU-torch PoseNet.

**Forward exact, backward relaxed — the distinction is load-bearing.** The forward evaluates
both `.round()` calls hard and is required to be BYTE-IDENTICAL to the retained `0.raw`.
Backward uses a straight-through estimator on both rounds, because the exact chain's Jacobian
is 0 almost everywhere and carries no information. Both `clamp` gradient masks are kept real —
a clamped pixel genuinely has zero sensitivity. Every J_i is therefore labelled
`MEASURED_ON_STE_RELAXED_CHAIN`, and §5 measures what that relaxation is worth.

**The control earned its keep immediately.** My first implementation omitted the sparse
frame-0 **selector** — `apply_pixel_mode` runs after the carrier render (`ra2c:207-210`) on
**5 of 600** frames. The byte-identity control failed on exactly those pairs and nowhere else.
Without it I would have shipped 5 wrong rows and never known.

### Instrument chain (four independent controls, all green)

| control | what it proves | result |
|---|---|---|
| forward byte-identity vs shipped `0.raw` | the chain is the SHIPPED chain | **600/600** |
| base `d_pose` rebuilt from residuals `r_i = pose6_gen − pose6_gt` | the scorer pipeline matches upstream | 0.0001474678 vs report 0.00014747 → **1.478e-5** rel |
| no-op: base coefficients back through the exact evaluator | the candidate evaluator is unbiased | 0.00014746613 → **2.62e-5** rel |
| **central finite difference vs the STE Jacobian** | **the BACKWARD, which nothing else checks** | cosine **0.9992–0.9998**, rel. Frobenius err **2.7–4.4%** (3 pairs, step 0.05) |

The first three controls all test the FORWARD; only the fourth tests the gradient.
`--fd-check` perturbs each of the 12 coefficients ±δ, runs the **hard-quantized** render both
ways, and compares `(pose6(c+δ) − pose6(c−δ))/2δ` against the STE column:

| pair | step | cosine | rel. Frobenius err | ‖fd‖/‖ste‖ |
|---:|---:|---:|---:|---:|
| 0 | 0.05 | 0.99956 | 0.0296 | 1.0001 |
| 299 | 0.05 | 0.99924 | 0.0444 | 1.0205 |
| 599 | 0.05 | 0.99978 | 0.0265 | 0.9839 |
| 0 | 0.01 | 0.98391 | 0.2155 | 1.1044 |

Agreement degrades as the step approaches the quantization floor (0.01 row) — the correct
signature, since below one LSB the render is byte-identical and the true finite difference is
exactly zero. **The STE Jacobian is accurate to ~3% against the real quantized chain.** That
matters for §5: the design failure there is not a broken gradient.

The report prints `d_pose` to 8 decimals (~3.4e-5 relative granularity), so both controls sit
at the report's own resolution. This producer reproduces `upstream/evaluate.py`'s pose
pipeline end to end — including the batch-1 vs batch-16 instrument question ([[et4]]), which
therefore does not bite on the pose axis at this magnitude.

### Retained payload (`/Volumes/APDataStore/pact/ddm_jc1/retained/`, 501.4 s, n600)

| file | shape | bytes | sha256 (16) |
|---|---|---:|---|
| `jacobian_pose6_x_coeff12.float64.npy` | (600, 6, 12) | 345,728 | `fc5743c970c13f4f` |
| `pose6_generated.float64.npy` | (600, 6) | 28,928 | `2dba9590b1811f0e` |
| `pose6_groundtruth.float64.npy` | (600, 6) | 28,928 | `f73ec194b379a7c0` |
| `selector_mode_kind_a_b_c.int32.npy` | (600, 4) | 9,728 | `00373e16daa81fc3` |
| `control_byte_identical.bool.npy` | (600,) | 728 | `66740ee364799746` |
| `pair_ids.int32.npy` | (600,) | 2,528 | `060dcb8dd46355c0` |
| `coeff.float64.npy` | (600, 12) | 57,728 | `0507cb14b533520f` |
| `basis_gram.float64.npy` | (12, 12) | 1,280 | `f96c790c4241a2e6` |

Receipts: `JC1_PRODUCER.json`, `JC1_CONSUMERS.json`, `JC1_EXACT_CANDIDATES_r11.json`,
`JC1_EXACT_NOOP_CONTROL.json`. Archive pinned at sha `80d9c8c6…`, 182,759 B.

## 3. Consumer (b) — K(tolerance). MEASURED K = 0, and the spectrum is the real finding

`v ∈ null(G_i) ∀i ⟺ J_i v = 0 ∀i ⟺ J_stack v = 0`, so the 600-fold intersection is one SVD
of a 3600×12 matrix and `K = 12 − rank(J_stack)`.

| rel. tolerance | 1e-16 | 8.0e-13 (numpy) | 1e-10 | 1e-6 | 1e-4 | 1e-2 | 1e-1 | 3e-1 |
|---|---|---|---|---|---|---|---|---|
| **K** (raw stored coord) | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 9 |
| **K** (coeff-RMS whitened) | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 9 |

Both coordinates are NAMED and derived, not chosen: *raw stored* is what the archive holds;
*coeff-RMS whitened* scales column `k` by `RMS_i(c_ik)`, which is the size of the move "drop
this column" actually makes. K is identical in both, as it must be for an exact null space
(diagonal rescaling cannot create or destroy one).

**K = 0. My pre-registered generic expectation, confirmed — but the informative number is
how far from null the carrier is:**

```
σ/σ₁ = 1, 0.465, 0.348, 0.299, 0.230, 0.216, 0.169, 0.148, 0.126, 0.116, 0.094, 0.083
cond(J_stack) = 12.02          per-column pose sensitivity spread = 2.16×  (140.9 … 65.3)
```

**And the pose spectrum is FLATTER than the Euclidean one it was supposed to beat.** ra2c
measured cond 17.32 on the rendered field. Pose: 12.02. The metric the score reads
concentrates the carrier *less* than the metric ra2a optimised. There is no attenuated
subspace to route bytes out of — which is the mechanism behind §4.

## 4. Consumer (a) — the pose-metric re-fit, MEASURED, and it loses

Both re-fits solve for the same free variables (the kept coefficients, dropped columns forced
to zero) over the same exhaustive C(12,r) keep-set search
(`ddm_ra2a_carrier_fidelity_pose_ladder.py:143-160`, same Gram, verbatim). Only the objective
differs: Euclidean minimises the rendered-FIELD error; pose-metric minimises
`‖r_i + J_i·δ_i‖²`, the scored quantity.

**EXACT n600 `d_pose`, at r=11 (1,854 B returned, advisory bar 1.0653, T4 bar 1.3197):**

| candidate | `d_pose` MEASURED | ratio | first-order predicted | model error |
|---|---:|---:|---:|---:|
| Euclidean incumbent | 0.03469834 | **235.3×** | 690.9 | 2.94× over |
| pose-metric, ridge 1e2 | 0.03523752 | 238.9× | 45.5 | 5.25× under |
| pose-metric, ridge 1e1 | 0.03813447 | 258.6× | 1.93 | **134× under** |
| pose-metric, ridge 1e0 | 0.04014700 | 272.2× | 0.425 | **640× under** |
| pose-metric, ridge 1e-1 | 0.03786066 | 256.7× | 0.241 | **1,065× under** |

Three readings, in order of importance:

1. **The pose-metric re-fit is not better. It is 1.5–15.7% WORSE than Euclidean.** The whole
   measured spread across five candidates is 235.3–272.2 — a factor of **1.16**. The metric
   choice on this object is worth ~16%. The rung needs **221×**.
2. **The bar is missed by 220.9× (advisory) / 178.3× (T4) at the ladder's BEST rung.** r=11 is
   the loosest rung: the first-order ladder's miss is monotone in dropped dimensions
   (648× at r=11 rising to 5,173× at r=1), so every other rung is worse. Rungs other than
   r=11 are **DERIVED** from that monotone ladder recalibrated by the measured r=11 point,
   not measured.
3. **`ΔS` is catastrophic on both axes.** Advisory: `√(10·0.0346983) − √(10·0.00014747)` =
   `0.58905 − 0.038402` = `+0.55065` pose, minus `0.0012344` rate → **`ΔS = +0.5494`**.
   T4 by ratio-transfer: `0.0082946·(√235.29 − 1) = +0.11893`, minus rate → **`ΔS = +0.1177`**.
   The remaining gap is **−0.0096**.

### The artifact I caught, and how

The unregularized pose re-fit reported predicted ratio **0.000** at every `r ≥ 6` — apparently
clearing every bar. It is a dimension count, not a result: `d_pose` reads 6 scalars per pair,
so with `r ≥ 6` free coefficients `J_S` (6×r) generically has full row rank and the
first-order residual is drivable to exactly zero for *every* keep-set of that size. The tell
was the step: **3.4× to 43.9×** the carrier's own RMS magnitude. The trust-region sweep
(Tikhonov on the kept block) brought the step back to 0.23–0.25 and the predicted ratio down
to 1.93 at ridge 10 — close enough to the 1.0653 bar that no honest first-order screen could
refuse it. That is precisely why the five candidates were measured exactly instead of reported.

The forbidden version of this memo says "the pose-metric re-fit clears the advisory bar at
r=11 with a predicted ratio of 0.24." Every word of that is derivable from my own numbers and
all of it is false.

## 5. What the linearization is actually worth (labelled, since two of my rungs used it)

| treatment | step (RMS rel.) | `d_pose` MEASURED | first-order predicted | error |
|---|---:|---:|---:|---:|
| ra2c rank-4 SVD truncation | 0.502 | 0.35402399 | 0.61319 | 1.73× over |
| Euclidean re-fit r=11 | 0.247 | 0.03469834 | 0.69087 | 2.94× over |
| pose re-fit r=11, ridge 1e-1 | 0.240 | 0.03786066 | 0.00003553 | 1,065× under |
| α = 0 (carrier deleted) | 1.000 | ~51.7 (from ra2c's law endpoints) | 1.91292 | ~27× under |

**Two things follow, and the second is the one that matters.**

*(i)* The Jacobian is a better calibrated screen than the damage law at the same point: at
rank-4 it errs **1.73×** where the endpoint-fitted quadratic (`K = 350,427`) erred **9.23×** —
a 5.3× improvement, and it needs no fitted constant.

*(ii)* **The error is not a function of step size — it is a function of whether the step was
CHOSEN BY the linear model.** Two candidates at nearly identical step (0.247 and 0.240) err by
2.94× and 1,065×. The difference is that the second was *designed* by the linear model to
cancel pose error. A linearization used as a **screen** is decent here; used as a **designer**
it over-estimates its own control authority by three orders of magnitude.

**And the gradient is not the culprit** — §2's finite-difference control puts the Jacobian
within ~3% of the real quantized chain. So the 1,065× is genuine nonlinearity of
coefficients → render → R → uint8 → PoseNet beyond a small neighbourhood, not a bad
derivative. That is direct evidence for ra2c §8.2's promotion of `p ≠ 2` from footnote to
live, and it is the sharper form of the claim: *an accurate local derivative bought no design
authority at all at a step of 0.24.* "Agreeing with your own optimizer" is the shape the
failure takes.

## 6. Verdict + scope

`verdict_scope: FORMULATION` — **coordinate keep-set + coefficient re-fit, in either metric,
on the hv1 carrier**. MEASURED at r=11 with five candidates; other rungs DERIVED from the
monotone first-order ladder recalibrated at the measured point. Not a FAMILY verdict on
"pose-aware carrier design", and not a claim about any other vehicle.

What this closes, plainly:

- `rank_refit_status = NOT_MEASURED_BY_THIS_EXACT_RACE` → **MEASURED, REFUSED**.
- ra2c §8.4's `K` question → **K = 0**, with the mechanism (cond 12.02, column spread 2.16×).
- ra2c §5's withdrawn "whiten by G then truncate" successor → the whitening is now measured
  and it is *worse* than the coordinate it replaced.

**The carrier stops consuming slots.** Both of ra2c §8.4's branches have now fired the same
way: no exactly-free direction, and no approximately-cheap one either. The carrier is
22,161 coded bytes that the pose axis genuinely needs at ~235× per 1,854 B returned.

## 7. Honest limits

- **Advisory axis, single instrument.** The T4 column assumes the `d_pose` ratio transfers.
  Not a score claim; not promotable.
- **r=11 is measured; the rest of the ladder is derived.** Monotonicity is a property of the
  first-order ladder, and §5 shows first order is unreliable as a designer. It is being used
  here only as an *ordering*, on treatments none of which were designed by it — the weakest
  use, but still an assumption. Measuring r=8 exactly would cost 62 s and is unowned.
- **`d_seg` is not re-measured here.** It is taken as invariant from four independent
  treatments (α=0, α=1, rank-4, and this producer's byte-identity control). If any future
  carrier edit touches frame_1, that invariance does not transfer.
- **The 235.3× Euclidean row measures the KEEP-SET + re-fit rung**, which is a different
  treatment from ra2c's rank-r SVD *truncation* (a subspace, not a coordinate keep-set). The
  two are not directly comparable and I have not compared them.
- **`α=0` measured value** in §5 is back-derived from ra2c's law endpoints, not re-measured
  here. It is labelled as such and carries no weight in the verdict.
- **`BYTES_PER_DROPPED_DIM = 1853.5` is ASSUMED-BY-TRANSFER**, not measured for this
  treatment. It comes from ra2c §8.1's table for the SVD *rank* ladder; I apply it to a
  *coordinate keep-set*, where dropping column `k` removes 1/12 of the coefficient payload.
  The two should coincide, but I did not re-code the archive to confirm it. The verdict is
  insensitive: the miss is 220.9×, so the byte figure would have to be wrong by two orders of
  magnitude to change it.
- **The finite-difference control covers 3 of 600 pairs.** It is a spot check on the gradient,
  not a per-pair guarantee.

## 8. Apparatus product (reusable, beyond this verdict)

`experiments/ddm_jc1_carrier_pose_jacobian.py --eval-coeff <coeff.npy>` measures TRUE n600
`d_pose` for any candidate carrier coefficient matrix in **62 s**, with no bulk artifact. The
ra2c full path costs 587 s and materialises 3.66 GB of raw frames. On any question where
`d_seg` is invariant — which is every frame_0 carrier question, measured four ways — this is a
**9.5× faster exact instrument**, validated by a no-op control at 2.6e-5 relative. It should
be the default screen before anything spends the full evaluate.py path.

---

**Pointer UNMOVED: hv1 ep0634, S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600].**
This unit produced no lower score. It closed a rung that was consuming slots, with a measured
row rather than an extrapolation, and it left behind a fast exact pose instrument.
