# ddm_obx2 — design A closure at formulation scope

<!-- FORMALIZATION_PENDING: the laws this memo rests on are registered as obx2_pose_vs_scorer_plane_rmse_v1; the closure itself is a verdict over measured rows, not a new equation. -->

Date: 2026-09-11
Status: `CLOSED AT FORMULATION SCOPE — MECHANISM VALIDATED, OBJECT REFUSED`
Measurement axis: `[macOS-CPU advisory]`, n600, shipping torch receiver on the parsed packet
Score claim: false · Promotion eligible: false · Pointer moved: false
Lane: `ddm_obx2_edge_local_implicit_correction_20260911`

## Verdict

**The edge-local implicit correction lattice WORKS. The object it corrects does not.**

Design A is closed at FORMULATION scope — the QBF1 born generator form with this lattice geometry and
this budget — for two measured reasons that are independent of each other and of the lattice:

1. **89.83 % of the seg error is partition-level**, at sites where the generator already represents
   the wrong class. An RGB correction is not aimed at it.
2. **The render floor ALONE is `d_seg` 0.0023099568 — 5.77× the 4.0e-4 seg ceiling.** Even with a
   PERFECT partition, this generator's render-plus-scorer path misses the gate on its own.

Leg 2 is the one that closes it. Leg 1 only says the lattice is working on the small share.

This is not a falsification of the mechanism. The ablation shows the lattice doing exactly what
design A claims, and doing it well.

## What the lattice did — MAIN's attribution rule, PASSED

Same checkpoint (lattice arm, stage w2, epoch 30), scored twice on all 600 pairs through the shipping
receiver, once with the lattice head live and once with it zeroed:

| | archive | `d_seg` | `d_pose` | distortion |
|---|---:|---:|---:|---:|
| with lattice | 122,778 B | 0.02271701389 | 0.01138376106 | 2.6090997 |
| lattice zeroed | 122,203 B | 0.02408198886 | 0.07326326191 | 3.2641385 |

* `seg_without − seg_with` = **0.001364974976**, against MAIN's threshold of 0.0008921983506
  (50 % of the stage's seg progress). **PASSES at 1.53× the bar.**
* The lattice captured **53.1 %** of the render floor available to it.
* On Pose it did more: **0.07326 → 0.01138, a 6.4× improvement** for 575 B of section.

So the mechanism is validated: an edge-local correction gated by the generator's own decoded
signed-interface field, costing ~16 KB, removes half the render floor and most of the pose damage.

## Why the object is still refused

**(a) The named mechanism: a render-manufactured argmax floor.**
Attributing every misclassified scorer pixel by whether the generator's own internal partition was
already wrong there:

| quantity | n600 value |
|---|---:|
| `d_seg` at the scorer | 0.02271701389 |
| the generator's OWN partition vs GT | 0.08993511624 (9.0 % of pixels) |
| the pointer's token plane vs GT | 0.0001602427165 (**561× better**) |
| share of seg error that is partition-level | **89.83 %** |
| share that is render floor | 10.17 % |
| **the render floor as `d_seg`** | **0.0023099568 = 272,494 argmax errors at a CORRECT partition** |
| that floor against the 4.0e-4 ceiling | **5.77× over, on its own** |

This is the same shape as `bz2d`'s finding on a different renderer — argmax errors that survive a
correct partition — and the store's correction there travels with it: the "×1.157" ratio was
retracted the same day and the real relation is affine, `argmax ≈ 17,241 + 1.1435 · tokens`, with the
INTERCEPT the transferable part. This object's intercept is 272,494, **15.8×** that one.

**(b) The hop-count split: this is not the pointer's boundary jitter.**
Misclassified pixels binned by four-neighbour hop count to the nearest GT class change:

| class | 0 px | 1 px | 2 px | 3 px | 4 px | >4 px | count |
|---|---:|---:|---:|---:|---:|---:|---:|
| all seg error | 0.377 | 0.133 | 0.070 | 0.042 | 0.033 | **0.345** | 2,679,808 |
| partition-level | — | (0.4944 within 1) | | | | **0.3597** | 2,407,315 |
| render floor | — | (**0.6511** within 1) | | | | 0.2140 | 272,493 |

The render floor behaves like jitter — two thirds within one cell, which is what an edge-local
mechanism reaches, and is why the lattice captured half of it. The partition error does not: **36 %
of it sits deeper than four cells**, a region wearing the wrong class. Set against the store's
reading of the pointer's own residual — *"99.58 % are one-pixel boundary displacements"* — the two
objects carry different debts, and only one of them is a jitter problem.

**(c) The rate rule.** MAIN's stop condition fired on measured data: over the last 20 epochs (27→46)
the seg surrogate fell **0.3152 %/epoch** against the 0.95 %/epoch continue-threshold, 3.0× below,
and the rate has decayed monotonically across the stage — 0.95 → 0.61 → 0.49 → 0.32. At the current
rate `d_seg` 4.0e-4 arrives at epoch 1321, **6.6× outside** the 200-epoch stage. The independently
pre-registered seg-slope falsifier reads INDETERMINATE throughout (its lever arm never reached
1.25×), which is correct: it refuses to extrapolate, and MAIN's rate rule needs no extrapolation.

**(d) Bytes.** The trained object is **122,778 B against the 122,000 B gate — 778 B over** — because
the trained lattice codes to 16,133 B with 98-99 % of its codes nonzero and saturating at ±127. The
zeroed control is 122,203 B, still 203 B over.

## Ceiling arithmetic, so the closure is a number and not a mood

At `d_seg` 0.02271701 the gate needs **56.8×**. The lattice's own ceiling is the render floor it
targets: removing **all** of it gives 1.113×. It has already taken 53.1 % of that, so what remains to
the mechanism is a further **1.05×**. The remaining 54× lives in the partition, which no RGB
correction reaches, and beneath that sits a 5.77× floor that survives a perfect partition.

## What is NOT closed

* **The mechanism.** An interface-gated local implicit correction is validated on this vehicle:
  +0.001365 `d_seg` and 6.4× on `d_pose` for ~16 KB. It should be carried to any successor generator
  whose render floor is small enough for it to matter.
* **Design B (sparse screened-Poisson fusion) and design C (temporal edge-state)** are untouched by
  this: both were queued behind A and neither was measured.
* **The `se(3)` pose hypothesis.** Its falsifier is built and running; nothing here bears on it.

## A constraint the successor inherits, measured here

Two rungs land at essentially the same scorer-plane RMSE and settle a design question with no fit
between the claim and the rows: `grid_384x512` (smooth resampling error, RMSE 4.544) measures
`d_pose` 0.011817, while `sp384_render_noise_8` (independent per-pixel noise, RMSE **4.340** — 4.5 %
LOWER) measures 0.111126. **Independent noise does 9.40× the Pose damage at matched magnitude.**

For any successor generator this is a constraint, not a curiosity: **its render error must be
SMOOTH.** High-frequency error costs nearly an order of magnitude more Pose per unit of RMSE than
smooth error of the same size, so a representation whose residual is noisy pays a penalty a
representation whose residual is a smooth resampling artefact does not. It argues against
high-frequency corrective machinery in the render path and for low-order, smooth representations —
and it is measured on this vehicle's own scorer, not imported.

## Reactivation criteria

Design A reactivates on a generator whose render floor and partition error both clear the arithmetic
above. Concretely, all three:

1. **A different GENERATOR FORM, not a loss or a schedule.** md3/md4 name this directly — *"the sites
   are scorer-hard for this generator FORM… the born accuracy corner needs a different GENERATOR
   FORM, never a start/schedule/seed"* — and the cure they name is **"a Lane carrier, not a loss."**
   The measurement here agrees from a different direction: 89.8 % of the error is the form's own
   partition, and 36 % of that is region-level.
2. **A measured render floor below 4.0e-4** on the candidate generator, using the decomposition in
   `experiments/ddm_obx2_trainer.py::decompose_seg_error`. This object's is 0.00231. The floor is a
   property of the generator's render, measurable before any lattice is trained, and md4's advice
   applies: *"the free step-0 probe predicts unreachability at 9 in 10 — use it before burning."*
3. **A smooth error spectrum**, by the constraint above: 9.40× at matched RMSE is too large a
   penalty to pay on a term already 5.77× over its ceiling.
4. **A lattice budget that survives training.** 15,840 codes at 8 bits coded to 16,133 B once trained
   (98-99 % nonzero); a successor needs either fewer codes, fewer bits, or an entropy model that
   holds under training, measured by a real coder race and not projected.

## Custody

`/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`. Retained: both ablation
rows with their per-chunk argmax, pose and targets; the n600 decomposition; every stage checkpoint
under its own stage-encoded name; the terminal packets and archives with their repeats. Nothing was
deleted; no live run's files were landed. No Modal call, no contest evaluation, no candidate archive
claim.

The frontier is unchanged: **composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]
(move 44)**.
