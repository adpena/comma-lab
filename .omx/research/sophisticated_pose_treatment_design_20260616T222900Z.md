# Sophisticated pose treatment for the bind-all small-basis arm_b — design + build

**UTC:** 2026-06-16T22:29:00Z
**Author:** subagent `sophisticated-pose-treatment-20260616`
**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. No GPU touched, no running job touched. $0/CPU only.
**Scope:** design + build a more optimal POSE TREATMENT for the bind-all small-basis arm_b run — beyond the
current FiLM-v2 + trunk-stopgrad + rgb0-pose-trainable + pose-every-epoch carrier. Default-OFF → byte-identical.

---

## 0. The problem (the failure this fixes)

The base_ch20 small basis byte-closes near the only clearly-sub-0.15 vehicle (rate-headroom floor ~0.111;
roadmap `272cdaa56`). At the n600 basin, `d_pose ~ 0.00034`. The score pose-term is `sqrt(10·d_pose)`, so its
marginal `∂S/∂d_pose = 5/sqrt(10·d_pose) ≈ 85.7` at the basin — i.e. ~86% of `∂S/∂d_seg = 100`. Pose is NOT
negligible at this operating point; it is the second-largest marginal score lever after d_seg.

**The observed regression:** the warm-throttled run drifts `d_pose` from 0.00034 → ~0.00049 — a `ΔS ≈
sqrt(10·0.00049) − sqrt(10·0.00034) = 0.0700 − 0.0583 = +0.0117` regression. That is a large fraction of the
gap to sub-0.15. The current carrier (FiLM-v2 residual on rgb_0 + trunk-stopgrad making `∂d_seg/∂pose=0`
exactly + rgb0-pose-trainable + pose-every-epoch) is a sound DECOUPLING (it stops pose from harming seg) and a
sound CADENCE, but it does NOT:

1. **balance the pose-vs-seg LOSS WEIGHT to the score-optimal (equimarginal) operating point** — `pose_weight`
   is a hand-set per-stage Lagrangian coefficient (the vendored schedule value, optionally oomph-scaled on the
   SEG side only). There is no controller that sets `pose_weight` so the pose-axis and seg-axis have EQUAL
   marginal SCORE impact. (Lever A.)
2. **exploit the MEASURED pose-sensitive geometry** — FiLM-v2's rgb_0 residual is a crude, dense carrier of "the
   pose-conditioned frame-0". The contest pose map is exactly 6-dimensional (`d_pose` = MSE on first 6 of 12
   PoseNet dims); the local frame→pose map is a 6×N Jacobian whose row space (≤6-dim) is the POSE-SENSITIVE
   subspace and whose (N−r)-dim complement is the POSE-NULL. Reconstruction/d_seg freedom steered INTO the
   pose-null cannot move d_pose. FiLM-v2 does not know this geometry. (Lever B.)
3. **weight the 6 pose dims by their score sensitivity** — the pose loss is a flat `MSE(pose_pred6, target6)`.
   The 6 dims have different per-dim variance/score-sensitivity; the optimal-teacher design (memory
   `feedback_optimal_teacher_and_sensitivity_tools_landed_20260531.md`) is per-dim Mahalanobis reweight (AIL).
   And dims `k >= out//2` of the 12-dim head are UNSCORED — the loss already slices `[:6]`, but the per-dim
   WEIGHTING of the 6 scored dims is uniform. (Lever C.)

---

## 1. What I REUSE (SEARCH-FIRST; named, not reinvented)

| Reused machinery | File (absolute) | Used for |
|---|---|---|
| Pose-sensitive subspace SVD + pose-null projection (#80 POSE CRUX) | `src/tac/boundary_math/posenet_subspace_spectrum.py` — `measure_pose_subspace_spectrum`, `project_onto_pose_null`, `participation_ratio`, `expected_isotropic_null_fraction` | Lever B measurement + projection |
| Per-pixel pose Jacobian saliency (#61) | `src/tac/boundary_math/posenet_jacobian_saliency.py` — `compute_posenet_pixel_saliency`, `saliency_to_weight_map` | Lever B saliency context (reported, not re-derived) |
| Frame1 joint safe cone + pose Jacobian per pixel (#35) | `src/tac/optimization/frame1_joint_safe_cone.py` — `measure_posenet_frame1_jacobian` | Lever B alt frame-1 measurement (reported) |
| Mahalanobis per-dim pose reweight | `src/tac/optimization/joint_p18_p19_waterfill.py` — `mahalanobis_pose_jacobian_norm`, `pose_ail_gain` | Lever C per-dim weighting math |
| Contest score calculus + marginal derivatives | `src/tac/score_composition/__init__.py` — `compose_score_from_axes`, `CANONICAL_*` constants | Lever A equimarginal `∂S/∂d_pose`, `∂S/∂d_seg` |
| Differentiable scorers + GT decode | `src/tac/scorer.py::load_differentiable_scorers`, `src/tac/differentiable_eval_roundtrip.py::patch_upstream_yuv6_globally`; GT via `frame_utils.yuv420_to_rgb` (canonical) | Lever B real-PoseNet Jacobian on real frames |
| Current carrier | `src/tac/torch_vehicle/pose_film_v2.py` (`PoseFiLMHNeRVWrapperV2`), driver `_split_by_head_backward`, `_non_film_grad_params`, Config `pose_film_*` | The carrier these levers refine, NOT replace |
| Driver pose path + StageSpec | `src/tac/torch_vehicle/driver.py` (`_split_by_head_backward`, `_overlay_oomph`-style stage replace), `src/tac/torch_vehicle/curriculum.py::StageSpec` (`seg_weight`/`pose_weight`) | Lever A/C wire-in |

**The pose calculus** (verified in `score_composition`): `S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489`.
`∂S/∂d_seg = 100` (constant). `∂S/∂d_pose = 5/sqrt(10·d_pose)` (operating-point dependent). At the basin
(`d_pose=0.00034`) the pose marginal is ≈85.7, ≈0.857× the seg marginal — the memory anchor.

---

## 2. The three levers (synthesize → build)

### Lever A — Equimarginal `λ_pose` weight controller (the WEIGHT analogue of the APGC cadence)

**Principle.** The Lagrangian the driver minimises is `L = w_seg·seg_l(F) + w_pose·pose_l(F)`. Crucially
`pose_l = sqrt(10·pose_mse + 1e-12)` is **already the contest pose-term in score units** — so `w_pose·∂(pose_l)/∂θ`
is the pose objective's contribution to `∂S/∂θ` scaled by `w_pose`, and the sqrt already bakes in the
`5/sqrt(10·d_pose)` marginal via the chain rule (`∂pose_l/∂pose_mse = 5/sqrt(10·pose_mse)`). The seg surrogate
`seg_l` is a PROXY for `d_seg` (CE or soft_cosine), not literally `100·d_seg`; `w_seg` calibrates its
score-units.

**The equimarginal rule.** Two axes are at the score-optimal balance when a marginal training step moves each
by the SAME amount of SCORE. The measurable, NO-FAKE version uses the per-axis frame-cotangent norms the driver
ALREADY computes in `_split_by_head_backward`: `cot_seg = ∂(w_seg·seg_l)/∂F` and `cot_pose = ∂(w_pose·pose_l)/∂F`.
Their L2 norms `‖cot_seg‖`, `‖cot_pose‖` are the per-axis "pull" on the shared frame tensor in their current
weighting. The equimarginal target is `‖w_pose·cot_pose_unit‖ ≈ ρ·‖w_seg·cot_seg_unit‖` where `ρ` is the
desired score-marginal ratio (default `ρ = 1.0` for true equimarginal; the operator may bias toward seg with
`ρ<1`). Because the pose term is already in score units and the seg term is calibrated by `w_seg`, balancing
the cotangent norms balances the per-step SCORE pull.

**Controller (closed-loop, EMA-smoothed, deadbanded — mirrors APGC's structure, NOT its cadence):**
```
ratio_t      = ‖cot_pose‖ / max(‖cot_seg‖, eps)         # measured per-axis score-pull ratio at w_pose=current
ratio_ema    = decay·ratio_ema + (1−decay)·ratio_t      # smoothed (decay default 0.9)
if |ratio_ema − ρ| > tol·ρ:                              # outside deadband (tol default 0.15)
    w_pose  ← clamp(w_pose · (ρ / ratio_ema), w_pose0·lo, w_pose0·hi)   # multiplicative correction
```
The multiplier is clamped to `[lo, hi]·w_pose0` (default `[0.25, 4.0]`) so the controller can never run away.
`w_pose0` is the stage's base `pose_weight`. **Default OFF** (`pose_equimarginal_enabled=False`) →
`w_pose` is the unmodified stage value every epoch → byte-identical.

**Why this is the score-optimal weight, not a hand-set constant.** It TRACKS the measured `5/sqrt(10·d_pose)`
operating point: as `d_pose` falls, `∂pose_l/∂pose_mse` rises, `‖cot_pose‖` rises, the controller's measured
`ratio_ema` rises, and it AUTO-LOWERS `w_pose` to hold the equimarginal point — exactly the behaviour the
score calculus dictates (a smaller d_pose has a higher marginal so needs less explicit weight to stay balanced).
A hand-set constant cannot follow this curve. The controller IS the equimarginal solve evaluated online.

**Built as:** `tac.torch_vehicle.equimarginal_pose_weight.EquimarginalPoseWeightController` (pure controller,
no torch state beyond the EMA scalar) + a default-OFF `cfg.pose_equimarginal_*` flag set consumed in
`_split_by_head_backward` to scale `spec.pose_weight` per epoch. Proof: a unit test feeds known cotangent
norms and asserts the multiplier moves `w_pose` toward `ρ`; a basin measurement reports the real cotangent-norm
ratio at the basin so the equimarginal `w_pose` is a MEASURED value, not a guess.

### Lever B — Jacobian-aligned pose-null carrier measurement (the orthogonal-solve, deepest lever)

**Principle.** Reuse `measure_pose_subspace_spectrum` to get the 6×N Jacobian SVD on the REAL PoseNet at the
basin (GT frames via `frame_utils.yuv420_to_rgb`). The row space (rank `r ≤ 6`, effective-dim `< 6`) is the
pose-sensitive subspace; the complement is the pose-null. The carrier design goal: steer the rgb_0 head's
reconstruction/d_seg freedom INTO the pose-null so improving rgb_0 (frame_0 fidelity, which helps the pair's
joint reconstruction and thus indirectly d_seg-pair coherence) cannot move d_pose; carry the minimal
pose-sensitive component (the ≤6 scalars in the row space — exactly the stored pose's job).

**What I build (measurement + projection hook, honest about integration boundary):**
1. **A basin pose-null ATLAS measurement** (`experiments/measure_pose_null_atlas.py`, $0 CPU, uses the real
   differentiable PoseNet): for a sample of pairs, measure `effective_dim`, `rank`, `sigma` spectrum,
   `expected_isotropic_null_fraction`, and the empirical null-energy-fraction of the FiLM-v2 residual's
   actual frame perturbation (`f0_film − f0_clean` projected onto the pose-null). This QUANTIFIES whether the
   current FiLM-v2 carrier is already mostly pose-null (good — it means the residual is cheap on pose) or
   leaking into the pose-sensitive subspace (bad — the regression source). This is the diagnostic that grounds
   the carrier design on the MEASURED geometry instead of asserting it.
2. **A default-OFF pose-null projection HOOK** (`tac.torch_vehicle.pose_null_projection_hook`): given a measured
   `PoseSubspaceSpectrum` (or a cached basis), project a frame-residual onto the pose-null before it is applied,
   so the carrier's contribution to rgb_0 is pose-null by construction. The hook is a thin, tested wrapper over
   `project_onto_pose_null`; it is wired as an OPTIONAL pre-apply step on the FiLM residual (default OFF →
   identity → byte-identical). The FULL training integration (projecting the residual every step at the working
   resolution, with the per-pair basis cached) is documented as the wire-in for the driver; the hook + the basin
   atlas make the geometry measurable and the projection callable NOW.

**Honest integration boundary.** Lever B is **measurement + design-hook**, not full-train-integrated. The atlas
measures the real pose-null on the basin; the projection hook is a tested callable; the per-step
project-the-residual training wire-in is specified (where in `_forward_with_film` it goes, what basis it needs,
the resolution match) but NOT switched on in the live driver this landing (it requires a per-pair cached basis
the basin run does not yet emit — that is the named next step). This avoids the "L1 scaffold + bytes without
overlay" trap: the hook is default-OFF and the atlas is a pure diagnostic; neither claims a score effect.

### Lever C — Per-dim pose Mahalanobis / AIL weighting + score-irrelevant-dim zeroing

**Principle.** Reuse the optimal-teacher design + `mahalanobis_pose_jacobian_norm`. The pose loss is currently
`MSE(pose_pred6, target6)` — uniform over the 6 dims. The score-optimal pose loss weights each scored dim by
its inverse per-dim variance (Mahalanobis) and the AIL gain `5/sqrt(10·d_pose)` (Saputra 2019), and zeroes any
dim with negligible score sensitivity. The 12-dim head's dims `k >= 6` are already dropped by the `[:6]` slice;
Lever C adds the per-dim WEIGHTING of the 6 surviving dims: `pose_mse_weighted = mean( w_k · (pred_k − tgt_k)^2 )`
with `w_k = inv_var_k / mean(inv_var)` (renormalised so the default-uniform `w_k=1` is byte-identical) and an
optional hard zero for dims whose measured Jacobian-row energy is below a tolerance.

**Built as:** a default-OFF `cfg.pose_dim_weights` (a length-6 tuple; `None` → uniform → byte-identical) consumed
in BOTH pose paths (`_split_by_head_backward` and the non-split fused path) as a per-dim weight on the squared
error before the mean + sqrt. The weights are MEASURED on the basin (per-dim target variance + per-dim Jacobian
row energy from the real PoseNet) by a small helper, NOT hand-set. Small, clean, fully integrated.

---

## 3. Predicted bands (Dykstra-feasibility / first-principles grounded; #296)

These are `[contest-CPU advisory]` predictions; the arm_b run + G3 exact-eval measure the real score win.

- **Lever A (equimarginal `w_pose`).** First-principles bound: the regression is `ΔS ≈ +0.0117` from d_pose
  drift 0.00034→0.00049. Holding the equimarginal point keeps the pose pull from being starved during refinement
  (the seg-oomph stages currently scale `w_seg` up 1.5× with NO matching pose adjustment → the controller
  RESTORES the balance the oomph overlay breaks). Predicted recovery of **0 to the full +0.0117** depending on
  how much of the drift is weight-starvation vs capacity; conservatively `[−0.012, 0.0]` ΔS (a recovery, sign
  toward lower S). The Dykstra-feasibility framing: the equimarginal point is the projection of the gradient
  onto the `‖cot_pose‖/‖cot_seg‖ = ρ` constraint set; the controller is the alternating-projection step.
- **Lever B (pose-null carrier).** The atlas measures `effective_dim < 6` (≤6 by construction; the row space is
  a 6×N matrix) and `expected_isotropic_null_fraction ≈ (N−r)/N ≈ 1−6/589824 ≈ 0.99999`. The lever's value is
  the GAP between isotropic and null-confined error: if the FiLM residual is measured to leak `f_sens > 0` into
  the pose-sensitive subspace, projecting it out removes exactly that leak's d_pose cost. Predicted band is
  measurement-gated (the atlas produces it); pre-measurement bound `[−0.012, 0.0]` (cannot exceed the full
  drift recovery; a pose-null carrier is a strict superset of FiLM-v2's accidental null fraction).
- **Lever C (per-dim weighting).** Reweighting 6 dims is a small re-allocation within an already-small pose
  term; first-principles it improves the WORST scored dim at the expense of the best, lowering the MSE-mean only
  if the per-dim sensitivities are unequal (measured on the basin). Predicted `[−0.003, 0.0]` ΔS — the smallest
  lever, but free (0 bytes, 0 d_seg by construction since SegNet reads only rgb_1).

Combined (non-additive; they touch the same pose term) predicted `[−0.012, 0.0]` ΔS — i.e. recover the warm
regression toward the basin's 0.00034 d_pose, NOT a new floor below it.

## Observability surface (#305)

1. **Inspectable per layer:** Lever A logs `(‖cot_seg‖, ‖cot_pose‖, ratio_t, ratio_ema, w_pose_effective)` per
   epoch; Lever B atlas logs per-pair `(rank, effective_dim, sigma_top3, null_frac_of_film_residual)`; Lever C
   logs the 6 per-dim weights + zeroed dims.
2. **Decomposable per signal:** the equimarginal ratio decomposes the per-step score pull into seg vs pose; the
   pose-null atlas decomposes the residual into sensitive vs null energy; the per-dim weights decompose d_pose
   into the 6 dims.
3. **Diff-able across runs:** all three emit JSON; two runs' `w_pose` trajectories / null fractions / dim
   weights are byte-diffable.
4. **Queryable post-hoc:** atlas → JSON; controller telemetry → the driver summary; dim weights → config echo.
5. **Cite-able:** every measurement carries `(commit, basin path, n_pairs, seed, compute_path="cpu_torch")`.
6. **Counterfactual-able:** the projection hook answers "what if the residual were pose-null?" without retrain
   (project the measured residual + recompute d_pose).

## Canonical-vs-unique decision per layer (#290)

| Layer | Decision | Rationale |
|---|---|---|
| Pose-null measurement | ADOPT_CANONICAL | `measure_pose_subspace_spectrum`/`project_onto_pose_null` are the canonical #80 machinery; reuse verbatim. |
| Per-dim Mahalanobis math | ADOPT_CANONICAL | `mahalanobis_pose_jacobian_norm` is canonical; reuse. |
| Score calculus | ADOPT_CANONICAL | `score_composition` canonical constants + `compose_score_from_axes`. |
| Equimarginal controller | FORK_PRINCIPLED | No canonical WEIGHT controller exists (APGC cadences, does not reweight). New, but mirrors APGC's structure (EMA + deadband + clamp) so it is the minimal new surface. |
| Pose loss per-dim weighting in driver | FORK_PRINCIPLED | The driver's pose path is substrate-specific (split-by-head + fused); the per-dim weight must be threaded into BOTH; the canonical Mahalanobis helper supplies the math, the threading is substrate-unique. |
| Carrier (FiLM-v2) | ADOPT_CANONICAL (refine, not replace) | Reuse `PoseFiLMHNeRVWrapperV2`; the pose-null hook is an optional pre-apply, not a new carrier. |

## 18-shared-assumption profile (D4) — the score-relevant ones

- **eval_roundtrip / differentiable yuv6:** ADOPT_CANONICAL (the pose Jacobian REQUIRES the differentiable
  yuv6 patch; `load_differentiable_scorers`/`patch_upstream_yuv6_globally` is HARD-EARNED — without it the
  Jacobian is severed = zero, and `measure_pose_subspace_spectrum` fail-closes on exactly that).
- **CPU authority for pose / NO MPS:** ADOPT_CANONICAL (HARD-EARNED — MPS corrupts PoseNet 23×; all Jacobian +
  d_pose measurement is CPU-torch).
- **GT via `frame_utils.yuv420_to_rgb`:** ADOPT_CANONICAL (HARD-EARNED — PyAV rgb24 manufactures ~100× phantom
  pose).
- **`pose_l = sqrt(10·pose_mse)` IS the score pose-term:** ADOPT_CANONICAL (verified against
  `upstream/evaluate.py:92` + `score_composition`).
- **d_pose = MSE on first 6 of 12 dims:** ADOPT_CANONICAL (verified upstream/modules.py; the `[:6]` slice).
- **FiLM-v2 residual is mostly pose-null:** UNCLEAR_NEEDS_EMPIRICAL → this is exactly what Lever B's atlas
  MEASURES (the design does not assume it; it tests it).
- **6 scored pose dims are equally score-sensitive:** UNCLEAR_NEEDS_EMPIRICAL → Lever C's per-dim weights are
  measured, not assumed; uniform is the default.

## NO-FAKE proof obligations (per lever)

- **A:** unit test feeds known cotangent norms → asserts the multiplier moves `w_pose` toward `ρ` and clamps;
  default-OFF test asserts `w_pose` unchanged (byte-identical). Basin measurement reports the REAL
  `‖cot_pose‖/‖cot_seg‖` ratio so the equimarginal `w_pose` is measured.
- **B:** atlas measures the REAL 6×N PoseNet Jacobian on REAL GT frames (fail-closes on severed gradient);
  the projection hook test builds a delta INSIDE the measured row space → asserts `null_frac ≈ 0`, and an
  isotropic delta → asserts `null_frac ≈ (N−r)/N` (the load-bearing proof from the #80 test suite, reused).
- **C:** test with non-uniform dim weights asserts the weighted pose-mse differs from uniform in the expected
  direction; uniform weights (`None`) test asserts byte-identical to the current path; weights measured from
  real per-dim target variance.

## Recommended arm_b pose-treatment flags

```
--pose-film-v2 --pose-film-trunk-stopgrad --pose-film-rgb0-pose-trainable   # the carrier (existing)
--pose-equimarginal --pose-equimarginal-rho 1.0                              # Lever A (new, the headline)
--pose-dim-weights-auto                                                      # Lever C (new, free)
# Lever B: run experiments/measure_pose_null_atlas.py on the basin FIRST (diagnostic); the projection
# hook stays OFF this landing (full train-integration is the named next step).
```
