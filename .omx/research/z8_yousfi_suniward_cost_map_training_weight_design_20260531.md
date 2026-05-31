# Z8 hierarchical-PC: Yousfi S-UNIWARD cost-map as a training-objective weight (DESIGN)

UTC: 2026-05-31. Lane: `lane_z8_hier_pc_full_stack_longrun_20260531`.
Status: DESIGN (next-iteration A/B). Non-promotable `[macOS-MLX research-signal]`
per Catalog #192/#341. `# FORMALIZATION_PENDING:design_memo_for_next_iteration_AB_no_empirical_claim_until_paired_run`

## The empirical motivation (from the real-teacher long run)

The Z8 real-teacher long run (`_full_main`, 600 real pairs, 2000 epochs,
real SegNet KL T=2.0 + real PoseNet pose-MSE, τ-anneal 1.0→0.1, warmup 50 +
cosine LR) produced the canonical honest per-axis split:

- **pose** collapsed 104.6 → ~0.1 (the categorical hierarchical-PC renderer +
  real PoseNet teacher essentially SOLVE the pose axis; `sqrt(10·0.1)` ≈ 1.0
  contribution).
- **seg** descends slowly 6.45 → ~3.0 and is the **dominant residual** — the
  STRUCTURAL CEILING. Per the contest score `100·d_seg + sqrt(10·d_pose) +
  rate`, seg is the binding constraint.

This is the known finding, not a failure: SegNet (`smp.Unet('tu-efficientnet_b2')`)
responds ONLY to class-boundary argmax flips; its stride-2 stem is blind below
(256, 192) and indifferent to perturbations in class INTERIORS. A
reconstruction loss spends capacity UNIFORMLY across all pixels — including the
class interiors where SegNet cannot see — so it under-allocates to the thin
class-boundary regions that ARE the entire seg signal.

## The canonical Yousfi/Fridrich lever

Fridrich inverse-steganalysis principle #1 (UNIWARD): errors in textured
regions are undetectable; the detector responds to perturbations at
low-texture / boundary structure. The canonical "what would Yousfi do" for the
seg axis is to **concentrate the representation's capacity toward SegNet-sensitive
class boundaries and spend nothing in class interiors where the detector is
blind**.

The tractable, $0/MLX implementation is the **S-UNIWARD texture cost-map as a
per-pixel weight on the reconstruction MSE** — NOT the scorer-finite-difference
sensitivity map (which is intractable, see below).

## API verified live (SEARCH-BEFORE-EDIT)

- `tac.uniward_delta.compute_uniward_cost_map(frames_bchw, *, sigma=1e-4) ->
  (B,H,W)` — directional-Haar wavelet S-UNIWARD cost on ALREADY-RENDERED
  frames. PURE IMAGE operation, NO scorer forward pass. "High in textured
  regions" (raw `_uniward_cost`, un-inverted). This is the tractable primitive.
- `tac.substrates.z8_hierarchical_predictive_coding.scorer_sensitivity_map`:
  - `uniform_sensitivity_map_for_level` — current M7 default (uniform = no
    concentration; the baseline the run above used implicitly via recon-MSE).
  - `yousfi_uniward_finite_difference_sensitivity_map` — STUB
    (`raise EmpiricalSensitivityMapNotYetLandedError`): the per-pixel
    ±ε scorer-finite-difference is intractable (~7.2B scorer forwards). This is
    DEFERRED-pending-paid-GPU per Catalog #307 IMPLEMENTATION-LEVEL. It is NOT
    the lever proposed here.
  - `empirical_sensitivity_map_from_master_gradient` — the analytic alternative
    (backprop scorer gradient → per-pixel) once a paired GPU run lands.

KEY DISTINCTION the SEARCH-BEFORE-EDIT discipline caught: the operator's ask
("UNIWARD cost-map as a TRAINING-OBJECTIVE WEIGHT") is the TRACTABLE
`compute_uniward_cost_map` (image-only), NOT the deferred scorer-finite-difference
`yousfi_uniward_finite_difference_sensitivity_map`.

## The wire-in (next-iteration A/B)

The active loss surface is the shared harness `score_aware_loss`
(`src/tac/substrates/_shared/mlx_score_aware/loss.py`):

```python
mse_0 = mx.mean((rgb_0 - gt_0) ** 2)   # line 104 — UNIFORM per-pixel MSE
mse_1 = mx.mean((rgb_1 - gt_1) ** 2)
recon = mse_0 + mse_1
```

The cost-map wire-in replaces the uniform mean with a cost-weighted mean:

```python
# w = normalized S-UNIWARD cost map on the GROUND-TRUTH frame (target),
# computed ONCE at decode time (not per-step), broadcast (B,H,W,1).
# Invert convention for SEG: we want HIGH weight at class BOUNDARIES (low
# texture / high structure) where SegNet is sensitive, LOW weight in textured
# interiors where it is blind. The raw _uniward_cost is "high in textured" so
# the seg-boundary weight is (1 - normalized_cost) OR a class-boundary edge map
# from the SegNet teacher argmax. Both are pre-computable from the FROZEN
# target + teacher (no per-step scorer forward).
mse_0 = mx.mean(w * (rgb_0 - gt_0) ** 2)
```

Two candidate weight sources (the A/B):

1. **A = S-UNIWARD inverse-texture weight** `w = normalize(1 - uniward_cost(gt))`
   — concentrates on low-texture / boundary structure (the canonical UNIWARD
   inverse for a DETECTOR-FACING objective). Cheap; image-only.
2. **B = SegNet-teacher class-boundary edge map** `w = dilate(edges(argmax(
   segnet_teacher(gt))))` — concentrates DIRECTLY on the class boundaries the
   seg distortion is computed over. Requires ONE teacher forward on the frozen
   target (already built in `_full_main`); zero per-step cost.

## Why this is DESIGN, not done-this-run

1. `score_aware_loss` is a SHARED harness consumed by ~15 substrates; adding a
   `recon_pixel_weight` channel to `RendererBundle` is a cross-substrate change
   that needs its own UNIQUE-AND-COMPLETE-PER-METHOD review + sister-substrate
   regression (Catalog #290). It is larger than one phase-1 run.
2. The active loop ALREADY carries the dominant convergence driver: the REAL
   SegNet KL + REAL PoseNet pose-MSE scorer-bound gradient (per the DreamerV3
   cross-family lesson, the Hinton-distilled scorer-bound gradient is the
   dominant in-training driver). The cost-map is a COMPLEMENTARY capacity-
   allocation PRIOR on the recon term, not a replacement for the scorer
   gradient. The baseline (this run) must land first so the A/B has a control.
3. Per the operator: "if wiring this fully is too large for one run, at minimum
   DESIGN the wire-in + run the baseline." This memo IS that design; the run
   above IS that baseline.

## Predicted effect (grounded, NOT claimed)

Per the contest score `100·d_seg + ...`, a 1% relative seg improvement is worth
~0.01 score points at the seg operating point — a LARGE lever because seg is
the structural ceiling. The cost-map concentrates capacity where SegNet argmax
flips happen, so the predicted DIRECTION is seg-distortion DOWN. Magnitude is
UNKNOWN until the paired A/B run lands; per Catalog #324 no predicted band is
asserted without post-training Tier-C validation. The Dykstra-feasibility note:
the cost-map re-allocates a FIXED capacity budget (it does not add bytes), so it
trades interior-recon fidelity (PoseNet-irrelevant once pose is solved) for
boundary fidelity (SegNet-relevant) — a feasible intra-budget projection, not
an additive claim.

## Reactivation criteria (the A/B run)

1. Land `recon_pixel_weight` channel on `RendererBundle` + sister-substrate
   regression (Catalog #290 + the 15-substrate harness test suite stays green).
2. Run A (S-UNIWARD inverse-texture) + B (SegNet class-boundary edge) + the
   uniform baseline (this run) at matched epochs/seed; compare per-axis seg.
3. Promote the winner only via paired Linux x86_64 + NVIDIA exact auth-eval
   (Catalog #192/#246); macOS-MLX is research-signal only.

## Cross-references

- `feedback_dreamer_v3_rssm_v2_tau_anneal_*` (the τ-anneal + real-teacher
  cross-family lesson this baseline applies).
- CLAUDE.md "Fridrich inverse steganalysis" + "SegNet vs PoseNet importance".
- `tac.uniward_delta.compute_uniward_cost_map` (the tractable primitive).
- `scorer_sensitivity_map.yousfi_uniward_finite_difference_sensitivity_map`
  (the DEFERRED scorer-finite-difference; NOT this lever).
