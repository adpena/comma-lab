# P0: Does the weight_entropy penalty (λ=15) gradient counter d_seg/pose convergence? — n600 MEASURED

**Date:** 2026-07-15 · **Ledger row:** `p0_weight_entropy_gradient_conflict_20260715` · **git:** f91c71b22f
**Operator hypothesis (2026-07-15):** *"Might the weight entropy penalty be countering convergence
depending on the strength of the signal and gradient? We can probably measure across n600."*
**Metric correction (operator 2026-07-15, binding):** verdict read in the FISHER / reachable-decision
geometry metric (margin = Fisher surrogate, ρ=0.978), Euclidean cosine as BASELINE only.

**Pointer 0.19108 UNMOVED — this is MEANS (a lever-hygiene measurement), not a score row.**

## Verdict — hypothesis NOT confirmed at the measured (strong-signal) scope

At the epoch-25 (strong d_seg signal) regime, the weight_entropy rate penalty at λ=15 is
**ORTHOGONAL and negligible** w.r.t. the d_seg descent — it does NOT counter convergence:

| quantity | Euclidean (BASELINE, n600 exact) | FISHER (AUTHORITY, n96 subset) |
|---|---|---|
| cos(g_we, g_dseg) | **−0.00105** (≈0) | **+0.0435** (orthogonal band) |
| cos(g_we, g_pose) | +0.00216 (≈0) | — |
| rel-norm ‖λ·g_we‖ / ‖g_dseg‖ | **0.00186** | **5.9e-5** |
| rel-norm ‖λ·g_we‖ / ‖g_pose‖ | 0.00983 | — |
| PCGrad neg-projection conflict (of g_dseg) | 0.00026 | **0.0** |

Verdict band: |cos| < 0.05 ⇒ **orthogonal** in BOTH metrics. The decisive quantity is the
**relative magnitude**: at λ=15 the weight-entropy gradient is **0.19%** of the seg-descent step in
weight space, and its induced logit-change is **~1.7e4× smaller** than the seg-descent's in the
decision geometry d_seg actually lives in. A force that is both near-orthogonal AND ~3 orders of
magnitude smaller cannot "counter convergence." The Fisher PCGrad negative-projection is exactly 0
(the tiny alignment is slightly POSITIVE, not conflicting).

**Euclidean-vs-Fisher gap (itself a finding):** the naive Euclidean cos is slightly NEGATIVE
(−0.001, would read as "very-faintly-conflicting"); the decision-geometry Fisher cos is slightly
POSITIVE (+0.044, "very-faintly-aligned"). The sign flips between metrics — a concrete instance of
"Euclidean-orthogonal ≠ decision-geometry-orthogonal." But here the magnitude (rel-norm ≪ 1)
dominates the story in either metric, so the gap does not change the verdict.

verdict_scope: **FORMULATION/INSTANCE** — this refutes the hypothesis at the **strong-signal
regime** (epoch 25, seg_loss ~20.6, an actively-descending d_seg trunk). It does NOT test the
signal-strength-dependence the operator flagged (see §Signal-strength dependence). Single cached
checkpoint, witness-alone render (see §Scope). Not a family-level claim about all regimes.

## Signal-strength dependence — the mechanism (why the operator's instinct is right for the LATE regime)

The weight-entropy gradient magnitude is set by the current weight distribution (fixed at a given θ);
the seg-descent gradient magnitude SHRINKS as d_seg approaches its floor. So
`rel-norm = ‖λ·g_we‖ / ‖g_dseg‖` GROWS as convergence proceeds. At epoch 25 (strong signal) it is
~2e-3 — inert. As `‖g_dseg‖ → small` (near the d_seg floor / a flattened slope), the fixed λ·g_we
becomes relatively larger and could begin to bind. This is exactly the **train-big-compress-small**
logic (#157): the penalty is harmless (and near-inert) during active descent, and only its LATE
influence matters. This late regime is UNMEASURED here (no deep-converged n600 checkpoint on hand);
it is the mechanistically-predicted risk, not a measured one.

## c2 recommendation

- **Strong-signal / active d_seg descent:** always-on λ=15 is harmless — it is not fighting
  convergence (near-orthogonal, ~0.19% step magnitude, 0 PCGrad conflict). No Fisher-PCGrad
  projection is warranted here (conflict = 0).
- **Preferred design = EVENT-GATE weight_entropy (a DSL curriculum lever, design-first under the
  dry-start freeze):** keep it OFF (or low λ) during active d_seg descent — where it is inert, so
  gating off costs ~nothing on d_seg — and turn it ON at the finish/compress phase once the d_seg
  slope flattens. Crossover criterion: d_seg slope flattens OR the live rel-norm
  `‖λ·g_we‖/‖g_dseg‖` crosses a threshold (e.g. > 0.1, ~50× today's value). This captures the rate
  win at the phase where it is free and forecloses the late-regime binding the operator flagged,
  without a per-step PCGrad projection.
- **Fisher-PCGrad projection** is the fallback ONLY if a LATE-regime measurement shows genuine
  antagonism (cos < −0.05 with non-trivial rel-norm). The cheaper event-gate dominates unless that
  antagonism is measured.

This is a **c2 curriculum/lever decision** to fold into the DSL as an event-gated weight_entropy
lever (per "Off is a tracked queue" — register with the crossover event, not a hardcoded default).
Do NOT wire into the live trainer under the dry-start freeze; design-first.

## Method (reproducible, NO-FAKE)

Standalone `experiments/measure_weight_entropy_gradient_conflict.py` (does NOT touch the live trainer).
Cached checkpoint `levelset_n600_witness_20260715T095030Z/levelset_witness_ema_BEST.npz` (epoch 25,
copied to `.omx/tmp/we_conflict_measure/ckpt_ema_BEST.npz`). At FIXED θ:

- `g_dseg = ∇_θ[w_seg·seg_l]`, `g_pose = ∇_θ[w_pose·pose_term]` via the trainer's real
  `make_loss_fn` base_loss (imported, seg_form=unify_tau, τ=1.0, w_seg=100, w_pose=1, hinge=4,
  margin_target=0.5), through the REAL frozen MLX SegNet/PoseNet adapter and the REAL through-R
  render — accumulated as the mean over n600 pairs (the actual step direction).
- `g_we = ∇_θ[λ·rate_term]` via `weight_entropy_rate_term_mlx(model, σ=0.2)`, λ=15 — pair-independent,
  computed once. Added ONCE per step (not P-scaled), matching the trainer.
- **Euclidean:** cosine + rel-norm of the flattened weight-space gradients (n600, exact).
- **Fisher (authority):** each weight direction pulled to LOGIT space, then per-pixel inner product in
  the categorical Fisher metric `g=diag(p)−p pᵀ` (`tac.information_geometry.optimal_metric.
  log_partition_hessian`). `mx.jvp` is unimplemented for Convolution (SegNet), but the uint8-STE
  ROUND lives in the RENDER, not the SegNet — so Δframe = JVP of the render-only map (no conv;
  STE-round jvp == identity ⇒ training-faithful), then Δlogit = central finite-difference of the
  SMOOTH SegNet along Δframe (no STE inside SegNet ⇒ FD faithful). n96 evenly-spaced subset.

Result JSON: `.omx/tmp/we_conflict_measure/result_full.json`.

## Scope caveats (honest)

1. **Witness-alone render** (`render_through_R_mlx`, no seed-island compose). At epoch 25 the
   trainer's lane-band (start-epoch 500) and chroma-boundary (start-epoch 450) terms are OFF, so the
   only omitted surface is the seed-island bulk composite. The witness-alone render IS the witness's
   own d_seg-descent surface — the surface g_we (a pure function of the same weights) acts against —
   so it is faithful for the gradient-DIRECTION question. Seed-compose could shift g_dseg's exact
   value; not expected to change the ~3-order-of-magnitude relative-magnitude verdict.
2. **Single checkpoint, epoch 25 (strong signal).** The late/low-signal regime the operator's
   hypothesis targets is unmeasured (§Signal-strength dependence).
3. **Fisher on n96 subset** (JVP+FD cost); Euclidean on full n600.

## Canonical-equation status

FORMALIZATION_PENDING. The durable measured law — *"weight_entropy at λ=15 is decision-geometry-
orthogonal and ~1e3× sub-dominant to d_seg during active descent; rel-norm grows as ‖g_dseg‖→0"* — is
documented here + in the result JSON + the reproducible script. A full `CanonicalEquation` (with the
signal-strength rel-norm scaling law across regimes) is deferred until a LATE-regime anchor lands to
calibrate the crossover, rather than register a one-point law under the dry-start freeze.
