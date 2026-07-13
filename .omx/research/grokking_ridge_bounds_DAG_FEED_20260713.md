# DAG FEED — ridge-grokking bounds autopsy (2026-07-13)

## FEED-GROKKING-RIDGE-BOUNDS-20260713

```yaml
feed_id: FEED-GROKKING-RIDGE-BOUNDS-20260713
lane_id: lane_grokking_ridge_bounds_20260713
research_only: true
authority: macOS-CPU NumPy-fp32 and frozen CPU SegNet; NON-PROMOTABLE
pointer_delta: zero
source_run_mutated: false
verdict: FEATURE_POVERTY_FORMULATION_NOT_UNDERTRAINED
```

### Source law

Xu, Vardi, and Safran, arXiv:2601.19791v3, Theorem 4.2 / Equations 6–7:
the delayed mode is the random-initialization component orthogonal to the
empirical data span, which decays as `(1-eta*lambda)^t`; its quantitative lower
bound assumes a realizable teacher, `m>n`, nonzero Gaussian initialization, and
small enough weight decay/step size.

### Actual Round-2 mapping

```yaml
feature_dimension_m: 31
scalar_training_rows_n: 1474560
m_minus_n: -1474529
numerical_rank: 28
empirical_null_dimension: 3
initialization_variance_nu2: 0.0
ridge_lambda: 3.2247040271759033
learning_rate_eta: 0.20673732459545135
eta_times_lambda: 0.6666666831905239
one_minus_eta_lambda: 0.3333333168094761
null_mode_half_life_steps: 0.6309297251026525
steps_for_1e7_null_shrink: 15
paper_theorem_applicability: REFUSED
```

The paper lower bound cannot be numerically instantiated: `m>n` and nonzero
random initialization both fail. The exact fixed-quadratic law is stronger for
this instance: `W0=0` makes the slow null component exactly zero, and the
spectral-ridge update contracts at measured `gamma=0.3333333461703458`.

### $0 real-n600 refit anchor

Receipt:
`experiments/results/grokking_ridge_round2_refit_20260713/measurement_receipt.json`
SHA-256 `fc8c79ef82d829f05cee79890c9b5d237e12d84e92ec83982f182de15ecb6b4d`.

```yaml
source_train_exact_labels_reused: 480
new_real_heldout_teacher_calls: 120
synthetic_data_used: false
gd15:
  weights_bit_equal_to_committed: true
  objective_gap: 0.0
  parameter_residual: 2.2703186949787158e-15
  heldout_cosine: 0.001415793417951615
  heldout_relative_l2: 1.000001870577624
gd150:
  max_abs_weight_delta_from_gd15: 8.881784197001252e-16
  heldout_cosine: 0.0014157934642280926
  cosine_delta_from_gd15: 4.627647760226061e-11
ridge_exact_optimum_ladder:
  ratios_to_data_lmax: [0, 1e-6, 1e-4, 1e-2, 1e-1, 1, 10]
  best_cosine: 0.007690592649965529
  best_cosine_ratio: 1e-6
  relative_l2_at_best_cosine: 1.0007586082750441
pointer_moved: false
```

### Decision edges

1. `GD15 -> GD150` changes heldout cosine by only `4.63e-11`, while the
   original weights are reproduced bit-for-bit and the fixed objective has
   zero measured gap -> `UNDERTRAINED` is falsified for the registered head.
2. Exact optima over seven ridge scales reach at most cosine `0.0076906` and
   relative-L2 `1.000759` -> the wall survives removal of both step-count and
   original-ridge confounds -> `FEATURE-POVERTY` at the fixed 31-feature linear
   formulation scope. Nonlinear/richer features remain open and are Round-3's
   required path.
3. Round-3 must use the delay guard before killing a feature set: zero init;
   declared ridge ladder; `eta=2/(mu+L)` for each positive-ridge arm; run until
   a terminal gradient/residual certificate or use the exact optimum; then
   judge heldout fidelity.
4. The paper's stage-delay bound does not transfer as an advance authority for
   the nonconvex, stage-changing, AdamW/Muon/EMA witness. A decay clock may be
   logged only as telemetry after measuring a stable Jacobian-null component;
   it cannot fire #315/#344 by itself.
5. No witness lever falls out; therefore no curriculum-candidate pool row is
   admitted. The only closed object is a surrogate-evaluation guard.
6. STEPS economics: the guard/tuning changes no teacher calls per step, so its
   standalone break-even is any measured `r_grok>1`. If composed with SPS's
   optimistic unmeasured overhead, break-even is
   `r_grok*r_sps > k_T*f_T+k_W*(1-f_T)=1.05`; a second scorer VJP raises the
   right-hand side toward `1.95–2.00`. No witness-stage reduction is measured.

### Triality

- DAG: this FEED.
- Equation:
  `tac.canonical_equations.grokking_ridge_undertraining_disambiguation_20260713`.
- DSL: N/A with rationale — research-only Round-3 evaluation guard; no live
  trainer argv and no stage-advance actuator.
- Pool: no row; no admissible witness lever emerged.

### Verdict scope / reactivation

`FEATURE_POVERTY` is scoped to the fixed 31-feature linear chart and the
measured ridge ladder on the three-checkpoint real-n600 replay. It does not kill
frozen-stem features, RFF lifts, margin-field targets, nonlinear heads,
on-policy replay, or another representation. Direct plateau transfer is
`NO_GO` at theorem-to-witness scope; reactivate only after a fixed-stage local
linearization measures a stable null projector, its nuisance amplitude, and a
heldout/evaluator-calibrated threshold.
