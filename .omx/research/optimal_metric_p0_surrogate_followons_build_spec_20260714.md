# Optimal-metric P0 surrogate follow-ons — build contract

**Lane:** `lane_optimal_metric_p0_surrogate_followons_20260714`  
**Authority:** `$0`, local read/build/verify only; no heavy launch, paid dispatch,
live-run mutation, score claim, or pointer movement.  
**Metric law:** `argmax_native_vjp_fidelity_v1`.

## Numerical contract

For teacher and candidate input cotangents `g_T,g_S`, renderer `R`, and
content-described positive-semidefinite preconditioner `M`:

```text
h_T = J_R^T g_T
h_S = J_R^T g_S
<g_T,g_S>_(R,M) = h_T^T M h_S
rho = dot_M / (||h_T||_M ||h_S||_M)
eta = dot_M / ||h_T||_M^2
```

`h` is a cotangent. A decision Fisher first defines a tangent metric
`G=A_theta^T A_theta`; the optimizer/preconditioner acting on `h` is the
damped inverse or Moore–Penrose inverse of `G`. Using `G` directly on
cotangents reverses the natural geometry.

NumPy-fp32 owns array construction and content hashes; scalar reductions use
float64. MLX-fp32 is an optional advisory parity path and never score or
selection authority.

For one sealed state with fixed PSD `M`, this is exactly the Hessian geometry
of the quadratic generator `F_M(h)=1/2 h^T M h`, with
`B_FM(h_S:h_T)=1/2 (h_S-h_T)^T M (h_S-h_T)`. A state-dependent Fisher does not
thereby prove one global finite Bregman divergence; that needs an integrability
or path certificate.

For centered categorical logits, `F(z)=logsumexp(z)`, `grad F=softmax(z)`, and
`B_F(z_S:z_T)=KL(p_T||p_S)`. The cheap dual-Euclidean distance
`||p_T-p_S||_2` is exact for the squared-Hessian `H_F^2` metric. It needs no
Fisher solve in that geometry, but it is not the finite KL, the
Fisher-natural cotangent metric, or VJP-alignment `rho`.

## Canonical APIs

Module: `src/tac/scorer_surrogate/vjp_fidelity.py`.

- `PullbackPreconditioner`
- `renderer_pullback_numpy_fp32`
- `identity_pullback_preconditioner_numpy_fp32`
- `categorical_fisher_preconditioner_numpy_fp32`
- `margin_fisher_preconditioner_numpy_fp32`
- `reachable_decision_geometry_summary`
- `reachable_pullback_geometry_summary`
- `weighted_pullback_summary_mlx_advisory`
- `nullspace_reweighting_disentanglement`
- `select_preconditioner_by_through_r_agreement`
- `categorical_bregman_geometry_summary`
- `extended_kl_mc_summary_from_log_ratios`

Sampled KL accepts only explicit iid-from-reference custody and uses the
pointwise-nonnegative extended integrand
`log(p/q) + q/p - 1`. A naive finite-sample mean of `log(p/q)` remains a
diagnostic and cannot become divergence authority.

Receipts must persist `metric_id=argmax_native_vjp_fidelity_v1` separately from
`reachable_decision_geometry_fidelity.v1` and
`reachable_decision_preconditioner_selection.v1` schemas.

## M selection

The selector refuses anything except exactly 600 unique states. Each state
must bind exact and candidate renderer pullbacks, the same named M candidates,
and `through_r_authority=exact_argmax_after_real_R`. Selection ranks Spearman
correlation between `rho_M` and exact real-R argmax agreement, then Pearson.
Name orders receipt rows only; an exact primary+tie-break tie emits no unique
selection. A zero-variance M is retained as uninformative and cannot win. Each
mapping key must equal its preconditioner's canonical name; the receipt binds
both the ordered state IDs and exact through-R outcome vector by SHA-256.
Categorical finite Bregman/KL and dual-Euclidean squared-Hessian distance are
evaluated as separately typed predictors on those same sealed rows; neither is
silently added to the preconditioner-M contest or renamed `rho`.

Future command after the manifest exists:

```text
PYTHONPATH=src .venv/bin/python tools/probe_surrogate_vjp_fidelity_metric.py \
  measure-m-selection-n600 \
  --manifest <content-bound-optimal_metric_n600_m_selection_manifest.v1.json> \
  --output <durable-SSD-or-repo-small-receipt.json>
```

## Null-space/reweighting experiment

On the same state and Jacobian, compute four stages:

1. raw ambient input cotangent;
2. `U^T g`, where `U` is an orthonormal basis for `range(J_R)` — null removal only;
3. `J_R^T g` with identity — renderer singular-value anisotropy;
4. `J_R^T g` under selected `M` — decision/optimizer reweighting.

Report the three incremental rho lifts separately. No stage may borrow another
state, candidate, Jacobian, or preconditioner.

## Provider admission

Raw input-costate cosine/L2 is diagnostic only. A provider needs:

1. a canonical reachable receipt with positive `rho` and `eta`;
2. exact real-teacher one-step or short-trajectory functional evidence;
3. exactly-n600 M-selection custody for whole-teacher distillation;
4. NumPy/framework parity in the identical student tensor chart;
5. matched charged economics and exact fallback.

Whole-teacher admission additionally rehashes the selection, n600 fidelity-
aggregate, and exact-functional receipt files at decision time. It refuses
duplicate JSON keys, schema/law/state-count drift, an absent selected M,
cross-receipt SHA drift, or aggregate worst-pair values that differ from the
typed evidence object.

Missing optimal geometry is `NO_VERDICT_DATA_CUSTODY`, never a technique
negative. The legacy whole-teacher ambient VJP fit driver is blocked until its
cache/driver can consume renderer pullbacks, selected M, and exact functional
validation.

## Current lawful commands

```text
PYTHONPATH=src .venv/bin/python tools/probe_surrogate_vjp_fidelity_metric.py \
  audit-current-p0-custody

PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_vjp_fidelity.py \
  src/tac/tests/test_segnet_gradient_replacement.py \
  src/tac/tests/test_scorer_gradient_policy.py \
  src/tac/tests/test_whole_teacher_distilled_student_vjp.py \
  src/tac/witness_dsl/tests/test_whole_teacher_distilled_student_policy.py
```

No current command can lawfully regenerate the deleted student/candidate
pullbacks from #205 plus `gt_n600.npz`; the GT cache is source/label custody,
not surrogate-state custody.

## V9·CGauge binding

The canonical metric is a required fail-closed V9 provider-selection leg, not
a standalone trainer. `surrogate_vjp_fidelity_metric_lever()` is the named DSL
lever. `compile_v9_optimal_metric_binding()` emits the manifest object with:

- `law_ref=metric_id=argmax_native_vjp_fidelity_v1`;
- distinct state and selector receipt schemas;
- the `CompiledScorerGradientPolicy` consumer;
- content-bound path+SHA-256 custody for measurement, functional, and
  terminality receipts;
- `fallback=full_frozen_teacher` and activation refused while current receipts
  are absent.
- `bregman_geometry_id=argmax_native_bregman_grounding_v1`, the categorical
  Bregman and extended-KL schemas, and the no-laundering geometry rules.

`verify_v9_optimal_metric_binding()` refuses drift or extra fields. The lane
that exclusively owns V9 config/autoconfig/provenance-bijection surfaces must
compose this lever and manifest into every V9 compiler; this lane does not
collide with those owned files.
