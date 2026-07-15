# DAG FEED — task #504 Bregman all-surfaces application

FEED id: `FEED-504-bregman-all-surfaces`

Research-only: `true`

Pointer moved: `false`

## New equation nodes

1. `cgauge_categorical_bregman_hessian_covariance_v1`
   - parents: `optimal_metric_unification_v1`, `cgauge_master_action_v1:A2`
   - operation: twice differentiate `F=logsumexp`; apply the affine chain rule
   - label: `DERIVED_EXACT`
   - gap: full live scorer-VJP/affine-chart custody is OWED
2. `bregman_closed_form_dual_cancellation_v1`
   - parents: `cgauge_categorical_bregman_hessian_covariance_v1`,
     `bregman_dual_metric_squared_hessian_v1`
   - operation: Legendre duality and sum the two directed divergences
   - label: `DERIVED_EXACT`
3. `bregman_nonnegative_right_centroid_invariant_v1`
   - parents: `bregman_closed_form_dual_cancellation_v1`
   - operation: first-order convexity and differentiate the weighted right-data
     objective
   - label: `DERIVED_EXACT`
4. `bregman_positive_unscented_propagation_v1`
   - parents: `bregman_nonnegative_right_centroid_invariant_v1`
   - operation: positive paired sigma construction; exact moment cancellation;
     exponential-family sufficient-statistic error reduction
   - label: `DERIVED` (input moments exact; nonlinear output approximate)

## Housekeeping nodes made query-visible

- `windowed_curvelet_parabolic_capacity_v1`: existing equation module, newly
  appended through `register_canonical_equation`; no new score/family verdict.
- `compact_shearlet_parabolic_capacity_v1`: new structural equation around the
  genuine compact-shearlet parabolic/shear law. Its proof receipt is structural;
  family-selection status remains `NO_VERDICT_DATA_CUSTODY`.

## Triality and ownership

- Equations: `src/tac/canonical_equations/`
- DAG: this FEED
- DSL: **OWED**, because no real trainer-consumed swept Bregman/centroid/sigma
  actuator is evidenced and the witness-DSL surface is owned by another arm.

No launch, paid dispatch, evaluator call, archive mutation, or pointer update.
