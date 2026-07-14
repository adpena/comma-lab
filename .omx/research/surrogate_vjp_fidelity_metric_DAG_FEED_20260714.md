# SURROGATE-VJP-FIDELITY DAG FEED — 2026-07-14

`research_only=true`  
Lane: `lane_surrogate_vjp_fidelity_metric_20260714`

## Node

`FEED-SURROGATE-VJP-FIDELITY-ARGMAX-NATIVE`

## Dependencies

- real n600 source/runtime custody
- actual-R rendered-state cache
- frozen teacher centered logits/probabilities, labels, margins, active rivals
- exact teacher input costate and comparable student input VJP
- renderer Jacobian pullbacks and optimizer/preconditioner custody
- identical-state teacher/surrogate perturbation outcomes
- time-ordered cache/live density-ratio receipt for on-policy correction
- deterministic repeat floor and sealed non-inferiority gate
- feature-chart terminality receipt: fixed-objective terminal-gradient/curvature
  bound, range-space/null custody, and heldout exact ridge ladder; nonlinear
  charts require a separate convergence certificate

## Transform

1. `q = (I-11^T/C) z`
2. `h_T = J_R^T g_T`, `h_S = J_R^T g_S`
3. `<g_T,g_S>_(R,M) = h_T^T M h_S`
4. `eta = (h_T^T M h_S)/(h_T^T M h_T)` with `rho`, norm ratio,
   relative L2, aggregate/worst-pair/per-class/regime facets
5. `A_theta = W^(1/2) C D_x T D_theta R`, `G_theta=A_theta^T A_theta`
6. compare low-margin winner–rival directional derivatives and crossings
7. compare Fisher/KL quotient effects with rank/support checks
8. importance-correct fidelity by sealed clipped/masked `pi/mu` weights
9. run identical-state exact functional step and short-trajectory A/B
10. evaluate `C_S,VJP + (C_T+U)/K < 150.453 ms/step`

## Refusal edges

- missing full n600 cache -> `BLOCKED_DATA_CUSTODY`
- missing or unsupported density ratio -> `BLOCKED_DISTRIBUTION_CUSTODY`
- zero exact norm, Fisher rank collapse outside gauge, exact tie without active
  set, nonpositive exact-control improvement -> `REFUSE_METRIC`
- only raw cosine/ordinal/soft value pass ->
  `FIRST_CUT_INSTANCE_DIAGNOSTIC_ONLY; QUEUE_OPTIMAL_FORM`
- after real-n600 optimal-form measurement only: functional lower confidence
  bound does not beat cost ratio or any exact d_seg/d_pose facet harms ->
  `OPTIMAL_FORM_INSTANCE_NOT_LICENSED`
- missing resume/stage checkpoint/storage cleanup proof -> `REFUSE_LAUNCH`
- missing chart-appropriate terminality receipt ->
  `UNDERTRAINING_UNRESOLVED; NO_NEGATIVE_INTERPRETATION`

## Outputs

- authenticated metric receipt with explicit measured/unmeasured fields
- typed default-off DSL policy
- canonical argmax-native equation
- preserved stage checkpoints and sufficient-statistic cache
- probe outcome/posterior row after actual empirical anchor
- exact host command for `$0` remeasurement and n600 preflight; corrected
  refit/timing remains fail-closed until its fit driver lands

## Current anchor

Retained real-n600-source heldout `n=120` reaggregation raises renderer-pullback
cosine over raw RGB cosine, but yields only `eta=0.0023036` (round 2) and
`eta≈0.004648` (round 3).  These first-cut instances remain below the static
license gate.  This is not a technique verdict.  The optimal centered-logit
value+Jacobian decision/Fisher/functional formulation remains live and
unmeasured because its n600 cache never existed.

## Consumer wire-in

- sensitivity map consumes active-margin decision derivatives after custody
- Pareto gate consumes exact d_seg/d_pose/bytes functional outcomes
- bit allocator consumes score-unit value per byte only after exact evaluation
- cathedral/autopilot consumes only an admitted sealed policy receipt
- continual-learning posterior consumes every measured pass/fail/blocker
- probe disambiguator retains separate reachable, decision/Fisher, ordinal,
  and exact-functional interpretations
