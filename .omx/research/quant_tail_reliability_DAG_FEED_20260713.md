# DAG FEED — quantitative control-interpolator tail reliability — 2026-07-13

`research_only=true` · `score_claim=false` · `pointer_delta=NONE` · `$0 cached local`

This is a standalone, append-ready FEED because the shared DAG surface was live-sibling-held.
It must not be promoted by inferred equivalence to a score or controller result.

```yaml
feed_id: FEED-quant-tail-reliability-20260713
lane_id: lane_quant_tail_reliability_20260713
means_only: true
authority: "[macOS-CPU advisory; NumPy-fp32 decision; no score authority]"
producer:
  tool: tools/measure_quant_tail_reliability_20260713.py
  receipt: .omx/research/quant_tail_reliability_receipt_20260713.json
triality:
  dsl: tac.witness_dsl.control_tail_reliability_policy_20260713.ControlTailReliabilityPolicy
  equations:
    - control_interpolator_tail_cvar_mean_gate_v1
    - fixed_design_correlated_gaussian_ridge_tail_v1
  dag: FEED-quant-tail-reliability-20260713
inputs:
  - sealed PRE-SE n600 custody: 480 inherited + 120 untouched heldout
  - cached PRE-SE 420-core/60-train-only-dev exact-mass arrays
  - inherited seeds: [455, 456, 457]
  - cached organ #433 trajectory: 9 intervals, 7 walk-forward folds
outputs:
  pre_se:
    block2: {lambda_dev: 1.0, shortfall_p95: 0.8687327671, shortfall_p99: 0.8857782308, cvar95: 0.8805590344}
    block3: {lambda_dev: 0.3, shortfall_p95: 0.9196654968, shortfall_p99: 0.9329094312, cvar95: 0.9291047003}
    official_n120_new_lambda: BLOCKED_MISSING_RAW_ARRAYS
  organ:
    A: {diagnostic_lambda: 1000.0, bracket_closed: false, operational_recommendation: prefer_persistence}
    P: {lambda_dev: 0.1, cvar_relative_to_own_default: -0.0411173347}
    Q: {lambda_dev: 0.1, cvar_relative_to_own_default: -0.0410981617}
  finite_bound: {symbolic_derived: true, numeric_closure: false}
consumers:
  - costate support-selector reliability gate
  - witness-control organ admission gate
  - ncde/prototype-router/rate-law/scorer-response tail-reporting debt
  - cathedral/autopilot fail-closed lambda admission
guards:
  - require positive lambda for load-bearing selection
  - require empirical mean gate
  - require closed lambda bracket
  - require state/trajectory-block holdout
  - require tail quantile and CVaR beside mean
  - never promote forecast-regret proxy as counterfactual control regret
  - never promote MEANS receipt as byte-closed score evidence
negative_edges:
  - verdict_scope: "MEASUREMENT x OFFICIAL-N120 x NEW-LAMBDA"
    verdict: NO-GO_CACHE_INSUFFICIENT
    req_R: preserve raw heldout features, exact masses, and state boundaries
  - verdict_scope: "BOUND x PRESENT-CACHED-PRE-SE-CUSTODY"
    verdict: SYMBOLIC_CLOSE_NUMERIC_NO_CLOSE
    req_R: covariance innovations and score-boundary margin envelope
  - verdict_scope: "FORMULATION x FROZEN-WITNESS-SGD"
    verdict: CONDITIONAL_NOT_THEOREM_ADMITTED
    req_R: fixed map, gamma, sigma, kappa_E, unbiasedness, native residual trace
```

Shared-DAG append remains deferred until the hot-file owner lands or main review applies
this FEED. No shared DAG bytes were absorbed into this lane.
