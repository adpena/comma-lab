# 600-Pair Independence Test Result

Timestamp: 2026-05-19T21:19:27Z  
Owner: codex  
Task: `codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z::CLUSTER_F2` freshness closure  
Evidence grade: `local_independence_diagnostic_only`; no score claim

## Result

`CLUSTER_F2` was already implemented and completed in commit `b83481970`. This pass freshness-verified it and closes the directive's exact memo-name gap.

Focused test:

```bash
.venv/bin/python -m pytest src/tac/tests/test_600_pair_independence_tool.py -q -p no:cacheprovider
# 4 passed
```

Live diagnostic rerun:

```bash
.venv/bin/python tools/test_600_pair_independence.py \
  --input-json experiments/results/lane_local_hardware_maximization_sweep_20260513_20260513T210232Z/per_pair_sensitivity_a1_baseline_600pair.json \
  --input-json experiments/results/d1_pair_component_xray_a1_baseline_cpu_20260515_codex/pair_component_xray.json \
  --input-json experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --max-lag 32 \
  --output-json experiments/results/600_pair_independence_test_20260519T211927Z_codex/report.json \
  --summary
```

Output:

```text
[600-pair-independence] verdict=independence_assumption_blocked series=16 output=experiments/results/600_pair_independence_test_20260519T211927Z_codex/report.json
```

Ignored local report SHA-256:

```text
6d8dc247bf4eea3f1f67e3c61753540d4d0225c1e66037dbcfc1d26282514c5c
```

## Interpretation

The iid 600-pair assumption is not defensible for these existing local per-pair artifacts.

High-signal row from the current report:

- `raw_per_pair.pose_per_pair`: max lag autocorrelation `0.4854`, serial effective N `40.22`, top-50 absolute share `0.3112`, verdict `independence_assumption_blocked`.

Cross-series dependence is also blocked; the report found `cross_vector_dependence_blocked` pairs, including component identities and near-identities that should not be treated as independent evidence.

## False-Authority Controls

The diagnostic remains non-promotional:

- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatchable=false`
- `auth_eval_skipped=true`
- `scorer_invoked=false`
- `provider_invoked=false`

## Consequence

Cathedral autopilot, cost-band posterior updates, Thompson sampling, and per-pair solvers should use block-aware or temporally correlated assumptions for 600-pair vectors. This does not rank, kill, promote, or dispatch any substrate.

The earlier pending probe row `fp64_master_gradient_600_pair_independence_pending_20260518` should be superseded as "empirical test run; iid assumption blocked" so it no longer reads as "pending empirical test."


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:codex-pivot-independence-test-result-memo-trigger-tokens-in-recommendation-section-not-new-equation-claim -->
