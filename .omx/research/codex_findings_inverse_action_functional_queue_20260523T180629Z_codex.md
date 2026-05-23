# Codex Findings: Inverse Action Functional Queue Bridge

UTC: 2026-05-23T18:06:29Z
Agent: Codex
Lane: lane_inverse_scorer_action_functional_cli_20260523

## Scope

Adversarial review of the inverse-scorer/action-functional landing and its queue/DAG integration path.

## Findings

1. Inverse-scorer cells were not queue-materializable as landed. The planner could emit `scorer_inverse_surface_cell` units, but the materializer registry had no explicit inverse-scorer target contract. Fix landed: `inverse_scorer_action_functional_v1` is an executable planning-only probe adapter, and `inverse_scorer_cell_candidate_v1` is a fail-closed candidate materializer contract until deterministic pixel/byte materialization and receiver proof exist.

2. The planner tie-breaker could choose the heavier fail-closed cell materializer over the action-functional probe when the predicted value was tied. Fix landed: operation candidates now carry `operation_priority` and prefer lower-priority/fewer-blocker operations after value terms. The inverse-surface path now selects `probe_inverse_scorer_surface_cell` -> `inverse_scorer_action_functional_adapter` first.

3. Native MLX inverse-window rows could leak native-window deltas into full-video-looking fields when `allow_native_mlx_window_objective` was enabled. Fix landed: native rows now keep `planning_value_scope=native_mlx_window`, use `native_window_delta_vs_baseline_score`, and propagate `native_mlx_window_objective_not_full_video_normalized` through refs, surface blockers, units, ranked rows, and selected operations.

4. Inverse-steganalysis observations accepted contest-looking axes/resource kinds without auth-eval payload validation. Fix landed: `normalize_inverse_steganalysis_observation(...)` rejects contest auth score axes and `resource_kind=contest_exact_eval`; contest authority still must flow through auth-eval schema validation.

5. Queue runtime telemetry was not consumable by the acquisition denominator. Fix landed: experiment-queue performance summaries expose resource-kind counts and false-authority telemetry, the observer surfaces them read-only, and inverse-steganalysis acquisition can convert them into planning-only observations with runtime/cache identity.

## Empirical Smoke

- Built action functional from the real MLX scorer-response artifact:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_steganalysis_action_functional_20260523_codex.json`
- Built byte-shaving plan from that action functional:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_action_functional_byte_shaving_plan_20260523_codex.json`
- Built inverse-scorer signal surface and plan from the real MLX response path:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_surface_byte_shaving_plan_20260523_codex.json`
- Materializer compile stayed fail-closed for the non-implemented cell candidate materializer:
  `executable_row_count=0`, `blocked_row_count=38`, `materializer_backlog_row_count=1`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_staircase_dag.py`
- `.venv/bin/python -m ruff check ...`
- `.venv/bin/python -m py_compile ...`

## Remaining Gaps

1. Implement deterministic `inverse_scorer_cell_candidate_adapter` materialization with runtime-consumption proof.
2. Feed queue-performance observations from actual materializer execution queues after the first executable probe queue run.
3. Add exact CPU/CUDA auth-axis calibration only after byte-closed archive/runtime packets exist.
