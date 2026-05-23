# Codex Findings: Inverse Queue Performance And Action Bridge

- timestamp_utc: `2026-05-23T18:03:51Z`
- lanes:
  - `lane_inverse_scorer_action_functional_cli_20260523`
  - `lane_inverse_steganalysis_acquisition_surface_20260523`
- score_axis: planning/local telemetry only; no new score claim

## Findings

1. `experiment_queue_performance_summary.v1` needed an explicit false-authority envelope before downstream planners consumed it. The summary is now telemetry-only and includes resource-kind counts/dominant resource kind so inverse acquisition can use measured wall-clock and artifact cost without inventing scorer gain.
2. Queue performance is now ingestible by `tac.optimization.inverse_steganalysis_acquisition.observations_from_queue_performance_summary(...)` as denominator calibration only. Direct scorer observations with real `observed_score_gain` outrank queue telemetry, preventing cheap-but-uncalibrated runs from overriding quality evidence.
3. Inverse scorer cells now have a safe materializer bridge for local planning artifacts: `probe_inverse_scorer_surface_cell` can build `inverse_steganalysis_discrete_action_functional.v1` through the materializer work queue, but `materialize_inverse_scorer_cell_candidate` remains non-executable until byte-closed archive grammar, runtime-consumption proof, and exact auth custody exist.

## Landed Surfaces

- `tools/build_inverse_steganalysis_action_functional.py`
- `tac.optimization.inverse_steganalysis_acquisition`
- `comma_lab.scheduler.experiment_queue`
- `comma_lab.scheduler.experiment_queue_observer`
- `comma_lab.scheduler.byte_shaving_materializer_registry`
- `comma_lab.scheduler.byte_shaving_campaign_queue`
- `tac.optimization.scorer_inverse_decision_surface`
- `tac.optimization.byte_shaving_campaign`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py`
- `.venv/bin/python -m ruff check ...` on touched Python surfaces
- `.venv/bin/python -m py_compile ...` on touched runtime/tool surfaces
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml observe --tail-lines 8 --format json`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml performance`
- `.venv/bin/python tools/lane_maturity.py validate`
- `.venv/bin/python tools/review_tracker.py scan`
- `.venv/bin/python tools/scan_best_anchor_per_axis.py`

## Next

The next high-EV bridge is a Dask executor under `experiment_queue.v1`: it should claim a queue step, verify definition hashes/postconditions, execute, and write back through normal queue events. Do not create a second queue authority.
