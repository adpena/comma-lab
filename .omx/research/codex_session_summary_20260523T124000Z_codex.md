# Codex Session Summary

- Timestamp: 2026-05-23T12:40:00Z
- Primary lane: `lane_dqs1_local_harvest_observation_canonicalization_20260523`
- Frontier status: unchanged; current canonical pointers remain `[contest-CPU Linux x86_64] 0.1920282830` and `[contest-CUDA T4] 0.2053300290`.

## Landed Work

- Added `tac.optimization.dqs1_local_first_harvest_observations` and `tools/build_dqs1_local_first_harvest_observations.py` to convert DQS1 local-first harvests into reusable `mlx_dynamic_sweep_observation.v1` rows.
- Extended dynamic sweep observations with selected-pair, acquisition-operation, baseline, score-delta, and archive-byte-delta metadata.
- Hardened cross-family action-summary false authority by adding `dispatch_attempted=false` and `gpu_launched=false`.
- Hardened DQS1 queue-builder schema compatibility for current cross-family summaries.
- Added a worker-level `max_experiments` bound and wired DQS1 autopilot to one candidate per harvest boundary.
- Fixed harvest-glob poisoning from generated observation artifacts.
- Ran one bounded local CPU candidate, harvested its advisory/eureka signal, retained certified rebuildable scratch, regenerated canonical observations, regenerated the cross-family portfolio, and rebuilt the checked-in one-candidate queue.

## Latest Local Signal

- Candidate: `pairset_drop_two_r010_009_p0376_0459`
- Axis: `[macOS-CPU advisory only]`
- Local score: `0.19203961709818362`
- Conservative projected contest CPU score: `0.1920321170981836`
- Eureka: `false`, `observe_only`
- Component delta vs compact top32 baseline: SegNet `+0.000002`, PoseNet `0`, rate `-0.0000013317179062416473`

## Validation

- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_dqs1_local_first_harvest_observations.py src/tac/tests/test_mlx_dynamic_sweep_observations.py` -> 73 passed.
- Ruff passed on all touched Python modules/tools/tests.
- `tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate` -> valid, one experiment, seven steps.
- `git diff --check` passed.
- `.gitignore` already ignores `experiments/results/`, `*.raw`, queue SQLite files, and `.omx/tmp/`; no update required this turn.

## Next

The next run should execute the checked-in queue for `pairset_drop_two_r010_005_p0376_0467` under the fixed one-candidate autopilot window, then regenerate harvest observations and portfolio again. In parallel, MLX same-candidate advisory should be wired into the same queue boundary so local MLX can replace the ~8 minute CPU advisory step once parity/calibration is proven.
