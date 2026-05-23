# Codex Session Summary - DQS1 Local-First Harvest 14

- timestamp_utc: `2026-05-23T12:56:52Z`
- schema: `codex_session_summary.v1`
- agent: `codex`
- topic: `dqs1_local_first_harvest14`
- evidence_axis: `[macOS-CPU advisory only]`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Landed

- Harvested `pairset_drop_two_r010_005_p0376_0467` through the local-first queue.
- Local advisory score: `0.19204161709818363`.
- Conservative projected contest-CPU score from drift calibration: `0.1920341170981836`.
- Eureka trigger: `false`; recommended action: `observe_only`.
- Retention executed for rebuildable local-CPU scratch: `2` paths, `3662588058` bytes reclaimed, `0` blocked candidates.
- Canonical harvest observation model regenerated at `.omx/research/dqs1_local_first_harvest_observations_20260523T125445Z.*`.
- Cross-family portfolio regenerated at `experiments/results/cross_family_candidate_portfolio/20260523T125445Z_full_drop_two_local_harvest14/`.
- Checked-in queue advanced to `pairset_drop_two_r010_003_p0376_0479` with one local CPU worker and post-advisory retention retained.

## Current Signal

- Best harvested local advisory remains `pairset_drop_two_r029_017_p0259_0242` at `0.19203861709818362`, which is still advisory-only and not promotion authority.
- The new `r010_005` drop-two result was worse than the best local advisory and did not justify exact auth dispatch.
- Observation and component-marginal response models are active in the latest portfolio; the queue builder skipped candidates with existing local advisory artifacts and selected the next unobserved high-acquisition drop-two candidate.

## Verification

- Queue validation: `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- Focused regression set: `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_dqs1_local_first_harvest_observations.py src/tac/tests/test_mlx_dynamic_sweep_observations.py`
- Result: `73 passed in 7.34s`
- Whitespace check: `git diff --check`

## Next Step

Run the bounded local-first autopilot on `pairset_drop_two_r010_003_p0376_0479`, harvest it append-only, regenerate the observation and portfolio surfaces, and keep exact CPU/CUDA dispatch blocked unless the calibrated eureka gate emits a contest-auth request.
