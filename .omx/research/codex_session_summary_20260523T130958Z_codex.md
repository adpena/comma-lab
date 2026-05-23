# Codex Session Summary - DQS1 Local-First Harvest 15

- timestamp_utc: `2026-05-23T13:09:58Z`
- schema: `codex_session_summary.v1`
- agent: `codex`
- topic: `dqs1_local_first_harvest15`
- evidence_axis: `[macOS-CPU advisory only]`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Landed

- Harvested `pairset_drop_two_r010_003_p0376_0479` through the local-first queue.
- Local advisory score: `0.19204161709818363`.
- Conservative projected contest-CPU score from drift calibration: `0.1920341170981836`.
- Eureka trigger: `false`; recommended action: `observe_only`.
- Retention executed for rebuildable locality/local-CPU scratch: `5` paths, `14649816858` bytes reclaimed, `0` blocked candidates.
- Canonical harvest observation model regenerated at `.omx/research/dqs1_local_first_harvest_observations_20260523T130907Z.*`.
- Cross-family portfolio regenerated at `experiments/results/cross_family_candidate_portfolio/20260523T130907Z_full_drop_two_local_harvest15/`.
- Checked-in queue advanced to `pairset_drop_two_r010_008_p0376_0496`.

## Current Signal

- Best harvested local advisory remains `pairset_drop_two_r029_017_p0259_0242` at `0.19203861709818362`, advisory-only.
- `r010_003` matched the prior `r010_005` local score and did not emit an exact auth request.
- The new queue target continues the `r010` partner sweep while skipping all candidates with existing local advisory artifacts.

## Verification

- Queue validation: `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- Process cleanup: no `run_dqs1_local_first_autopilot`, `contest_auth_eval`, `evaluate.py`, `inflate.py`, or retention worker remained after harvest.
- Disk after retention: about `68G` free on `/System/Volumes/Data`.

## Next Step

Run the bounded local-first autopilot on `pairset_drop_two_r010_008_p0376_0496`, then regenerate the same observation, portfolio, and queue surfaces. Do not dispatch exact CPU/CUDA unless a calibrated eureka record creates a contest-auth request.
