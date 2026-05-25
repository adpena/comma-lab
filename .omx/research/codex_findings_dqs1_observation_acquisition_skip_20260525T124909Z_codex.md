# Codex Findings: DQS1 Observation Acquisition Skip

- UTC: 2026-05-25T12:49:09Z
- Lane: `codex_dqs1_observation_acquisition_skip_20260525`
- Scope: DQS1 local-first queue selection, harvest observation JSONL consumption, materializer-feedback bridge continuity.

## Findings

1. DQS1 harvest observations were visible in the materializer-feedback bridge, but queue candidate selection still relied on completed result roots for rerun suppression. If storage roots moved or were incomplete, an already observed local-first candidate could be selected again.
2. `src/comma_lab/scheduler/dqs1_local_first_queue.py` now treats canonical DQS1 local-first harvest observation rows as queue acquisition skip signal by default. Observation rows are checked for truthy authority before they can suppress a candidate.
3. `tools/build_dqs1_local_first_queue.py` and `tools/run_dqs1_local_first_tranche.py` now expose `--include-observed-dqs1-candidate` as an explicit replay/debug opt-in. Default unattended tranche rebuilds suppress observed candidates.
4. Queue metadata now preserves `skipped_candidates`, and selected experiment metadata carries a `dqs1_observation_acquisition_skip.v1` policy record when observation-based suppression is active.
5. The materializer-feedback bridge remains attached to the queue and still records observed DQS1 outcomes. Observation-based skip does not grant score, promotion, rank/kill, exact-eval, GPU, or dispatch authority.

## Durable Proof

- Bridge:
  `.omx/research/codex_dqs1_observation_acquisition_skip_20260525T124909Z/dqs1_materializer_feedback_bridge.observation_skip.json`
  - SHA-256: `6dbbdb4d9144dd234eb8d4e931b1e5a0cd87c6ee1ea70c8288f745bdd6ed711c`
  - `observed_dqs1_candidate_count`: 20
  - `dqs1_harvest_outcome_counts`: 2 improved, 18 regressed, 0 flat/byte-only
  - recommended next action: `continue_dqs1_pairset_composition_from_positive_harvest_signal`
- Queue:
  `.omx/research/codex_dqs1_observation_acquisition_skip_20260525T124909Z/dqs1_pairset_local_first.observation_skip.json`
  - SHA-256: `d065c3cc7c790d836b7574a18b4f52c8f0dab9d7a63da69a2d48cac3eb860f3a`
  - `queue_id`: `dqs1_pairset_local_first_observation_skip`
  - Selected candidates:
    - `pairset_drop_two_r012_010_p0134_0376`
    - `pairset_drop_two_r009_008_p0459_0496`
    - `pairset_drop_two_r009_001_p0459_0501`
    - `pairset_drop_two_r008_005_p0496_0467`
  - First experiment records 30 skipped candidates, including 10 suppressed by `dqs1_harvest_observation_exists`.
  - `experiment_queue.v1`, 4 experiments, 28 steps, validation green.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/dqs1_materializer_feedback_bridge.py src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_dqs1_local_first_queue.py tools/run_dqs1_local_first_tranche.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py --no-cache`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_dqs1_local_first_harvest_observations.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q --durations=30 --durations-min=0.01`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --candidate-limit 4 --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/header_elide/sweep.json --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/recompress/sweep.json --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260523T133010Z.jsonl --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260524T065107Z.jsonl --materializer-feedback-bridge-out .omx/research/codex_dqs1_observation_acquisition_skip_20260525T124909Z/dqs1_materializer_feedback_bridge.observation_skip.json --write --output .omx/research/codex_dqs1_observation_acquisition_skip_20260525T124909Z/dqs1_pairset_local_first.observation_skip.json --queue-id dqs1_pairset_local_first_observation_skip`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_dqs1_observation_acquisition_skip_20260525T124909Z/dqs1_pairset_local_first.observation_skip.json validate`
- `rg -n '"(score_claim|promotion_eligible|rank_or_kill_eligible|ready_for_exact_eval_dispatch|dispatch_attempted|gpu_launched)": true' .omx/research/codex_dqs1_observation_acquisition_skip_20260525T124909Z/dqs1_materializer_feedback_bridge.observation_skip.json .omx/research/codex_dqs1_observation_acquisition_skip_20260525T124909Z/dqs1_pairset_local_first.observation_skip.json`
- `git diff --check`

## Next

Use the same observed-candidate skip policy in any future DQS1 acquisition-plan regeneration path so static pairset acquisition plans and queue-selected action summaries cannot drift apart after harvest.
