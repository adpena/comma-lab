# Codex Findings: Dynamic-Sparse Sweep Feedback And DQS1 Bridge

- UTC: 2026-05-25T12:20:22Z
- Lane: `codex_dynamic_sparse_sweep_artifact_receiver_counts_20260525`
- Scope: queue-owned materializer feedback integration, receiver-gated dynamic-sparse hints, DQS1 local-first replanning bridge.

## Findings

1. `tools/run_byte_shaving_materializer_campaign.py` only counted shallow `expected_artifacts` rows when deciding whether a dynamic-sparse feedback compiler hint had receiver-positive materializer signal. If a successful queue step pointed at a `family_agnostic_materializer_empirical_sweep.v1` JSON artifact, receiver-positive rows inside that sweep were invisible to the runner preflight even though the dynamic-sparse oracle could consume them.
2. The runner now resolves bounded `.json`/`.jsonl` feedback artifacts relative to the queue observation, counts nested empirical observation rows, and separates:
   - receiver-positive rate-saving rows,
   - receiver-positive but rate-nonpositive rows,
   - receiver-negative rows,
   - rate-saving rows with no receiver proof.
3. A fallback artifact with no receiver proof remains `rate_saving_without_receiver_proof_count`; it is not misclassified as receiver-negative. Truthy authority fields in nested rows remain ignored by the counter and do not unblock compiler-hint emission.
4. Added `tac.optimization.dqs1_materializer_feedback_bridge` and DQS1 queue wiring so saturated family-agnostic materializer feedback can be stamped onto DQS1 local-first queue metadata as false-authority replanning context. Empty/no-feedback DQS1 queues do not receive the bridge.
5. `tools/build_dqs1_local_first_queue.py` now accepts repeated `--materializer-feedback` inputs and can write an optional bridge JSON via `--materializer-feedback-bridge-out`.
6. `tools/run_dqs1_local_first_tranche.py` now forwards repeated `--materializer-feedback` inputs into every queue rebuild, so the bridge is reachable from the unattended DQS1 tranche control loop rather than only from the direct queue-builder CLI.

## Proof Artifacts

- Bridge proof: `.omx/research/codex_dqs1_materializer_feedback_bridge_20260525T121724Z/dqs1_materializer_feedback_bridge.json`
  - SHA-256: `738603d7c52fc2ea91a2fd50a6ec903a8a470cf4152c4b78d7170f4d23ab7ac9`
  - Observation count: 2
  - Recommended next action: `switch_to_dqs1_pairset_composition_followup`
- Queue proof: `.omx/research/codex_dqs1_materializer_feedback_bridge_20260525T121724Z/dqs1_pairset_local_first.materializer_feedback.json`
  - SHA-256: `fc4071a9552d8d591c3bd37506d6fe10ffa1025fb57dcf1554bbc8d44f33cf04`
  - Candidate count: 4
  - Queue schema: `experiment_queue.v1`

## Authority

All new records are local planning context only. They set `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`, and do not claim contest CPU/CUDA score authority. Sweep artifacts with truthy top-level authority fields are rejected before their nested rows can unblock a compiler hint.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py tools/build_dqs1_local_first_queue.py tools/run_dqs1_local_first_tranche.py src/comma_lab/scheduler/dqs1_local_first_queue.py src/tac/optimization/dqs1_materializer_feedback_bridge.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py --no-cache`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_inverse_steganalysis_acquisition.py::test_queue_observation_reads_succeeded_materializer_observation_jsonl src/tac/tests/test_inverse_steganalysis_acquisition.py::test_queue_observation_deduplicates_materializer_sweep_json_and_jsonl -q --durations=30 --durations-min=0.01`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary experiments/results/cross_family_candidate_portfolio/20260523T134800Z_full_drop_two_local_harvest16/action_summary.json --candidate-limit 4 --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/header_elide/sweep.json --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/recompress/sweep.json --materializer-feedback-bridge-out /tmp/pact_dqs1_bridge_smoke_bridge_codex_20260525T123000Z.json --write --output /tmp/pact_dqs1_bridge_smoke_queue_codex_20260525T123000Z.json`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_dqs1_materializer_feedback_bridge_20260525T121724Z/dqs1_pairset_local_first.materializer_feedback.json validate`
- `git diff --check`

## Next

The next bridge should consume completed DQS1 local CPU advisory rows as `dqs1_observations`, so the replanning context can compare materializer demotion signal against actual DQS1 pairset composition outcomes without operator hand-carrying candidate IDs.
