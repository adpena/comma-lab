# Codex Findings: Frontier Rate-Attack Feedback Compiler

UTC: 2026-05-25T13:33:56Z
Lane: `codex_frontier_rate_attack_feedback_compiler_20260525`

## Verdict

The next tranche is no longer a single materializer/receiver leaf. The landed
surface compiles frontier final-rate materializer observations plus DQS1
local-first harvest observations into a queue-owned bounded follow-up plan.
The compiler discovers family-agnostic materializer feedback under artifact
roots, deduplicates repeated `sweep.json` / `observations.jsonl` echoes, carries
the resulting bridge into every DQS1 experiment, and emits a normal
`experiment_queue.v1` for local execution.

## Artifacts

- Compiler module: `src/comma_lab/scheduler/frontier_rate_attack_feedback.py`
- Operator CLI: `tools/build_frontier_rate_attack_feedback_refresh.py`
- Regression tests: `src/tac/tests/test_frontier_rate_attack_feedback.py`
- Live proof bundle: `.omx/research/codex_frontier_rate_attack_feedback_compiler_20260525T133356Z/`
- Follow-up queue: `.omx/research/codex_frontier_rate_attack_feedback_compiler_20260525T133356Z/dqs1_followup_queue.json`

## Live Proof

Inputs:

- Frontier materializer root: `experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z`
- DQS1 harvest rows:
  - `.omx/research/dqs1_local_first_harvest_observations_20260523T133010Z.jsonl`
  - `.omx/research/dqs1_local_first_harvest_observations_20260524T065107Z.jsonl`
- Refreshed action summary: `.omx/research/codex_dqs1_tranche_acquisition_refresh_frontier_bootstrap_20260525T131428Z/action_summary.observation_refreshed.json`

Output:

- Discovered materializer payloads: 2 unique payloads after suppressing 2 duplicate echo rows.
- DQS1 observations consumed: 20.
- Follow-up candidates selected:
  - `pairset_drop_two_r013_009_p0327_0459`
  - `pairset_drop_two_r013_005_p0327_0467`
  - `pairset_drop_two_r013_003_p0327_0479`
  - `pairset_drop_two_r013_008_p0327_0496`
- Queue shape: 4 experiments / 28 steps.
- Bridge next action: `continue_dqs1_pairset_composition_from_positive_harvest_signal`.

## Authority Boundary

All emitted planner, bridge, discovery, queue-summary, and queue metadata keep:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- `gpu_launched=false`

This is local queue replanning signal only. It does not claim score, promotion,
rank/kill authority, or paid dispatch readiness.

## Verification

- `ruff` on feedback compiler, CLI, scheduler export surface, bootstrap bridge,
  HNeRV section parser, and focused tests: clean.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 3 passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_hnerv_packet_sections.py -q`: 26 passed.
- `pytest src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_family_agnostic_materializers.py -q`: 93 passed.
- `pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`: 180 passed.
- Queue validation passed for `.omx/research/codex_frontier_rate_attack_feedback_compiler_20260525T133356Z/dqs1_followup_queue.json`.
- False-authority scan found no truthy score/promotion/rank/dispatch/GPU authority under the live proof bundle.

## Remaining Gap

This closes the manual feedback-refresh loop for local materializer and DQS1
observations. The next high-value step is to run the generated follow-up queue
locally, harvest its observations automatically, and let this compiler re-enter
the loop without hand-picked paths.
