# Codex Findings - Repair Campaign Score Queue Autowire - 2026-05-27T18:45Z

## Summary

The final-rate feedback cycle now emits a queue-owned repair-campaign scoring
stage after the repair-budget waterfill queue. This closes a manual gap between
rate-only savings preservation and distortion-budget follow-up: generated
waterfill work orders now have a default downstream queue that scores the typed
repair ledger with `tools/score_repair_campaign.py`.

## Landed

- Added `comma_lab.scheduler.repair_campaign_score_queue` and
  `tools/build_repair_campaign_score_queue.py`.
- Wired `write_frontier_refresh_artifacts(...)` to write
  `repair_campaign_score_queue.json` whenever `repair_budget_waterfill_queue.json`
  is emitted.
- Added operator commands:
  - `validate_repair_campaign_score_queue`
  - `init_repair_campaign_score_queue`
  - `run_repair_campaign_score_queue_bounded_local`
- Added a prerequisite step
  `assert_repair_budget_waterfill_work_order_materialized` so the score queue
  can be generated before the waterfill queue has run. If run early, the queue
  fails clearly at the prerequisite rather than requiring a manual rebuild.
- Exported the queue builder and schemas from `comma_lab.scheduler`.

## Evidence

- `python -m pytest src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  passed: 57 tests.
- Stale runtime identity guard checks still pass:
  `test_observer_rejects_materializer_stale_runtime_tree_identity` and
  `test_harvest_rejects_chain_manifest_stale_runtime_tree_identity`.
- Generated artifact proof:
  `.omx/research/frontier_rate_attack_feedback_cycle_repair_score_queue_20260527T1840Z/frontier_rate_attack_feedback_cycle.json`
- Queue validation proof:
  `.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_rate_attack_feedback_cycle_repair_score_queue_20260527T1840Z/initial_refresh/repair_campaign_score_queue.json validate`
  returned valid with 3 experiments and 3 steps.

## Current Blocker Classification

The generated score queue is valid but all three experiments are frozen in this
specific no-auxiliary cycle because the upstream repair waterfill queue is also
frozen:

- missing prerequisite artifact: `targeted_component_correction_response_harvest`
- no materialized repair-budget waterfill work order path yet

An attempted auxiliary execution run at
`.omx/research/frontier_rate_attack_feedback_cycle_repair_score_queue_20260527T1830Z/`
failed before reaching this new queue because the receiver-repair worker could
not create under
`/Volumes/VertigoDataTier/experiments/results/frontier_rate_attack_feedback/repair_score_queue_20260527T1830Z`
with `Operation not permitted`. That is an external storage/write-permission
failure, not a repair-score queue schema failure.

## Authority

This landing is planning and queue automation only.

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- no Modal or CUDA dispatch was attempted

## Next Integration

The next closure is upstream: make
`targeted_component_correction_response_harvest` materialize in normal bounded
local cycles, then run:

1. `repair_budget_waterfill_queue`
2. `repair_campaign_score_queue`
3. MLX/local CPU advisory acquisition for the selected repair rows
4. exact-eval eureka dispatch only after the existing authority gates pass
