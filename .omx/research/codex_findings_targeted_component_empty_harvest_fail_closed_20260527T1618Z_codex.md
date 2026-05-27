# Codex Findings - Targeted Component Empty Harvest Fail-Closed Wiring

UTC: 2026-05-27T16:18Z

## Finding

The feedback cycle could produce a valid targeted-component acquisition with
zero queue-actionable rows, but `build_frontier_targeted_component_correction_queue`
returned `None`. Downstream waterfill then saw a missing response-harvest
artifact instead of a modeled blocker. After the first fix made the response
harvest explicit, a second false-ready surface appeared: repair waterfill still
queued work when the response harvest existed but had zero accepted rows and no
receiver-closed bytes.

## Landing

- Empty targeted-component selection now emits a frozen
  `experiment_queue.v1` with `selected_row_count=0`,
  `queue_actuation_ready=false`, and canonical blockers.
- `build_frontier_targeted_component_correction_response_harvest` now consumes
  the frozen queue and emits a first-class empty response harvest with
  `row_count=0` and `recommended_next_action=resolve_targeted_component_correction_queue_blockers`.
- Repair-budget waterfill now freezes when there are no accepted targeted
  component responses or no receiver-closed saved bytes, preserving the
  rate-only archive instead of inventing a distortion-spend lane.
- Repair-campaign scoring then freezes behind the waterfill blocker instead of
  presenting ready local scoring work.

## Verification

- `ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py`
  passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` passed:
  55 tests.
- Cycle proof:
  `.venv/bin/python tools/run_frontier_rate_attack_feedback_cycle.py --output-dir .omx/research/frontier_rate_attack_feedback_cycle_empty_targeted_harvest_20260527T1615Z --queue-id frontier_rate_attack_feedback_cycle_empty_targeted_harvest_20260527T1615Z --post-harvest-queue-id frontier_rate_attack_feedback_cycle_empty_targeted_harvest_post_20260527T1615Z --results-root /Volumes/VertigoDataTier/experiments/results/frontier_rate_attack_feedback/empty_targeted_harvest_20260527T1615Z --candidate-limit 2 --skip-raw-retention-plan --skip-mlx-retention-plan`

## Proof Artifacts

- `.omx/research/frontier_rate_attack_feedback_cycle_empty_targeted_harvest_20260527T1615Z/initial_refresh/targeted_component_correction_queue.json`
- `.omx/research/frontier_rate_attack_feedback_cycle_empty_targeted_harvest_20260527T1615Z/initial_refresh/targeted_component_correction_response_harvest.json`
- `.omx/research/frontier_rate_attack_feedback_cycle_empty_targeted_harvest_20260527T1615Z/initial_refresh/repair_budget_waterfill_queue.json`
- `.omx/research/frontier_rate_attack_feedback_cycle_empty_targeted_harvest_20260527T1615Z/initial_refresh/repair_campaign_score_queue.json`

Key proof values:

- targeted queue: first experiment `status=frozen`, `queue_actuation_ready=false`.
- response harvest: `row_count=0`,
  `recommended_next_action=resolve_targeted_component_correction_queue_blockers`.
- waterfill queue: first experiment `status=frozen`,
  `queue_actuation_blockers` includes
  `no_accepted_targeted_component_correction_responses`.
- score queue: `ready_experiment_count=0`, `blocked_experiment_count=2`.

## Authority

No score movement is claimed. This is a false-authority and automation-hardening
landing: it prevents an empty response surface from becoming implicit ready work
and preserves the rate-only candidate until measured component-response signal
exists.
