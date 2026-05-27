# Codex Findings: Activation Posterior Ingest

Generated: 2026-05-27T20:21Z

## Finding

Child-queue activation plans were being materialized as false-authority
artifacts and converted into blocked-learning signal reports, but the final
rate autoloop still required a human or a later one-off command to ingest those
signals into the repair-campaign posterior. That left frozen queue blockers
visible but not automatically compounding into acquisition policy.

During validation, a related custody bug surfaced: repair-campaign score queues
could be built with a custom posterior for scoring, while the later
`append_blocked_repair_campaign_learning_posterior` step still defaulted to the
repo-level posterior. Tests and isolated queues could therefore append
temp-path rows into `.omx/state/repair_campaign_stackability_posterior.jsonl`.
Finally, posterior rows were biasing repair-campaign scoring, but they did not
yet expose first-class acquisition follow-up routes. The next queue step still
had to infer whether a row meant targeted component response harvest,
receiver-closed byte credit, exact-auth handoff, or local MLX custody repair.

## Fix

- `frontier_final_rate_attack_autoloop` now appends activation
  `repair_campaign_blocked_learning_signal_report.v1` rows into the
  repair-campaign posterior with duplicate suppression.
- Deferred frozen child queues are ingested too, so a bounded run that spends
  its execution slots on runnable queues still preserves and learns from the
  blocked frozen work.
- Child-queue reports now distinguish selected and deferred posterior append
  report counts, total appended rows, and skipped duplicates.
- `repair_campaign_score_queue` now propagates custom posterior and lock paths
  into the blocked-posterior append step.
- The repair-campaign score queue CLI test now uses a temp posterior and asserts
  that append custody stays isolated.
- `repair_campaign_scorer` now folds posterior rows into typed acquisition
  follow-up routes keyed by recommended acquisition policy. These routes expose
  priority, activation action, queue artifact key, required evidence surface,
  affected families, typed response ids, and blocker/missing-artifact totals.

All artifacts remain false-authority: no score claim, no promotion, no
rank/kill, no budget-spend authority, and no exact-eval dispatch authority.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/comma_lab/scheduler/repair_campaign_score_queue.py src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_repair_campaign_score_queue.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py::test_post_feedback_child_queue_execution_preserves_deferred_frozen_plan src/tac/tests/test_frontier_rate_attack_bootstrap.py::test_post_feedback_child_queue_execution_classifies_frozen_child_queue -q`
  - Result: 2 passed in 0.65s.
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
  - Result: 30 passed in 4.39s.
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_repair_campaign_score_queue.py -q`
  - Result: 5 passed in 4.19s.
- `.venv/bin/ruff check src/comma_lab/scheduler/repair_campaign_score_queue.py src/tac/optimization/repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_scorer.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py -q`
  - Result: 13 passed in 4.45s.
- Bounded queue-owned smoke:
  `.venv/bin/python tools/build_frontier_final_rate_attack_queue.py --output-dir .omx/research/frontier_final_rate_attack_activation_posterior_smoke_20260527T2021Z --results-root /Volumes/VertigoDataTier/pact/frontier_final_rate_attack_activation_posterior_smoke_20260527T2021Z --queue-id frontier_final_rate_attack_activation_posterior_smoke_20260527T2021Z --max-steps 8 --max-parallel 2 --execute --post-execute-feedback-refresh --execute-post-feedback-queues --post-feedback-child-queue-limit 2 --post-feedback-child-queue-max-steps 1 --post-feedback-child-queue-max-parallel 1 --post-execute-feedback-candidate-limit 16 --post-execute-feedback-max-files-per-root 64`
  - Result: `failed_command_count=0`, feedback refresh return code `0`.
  - Executed runnable child queues: `operation_chain_compiler_queue` and
    `receiver_repair_queue`; both made progress and both observer
    revalidations were valid.
  - Deferred activation posterior append reports: 4.
  - Posterior rows appended from deferred activation plans: 10.
  - Posterior duplicate skips in this fresh smoke: 0.
  - Footprint: 2.7M under `.omx/research`, 1.3M under VertigoDataTier.

## Next Integration

The next queue-level step is to let the feedback-refresh builder consume
`acquisition_followup_routes` directly and thaw or generate the prerequisite
queues: targeted component response harvest, receiver-closed byte-credit
materialization, exact-auth handoff candidates, and grouped SegNet/PoseNet
waterfill chains.
