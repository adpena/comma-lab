# Codex Findings - Queue Health Group Prior

Timestamp: 2026-05-25T03:25:29Z
Lane: `codex_queue_health_group_prior_20260525`

## Summary

Queue observation feedback was reaching inverse action planning, but repeated
queue-health failures were still mostly leaf-level blockers. The action
functional now preserves queue-health feedback as a grouped planning prior:
direct queue failures hard-block water-bucket selection, repeated source or
materializer groups expose repeat counts/group ids, and related materializer
failures apply a deterministic planning penalty without suppressing unrelated
materializers.

## Landing

- Added `inverse_steganalysis_queue_health_feedback.v1` to
  `src/tac/optimization/inverse_steganalysis_acquisition.py`.
- Threaded per-cell `queue_health_feedback`, `queue_health_group_ids`,
  `queue_health_repeat_count`, and `queue_health_penalty_applied` through the
  discrete action functional.
- Nested `queue_health_group_priors` under `observation_feedback` so existing
  consumers do not need a new queue artifact path to preserve the signal.
- Preserved optional queue-step `target_kind`, `materializer_id`, and
  `receiver_contract_kind` metadata when converting queue observations into
  inverse-steganalysis observations.

## Safeguards

- All queue-health feedback remains false-authority planning metadata.
- Direct candidate/source/work/backlog/global queue failures use a zero
  planning multiplier and cannot be water-bucket selected.
- Repeated related materializer/target failures use a count-derived multiplier
  (`1 / (1 + repeated_observation_count)`) and do not block unrelated
  materializers.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py -q`
  - `41 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - `155 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_acquisition.py`
  - `All checks passed`

## Remaining Gap

The scheduler-side dirty worktree currently contains an uncommitted recovery
queue builder. That should be reviewed and landed separately if it is intended
to turn required recovery actions into paused local queues. This landing does
not stage or overwrite those files.
