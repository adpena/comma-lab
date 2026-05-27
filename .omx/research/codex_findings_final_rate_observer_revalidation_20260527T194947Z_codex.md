# Codex Findings - Final Rate Observer Revalidation

Generated: 2026-05-27T19:49:47Z

## Summary

The final-rate post-feedback autoloop now treats observer output as custody
evidence that must match the child queue it claims to observe. A successful
worker run is no longer enough for the parent campaign to record progress when
the observer payload is stale, missing, malformed, or from another queue.

## Engineering Outcome

- Added a shared scheduler JSON identity helper for stable queue/policy hashes.
- Moved queue observer and feedback-replan policy hashing onto that shared
  helper to avoid hash-rule drift.
- Child queue runs now record queue SHA-256, observer revalidation details,
  observer artifact SHA-256, and aggregate observer-revalidation failure counts.
- Observer payloads must match the expected schema, queue id, stable queue hash,
  and read-only observer flag.
- Observer revalidation failures force `progress_made=false` and add explicit
  progress blockers, preserving no-progress signal for parent campaigns.

## Validation

- `.venv/bin/ruff check src/comma_lab/scheduler/json_identity.py src/comma_lab/scheduler/experiment_queue_observer.py src/comma_lab/scheduler/queue_feedback_replan_policy.py src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/tac/tests/test_frontier_rate_attack_bootstrap.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/json_identity.py src/comma_lab/scheduler/experiment_queue_observer.py src/comma_lab/scheduler/queue_feedback_replan_policy.py src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/tac/tests/test_frontier_rate_attack_bootstrap.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_queue_feedback_replan_policy.py -q`

Final focused result: `89 passed`.

## Next Integration

Run the queue-owned final-rate campaign again with this observer gate. Any child
queue that cannot produce a fresh matching observer artifact should remain a
repair target, not a progress signal.
