# Codex Findings: Queue False-Authority Observer Override

UTC: 2026-05-27T21:16:47Z

## Landing

Closed the follow-on observer edge created by queue artifacts that are themselves
`experiment_queue.v1` roots. Queue roots often do not carry root-level
`score_claim=false` fields, so queue-producing steps may now set
`required_false=[]` on `json_false_authority` postconditions while retaining the
default false-or-missing checks.

The observer now honors those explicit postcondition settings instead of
re-imposing root-level false-field requirements. It still recursively scans the
entire artifact and rejects any truthy authority field anywhere in nested
metadata, experiments, steps, or telemetry.

## Why This Matters

This keeps repair-waterfill and repair-score queues executable without turning
queue JSON into fake score authority:

- a queue artifact without root false fields can pass when it contains no
  truthy authority;
- a queue artifact with nested `metadata.score_claim=true` is rejected by the
  observer even if the worker postcondition passed.

## Verification

- `ruff check` on touched observer, tests, repair waterfill, and score queue
  files passed.
- `pytest src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`:
  49 passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`:
  57 passed.
