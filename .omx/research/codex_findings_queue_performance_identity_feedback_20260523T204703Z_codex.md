# Codex Findings: Queue Performance Identity Feedback

UTC: 2026-05-23T20:47:03Z
Agent: Codex
Lane: codex_queue_performance_identity_feedback_20260523

## Summary

The queue performance summary was useful as raw timing telemetry, but it did
not preserve enough identity to feed the inverse-steganalysis action surface
without an external candidate map. That made the feedback loop fragile: a
bundle, materializer work row, backlog key, or source unit could lose its
candidate identity before reaching acquisition.

## Change

`queue_performance_summary(...)` now exports identity maps and bucket-level
identity fields:

- `candidate_id_by_experiment`
- `work_id_by_experiment`
- `backlog_key_by_experiment`
- `source_unit_ids_by_experiment`
- `source_selection_ids_by_experiment`
- `by_experiment`
- `by_work_id`
- `by_backlog_key`
- `by_source_unit_id`
- `by_source_selection_id`

The inverse-steganalysis acquisition reader now consumes embedded
`candidate_id_by_experiment`, rejects anonymous non-empty summaries, and allows
an explicit candidate map only as a legacy override. If both embedded and
explicit maps are provided, mismatches fail closed.

## Why It Matters

This turns scheduler timing into reusable solver signal. Queue execution cost
can now be attributed back to the candidate/work/source-unit coordinates that
the action functional uses, instead of silently falling back to experiment IDs.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py`
