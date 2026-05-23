# Codex Findings - Scorer-Response Normalized Planning Guard

UTC: 2026-05-23T07:13:25Z
Author: Codex
Lane: lane_codex_scorer_response_normalized_planning_guard_20260523

## Scope

Follow-on hardening after the DQS normalized-gain guard. This pass closes the
broader scorer-response planning boundary that consumes MLX rows for local
planning and exact-eval spend-triage routing.

## Finding

`scorer_response_dataset` already emitted normalized full-video scorer-gain,
projected full-video delta, break-even bytes, and byte-budget margin fields for
partial-window MLX rows. The next-probe planner preferred those fields when
present, but it trusted them as stored values. A stale generated dataset or
manual row could therefore route a raw partial-window gain through the
normalized planning surface and make the byte-margin / best-scorer selection
over-optimistic.

## Fix

`tac.optimization.scorer_response_dataset` now:

- computes normalized row-builder gain through
  `compute_normalized_full_video_gain(...)`;
- validates every newly emitted normalized objective row with
  `require_normalized_full_video_objective(...)`;
- validates MLX normalized-objective fields before planner helpers read
  projected full-video delta, normalized scorer gain, normalized break-even
  bytes, or normalized byte-budget margin;
- preserves native-row fallback only when no normalized objective output fields
  are present, so legacy/native rows are not accidentally coerced into the
  normalized full-video contract.

The new adversarial test builds an MLX row whose normalized gain is actually the
raw window gain and verifies `build_next_probe_plan(...)` fails closed with
`normalized_full_video_gain_mismatch`.

## Verification

Executed:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_normalized_objective.py

.venv/bin/python -m ruff check \
  src/tac/optimization/scorer_response_dataset.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/optimization/normalized_objective.py \
  src/tac/tests/test_normalized_objective.py

git diff --check
```

Results:

- `110 passed in 1.69s`
- `ruff`: all checks passed
- `git diff --check`: clean

## Next Integration

The next useful consumer is the queue/acquisition layer: candidate-queue rows
that carry MLX response-derived projected deltas should call this same planning
guard before they are admitted to local follow-up or exact-eval spend triage.
This keeps the normalized-objective truth source shared from response dataset
construction through DQS and into autonomous queue scheduling.
