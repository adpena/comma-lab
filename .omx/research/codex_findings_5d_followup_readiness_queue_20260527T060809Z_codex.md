# 5D Follow-Up Readiness Queue Landing

UTC: 2026-05-27T06:08:09Z
Agent: codex

## Verdict

The 5D coverage-acquisition queue now owns a fail-closed readiness audit for
blocked follow-up requests instead of leaving exact-axis and MLX-negative-delta
handoffs as loose plan text.

## What Changed

- Added `tools/audit_5d_coverage_followup_requests.py`, a queue-callable CLI that
  scans emitted acquisition plans and writes `followup_readiness_report.json`.
- Added typed readiness rows for paired contest CPU/CUDA anchor requests and
  MLX local negative-delta requests.
- Wired the coverage-acquisition queue to run the readiness audit after plan
  emission and before the refresh/refire step.
- Expanded tests for missing-input blockers, ready command materialization, CLI
  output, and parent refresh queue shape.

## Authority

The readiness report is routing-only. It does not claim score, promote, rank,
kill, dispatch, or bypass paired auth eval. Missing submission bundles, MLX cache
manifests, or archive size inputs remain explicit blockers.

## Verification

- `.venv/bin/python -m ruff check --fix src/comma_lab/scheduler/pair_frame_5d_coverage_acquisition_queue.py tools/audit_5d_coverage_followup_requests.py src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py src/tac/tests/test_pair_frame_scorer_geometry_lattice_5d_canvas_coverage.py src/tac/tests/test_modal_paired_dispatch_contract.py -q`
- Live queue smoke at `/tmp/pact_5d_followup_readiness_smoke_20260527T060851Z`:
  validate returned `valid=true`, `experiment_count=7`, `step_count=10`; bounded
  worker executed 6 steps with `success_count=6`, `failure_count=0`, including
  `audit_blocked_followup_requests`.

Result: 20 tests passed; live readiness report emitted 2 fail-closed blocked
requests and 0 ready requests, matching the missing submission-bundle and MLX
cache/input blockers.
