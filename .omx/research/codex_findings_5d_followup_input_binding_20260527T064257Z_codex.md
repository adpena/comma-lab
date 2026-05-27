# Codex Findings: 5D Follow-up Input Binding

UTC: 2026-05-27T06:42:57Z

## Summary

The 5D coverage acquisition lane now binds blocked follow-up requests to
concrete local inputs before it emits a refreshed readiness report and child
execution queue. This closes the schema-only gap: MLX follow-up rows no longer
accept an arbitrary `manifest.json`; they require tensor files, byte/hash
custody, pair-index consistency, false-authority fields, archive identity, and
candidate cache audit provenance before a local MLX follow-up command can be
materialized.

## What Changed

- Added `pair_frame_5d_canvas_coverage_followup_input_binding_report.v1`.
- Added `tools/bind_5d_coverage_followup_inputs.py`.
- Wired the acquisition queue to run input binding before readiness refresh.
- The refreshed readiness report feeds the child follow-up execution queue.
- Exact-axis follow-up rows still require a byte-closed submission bundle and
  remain operator-gated.
- MLX rows remain local research-signal only and false-authority throughout.
- Frontier refresh summaries now expose the input-binding report path and
  distinguish planned queue ownership from completed bounded local execution.
- The library refresh writer and standalone refresh CLI now expose the same
  follow-up input-binding and child-queue summary fields.

## Authority Contract

The binding report is not a score, dispatch, promotion, rank/kill, or
reproduction authority. It only discovers and validates local inputs for the
existing readiness gate. Candidate MLX caches must carry either a contest-auth
identity audit stamp or an allowed local-CPU advisory identity audit stamp, and
the downstream readiness report still carries false-authority fields.

## Verification

- `ruff check` on the changed scheduler/tool/test files passed.
- `pytest src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  passed: 75 tests.
- `pytest src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_production_contract.py -q`
  passed: 68 tests.
- `tools/lane_maturity.py validate` passed: 1428 lanes validated cleanly.

## Remaining Edge

The acquisition queue now owns the path from blocked request to input binding,
readiness refresh, child-queue emission, validation, and bounded local worker
launch. The next unresolved frontier is larger: feed real current refresh roots
with receiver-closed byte bundles and audited MLX cache directories so the
bounded worker drains actual negative-delta repair probes instead of merely
proving the queue mechanics.
