# 5D Follow-Up Self-Execution Queue Wiring

## Verdict

`PROCEED`: the 5D coverage acquisition queue no longer stops at emitting a
follow-up execution queue.  It now validates and runs that child queue through a
bounded local worker, so MLX scorer-response follow-up work can execute as part
of the same queue-owned DAG.

## What Changed

- The `audit_blocked_followup_requests` experiment now emits
  `followup_execution_queue.json`, validates it, and runs a bounded local worker
  over it.
- The child queue still freezes exact-axis paired-auth rows, so the automatic
  worker only runs locally queued false-authority work such as MLX negative-delta
  scorer-response acquisition.
- The worker result is preserved at
  `followup_execution_worker_result.json` and must report
  `experiment_queue_worker_result.v1` with `failure_count == 0`.
- Frontier refresh reports now surface deterministic child queue and worker
  result paths and include after-acquisition validate/run commands for manual
  reruns.
- The lane `lane_pair_frame_5d_coverage_followup_execution_20260527` is
  registered and marked L1 for implementation plus memory-entry coverage.

## Authority

This remains encoder-side research-signal plumbing.  The queue has no score,
promotion, rank/kill, paid dispatch, or exact-eval authority.  Contest CPU/CUDA
authority is still a separate exact-auth path.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/pair_frame_5d_coverage_acquisition_queue.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`

All three passed before commit.
