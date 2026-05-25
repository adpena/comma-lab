# Codex Session Summary - 2026-05-25T10:40:00Z

## Scope

Lane: `codex_materializer_runner_latency_20260525`

Focused on the materializer campaign runner test/worker latency that was making
local queue tests feel slower than the machine warrants.

## Changes Landed

- Lowered runner-owned queue-worker active polling from the generic
  `experiment_queue.py` 250 ms default to a campaign-local
  `DEFAULT_LOCAL_QUEUE_POLL_INTERVAL_SECONDS = 0.005`.
- Forwarded `--poll-interval-seconds` through the main materializer worker,
  feedback-replan followup worker, and queue-observation recovery worker.
- Lowered runner-owned control CLIs into an in-process execution path for:
  `tools/build_byte_shaving_campaign_queue.py`,
  `tools/build_inverse_steganalysis_action_functional.py`,
  `tools/build_mlx_acquisition_batch.py`,
  `tools/experiment_queue.py`, and
  `tools/plan_byte_shaving_campaign.py`.
- Added regression coverage proving default low polling, explicit zero-poll
  synthetic e2e tests, poll forwarding, and no Python subprocess fallback for
  runner-owned control tools.

## Verification

- `src/tac/tests/test_byte_shaving_materializer_campaign_runner.py` improved
  from `65 passed in 4.59s` before this pass to `66 passed in 2.87s`.
- Slowest packet handoff e2e improved from about `1.01s` to about `0.54s`.
- `ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
  passed.

## Residual Work

- The remaining slow e2e time is mostly real queue-step subprocess execution
  for materializer, harvest, and exact-dispatch-plan tools. Further reduction
  requires a deliberate trusted in-process Python step executor in
  `experiment_queue`, not another runner-level wrapper.
- Unrelated existing diffs were present in observer/acquisition files and were
  left untouched.
