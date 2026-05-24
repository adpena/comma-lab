---
schema: codex_findings_v1
author: codex
created_at_utc: 2026-05-24T02:02:14Z
lane_id: codex_storage_ssh_policy_hardening_20260524T020214Z
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
---

# Codex Findings - Storage And SSH Policy Hardening

## Findings

- Scheduler storage preflight could emit an executable `move` cleanup step with
  no cold-store root. That would pass queue construction and fail later inside
  the cleanup command, wasting queue wall clock and obscuring the policy error.
- SSH dry-run initialized the requested queue state before returning a dry-run
  result. A read-only planning path must not insert, requeue, or otherwise
  mutate canonical queue state.
- SSH telemetry pullback treated all telemetry artifact paths as pullback
  authority. Recursive telemetry directories now need explicit
  `pullback_artifact_paths` and a recursive entry cap before they can be used
  as rsync sources.
- Remote git preflight ran before execution, but the execution call itself did
  not re-check the expected commit and clean-tree contract in the same remote
  shell invocation.
- Rsync pullback used a looser SSH transport than command execution and queue
  performance recorded remote command time without artifact pullback time.

## Landing

- Added `validate_scheduler_storage_preflight_config(...)` and wired it into
  DQS1, byte-shaving materializer queues, and the materializer campaign runner.
  `move` cleanup execution now requires at least one cold-store root.
- SSH dry-run now uses an existing SQLite state read-only, or an ephemeral
  temporary state when no canonical state exists. The requested state path is
  left untouched.
- Materializer telemetry now distinguishes observation paths from explicit
  pullback paths. Recursive telemetry paths without pullback authority are
  blocked instead of silently transferred.
- Remote execution commands now include the same HEAD and clean-tree guard as
  preflight, and rsync uses the same noninteractive SSH options.
- SSH terminal queue events now record total elapsed time including artifact
  pullback, plus `remote_elapsed_seconds` for attribution.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
  passed: 104 tests.
- Focused `ruff check`, `py_compile`, `git diff --check`, review-tracker
  policy check, and `tools/lane_maturity.py validate` passed.
- Live storage policy probe selected
  `/Volumes/VertigoDataTier/pact/experiments/results/materializer_next`,
  left APDataStore as second tier, and rejected local disk fallback.

## Boundaries

No score was claimed, no candidate was promoted, and no exact auth eval was
dispatched. These changes harden queue custody, storage policy, and SSH
execution semantics for the next larger local/remote materializer tranche.
