# Codex findings: materializer follow-up state custody

UTC: 2026-05-25T02:41:48Z

## Scope

This pass hardens the materializer-chain harvest and generated exact-readiness
follow-up path so queue-owned final-byte candidates cannot be harvested through
state-less explicit manifests or cross-queue stale success rows. It also
preserves adjacent observer health signal so orphaned execution state is visible
in automated runner summaries.

## Findings

- `--allow-unfinished-state` previously relaxed the entire state filter, so an
  explicit `--chain-manifest` plus `--state` could bypass work-id provenance.
- State-backed harvest without `--queue-id` loaded every queue in a SQLite state
  file; a stale row from another queue could satisfy work-id matching.
- Generated exact-readiness follow-up harvest commands carried only
  `--chain-manifest`, even though the execution queue knows the source work
  queue and state context needed to prove the manifest belongs to the succeeded
  local proof-chain row.
- Queue observation had useful drift/orphan/artifact information, but no single
  fail-closed health surface for runners to preserve it as durable signal.

## Landed integration

- `harvest_materializer_chain_manifests` now requires
  `experiment_queue_id` whenever a state filter is supplied.
- State provenance blockers run before the succeeded-status check, so
  `require_succeeded_state=False` still requires a work id and a matching state
  row when state filtering is active.
- Generated materializer exact-readiness harvest steps now include
  `--work-queue`, `--state`, and `--queue-id`; executable follow-up generation
  fails closed when the source work-queue path is missing.
- The queue builder and materializer campaign runner can pass an explicit
  materializer execution state path, defaulting to the canonical queue state.
- Experiment queue observation now emits `healthy`, `blockers`, and
  `blocker_count`, and the campaign runner asks the observer for orphan details.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_experiment_queue_observer.py -q`
  passed: 178 tests.
- `.venv/bin/python -m ruff check ...` on the touched scheduler, tool, and test
  files passed.
- `git diff --check` passed.

No score claim, rank/kill decision, or promotion authority is created here.
This is custody and no-orphan automation for local proof-chain harvests only;
contest CPU/CUDA exact eval remains required for any score movement.
