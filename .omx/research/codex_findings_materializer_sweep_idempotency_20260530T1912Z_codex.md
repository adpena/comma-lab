# Codex Findings - Materializer Sweep Idempotency - 2026-05-30T1912Z

## Scope

Follow-up from the queue-owned local drain pass. A wider bounded drain exposed
that rerunning materializer sweep steps over existing artifacts was not
idempotent even when the queued command passed `--allow-overwrite`.

## Bug Class

`tools/run_family_agnostic_materializer_sweep.py` accepted `--allow-overwrite`
but did not auto-supply `expected_existing_sha256` for existing row artifacts.
The lower-level artifact writer correctly refused the overwrite. In queue
execution this made crash/resume/rerun behavior fail with:

`expected_existing_sha256 is required before overwrite`

Affected surfaces included candidate archives, runtime-consumption proofs,
row manifests, sweep JSON, and observation JSONL.

## Fix

- Added explicit expected-existing SHA derivation for file artifacts when
  `--allow-overwrite` is set.
- Added expected tree SHA derivation for packet-member-merge receiver runtime
  rebuilds.
- Added an idempotent CLI regression test that runs the same archive repack
  sweep twice with `--allow-overwrite`.

## Live Proof

- Failure capture:
  - `.omx/research/queue_fleet_local_drain_exec_20260530T190601Z/fleet_local_drain_result.json`
  - first materializer execution queue supervisor child failed with the expected
    overwrite refusal.
- Recovery proof after fix:
  - `.omx/research/queue_fleet_local_drain_recovery_20260530T1910Z/current_cpu_materializer_exec/supervisor_result.json`
  - rewound the two failed steps and reran the queue to
    `final_reason=terminal_queue_state`, `status_counts={"succeeded": 3}`.

## Tests

- `.venv/bin/python -m ruff check tools/run_family_agnostic_materializer_sweep.py src/tac/tests/test_family_agnostic_materializer_sweep.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_queue_fleet_tool.py`

Result: 32 tests passed.

## Frontier

No contest score authority changed in this pass. This improves local automation
and restart safety for final-rate attack materializers only.
