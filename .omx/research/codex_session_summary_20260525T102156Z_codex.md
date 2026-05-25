# Codex Session Summary - Test Profiling Fast Path - 2026-05-25T10:21:56Z

## Scope

Profile and reduce the slow relevant pytest slice for the materializer runner /
inverse-steganalysis queue work without weakening fail-closed authority coverage.

## Findings

- Baseline focused slice:
  `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py -q --durations=40`
  reported `238 passed in 14.97s`.
- Slow tests were concentrated in
  `src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`.
  The cost was repeated `tools/experiment_queue.py` subprocess/import startup
  for validate/init/control/worker/observe/performance orchestration, not MLX,
  SVD, ZIP mutation, or inverse-steganalysis math.

## Landed

- `tools/run_byte_shaving_materializer_campaign.py` now executes its own
  `tools/experiment_queue.py` control commands in-process when the interpreter
  and script match the runner-owned command shape.
- The fast path preserves `CommandResult` telemetry and captures stdout,
  stderr, return code, elapsed time, `SystemExit`, and unexpected exceptions in
  the same nonzero-result style used by subprocess execution.
- Other commands still use subprocess execution, including actual materializer
  commands run by the queue worker.
- Added
  `test_materializer_campaign_runner_executes_queue_cli_in_process` to prevent
  regression back to unnecessary subprocess startup.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_queue_cli_in_process src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_post_recovery_feedback_replan_emits_fresh_followup_and_continuation -q --durations=10`
- Re-run focused slice after the fast path: `239 passed in 7.93s`.

## Remaining Optimization Targets

- `test_byte_shaving_campaign_queue.py` still has repeated CLI subprocess smoke
  tests around inverse-scorer context/parity generation. Keep one subprocess
  sentinel and move duplicate semantic checks in-process.
- `test_family_agnostic_materializers.py` still pays small receiver-runtime
  smoke costs. Keep one actual receiver proof path and make duplicate structure
  checks direct.
