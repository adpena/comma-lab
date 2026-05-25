# Codex Findings: Action Functional CLI Test Latency

UTC: 2026-05-25T10:57:00Z
Agent: Codex
Lane: `codex_action_functional_cli_test_latency_20260525`

## Scope

Hardened an in-tree latency optimization for action-functional CLI tests. The
test file now dispatches repository CLIs by calling their `main()` functions
in-process instead of spawning Python subprocesses for every fixture.

## Findings And Fixes

- Added a small captured-output `_run_repo_tool(...)` test helper that preserves
  the subprocess-style return object shape used by existing assertions.
- Routed `tools/build_inverse_steganalysis_action_functional.py` and
  `tools/build_mlx_acquisition_batch.py` test calls through the helper.
- Preserved failure behavior for `check=True` and captured `stdout`/`stderr`.
- Removed the last lingering subprocess call and fixed import ordering so the
  file passes ruff.

## Verification

- `.venv/bin/python -m ruff check src/tac/tests/test_inverse_steganalysis_action_functional_cli.py`
  - Result: pass
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q --durations=20`
  - Result: `16 passed in 0.29s`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q --durations=20`
  - Result: `265 passed in 6.21s`

## Remaining Work

- Apply the same in-process test harness pattern to other CLI-heavy unit slices
  that still spend most of their time in Python startup and import cost.
