# Codex Findings: Byte Shaving Queue CLI Test Latency

UTC: 2026-05-25T11:00:19Z
Agent: Codex
Lane: `codex_byte_shaving_queue_cli_test_latency_20260525`
Authority: test-infrastructure only; no score claim, promotion claim, rank/kill authority, or exact-eval dispatch authority

## Scope

Finished and hardened the in-process CLI harness for
`test_byte_shaving_campaign_queue.py`. The queue-builder tests now call
`tools/build_byte_shaving_campaign_queue.py` through its `main()` function
instead of paying Python subprocess startup cost for every internal queue
fixture that is not specifically proving process-boundary behavior.

## Findings And Fixes

- Added a captured-output `_run_queue_tool(...)` helper preserving the subset of
  `subprocess.run(..., text=True, capture_output=True, check=...)` semantics used
  by the test file.
- Converted the DQS1 queue writer and repeated queue-builder CLI test calls to
  the helper.
- Left generated materializer smoke commands as subprocesses because those rows
  intentionally prove the produced command lines are executable from the real
  tool boundary.

## Measured Effect

Initial profile from this turn:

- combined materializer/acquisition/queue/action slice before the
  dynamic-sparse-gate oracle test entered the proof surface:
  `262 passed in 7.81s`
- after the action-functional CLI helper already present on `main`:
  `262 passed in 6.19s`
- after this queue-builder helper:
  `265 passed in 5.41s` including the dynamic-sparse-gate oracle proof surface
  that landed immediately before this pass.

Net slice improvement against the initial profile: about 30.7% wall-clock
reduction while carrying the newer 265-test proof surface.

## Verification

- `.venv/bin/python -m ruff check src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_byte_shaving_campaign_queue.py --no-cache`
  - Result: pass
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q --durations=20 --durations-min=0.01`
  - Result: `77 passed in 0.73s`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q --durations=30 --durations-min=0.01`
  - Result: `265 passed in 5.41s`

## Remaining Hotspots

The remaining visible costs are integration tests that exercise real runner,
queue, runtime-policy, recovery, and handoff surfaces:

- `test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff`
  around 0.75s;
- `test_materializer_campaign_runner_executes_no_paid_packet_member_handoff`
  around 0.51s;
- `test_post_recovery_feedback_replan_emits_fresh_followup_and_continuation`
  around 0.38s.

## Remaining Work

- Reuse this harness pattern for additional repo CLI-heavy tests after profiling
  identifies files where process startup still dominates wall-clock.
