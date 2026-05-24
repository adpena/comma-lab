# Codex Findings: Materializer Runner Auto Queue Identity

- UTC: 2026-05-24T18:25:22Z
- Lane: `codex_materializer_runner_auto_queue_identity_20260524`
- Scope: make queue-performance feedback immediately consumable by the next inverse-action replan without requiring ad hoc runtime/cache identity files.

## Findings

`queue_feedback_replan_request.json` could name the standalone performance
summary, but remained blocked unless the operator or a future agent supplied
`--queue-performance-runtime-identity` and
`--queue-performance-cache-identity`. That preserved correctness, but left the
closed-loop runner one manual step short.

The runner now generates local queue identities when explicit identities are not
provided:

- `queue_performance_runtime_identity.json`
- `queue_performance_cache_identity.json`

These artifacts are scoped to local queue telemetry only. The runtime identity
records the runner, scheduler, materializer, and inverse-cell code file hashes,
repo git SHA, Python executable/version, queue path, and runtime-policy paths.
The cache identity records the queue definition SHA, queue state path/SHA, and
the performance summary SHA. Both artifacts carry false-authority fields and are
forbidden from score, promotion, rank/kill, or dispatch authority.

With successful completed queue events, the replan request now becomes
`ready_for_action_functional_feedback=true` and includes a runnable
`tools/build_inverse_steganalysis_action_functional.py` command template with
the generated identity paths. Poisoned or empty performance summaries still
fail closed.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_inverse_surface_cells_compile_to_action_functional_work_queue -q`
- `.venv/bin/python tools/lane_maturity.py validate`

## Remaining Gap

The next scheduler improvement should let a successful materializer campaign
optionally enqueue the feedback action-functional command as a paused/follow-up
queue row, rather than only writing the command template.
